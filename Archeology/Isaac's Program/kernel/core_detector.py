"""
G₂ Structure Detector - Core Detection Orchestrator

This module provides the main detection orchestrator that coordinates φ⁰ detection
with parallel feature module execution and recursive aggregation.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    from .aggregator import RecursiveDetectionAggregator, AggregationResult, StreamingDetectionAggregator, StreamingAggregationResult
    from .modules import feature_registry, FeatureResult
    from .detector_profile import DetectorProfile, DetectorProfileManager, StructureType
except ImportError:
    # Fallback for direct execution
    from aggregator import RecursiveDetectionAggregator, AggregationResult, StreamingDetectionAggregator, StreamingAggregationResult
    from modules import feature_registry, FeatureResult
    from detector_profile import DetectorProfile, DetectorProfileManager, StructureType

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ElevationPatch:
    """Container for elevation data and metadata - independent from phi0_core"""
    elevation_data: np.ndarray
    lat: float = None
    lon: float = None
    source: str = "unknown"
    resolution_m: float = 0.5
    coordinates: Tuple[float, float] = None
    patch_size_m: float = None
    metadata: Dict = None


@dataclass
class G2DetectionResult:
    """Enhanced detection result with G₂-level reasoning - independent system"""
    detected: bool
    confidence: float
    final_score: float
    base_score: float  # G₂ base score (replaces phi0_score)
    aggregation_result: AggregationResult
    feature_results: Dict[str, FeatureResult]
    refinement_attempts: int = 0
    refinement_history: List[AggregationResult] = None
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    @property
    def feature_scores(self) -> Dict[str, float]:
        """Backward compatibility property to extract scores from feature_results"""
        if not self.feature_results:
            return {}
        
        scores = {}
        for name, result in self.feature_results.items():
            if hasattr(result, 'score'):
                scores[name] = float(result.score)
            elif isinstance(result, (int, float)):
                scores[name] = float(result)
            else:
                scores[name] = 0.0
        return scores


class G2StructureDetector:
    """
    G₂-level structure detector with recursive geometric reasoning.
    
    Combines φ⁰ core detection with parallel feature module execution
    and recursive refinement capabilities. Now supports profile-driven configuration.
    """
    
    def __init__(self, 
                 profile: Optional[DetectorProfile] = None,
                 profile_name: Optional[str] = None,
                 # Legacy parameters for backward compatibility (deprecated)
                 resolution_m: Optional[float] = None,
                 structure_radius_m: Optional[float] = None,
                 structure_type: Optional[str] = None,
                 max_workers: Optional[int] = None,
                 enable_refinement: Optional[bool] = None,
                 max_refinement_attempts: Optional[int] = None):
        """
        Initialize G₂ detector with profile-driven configuration
        
        Args:
            profile: DetectorProfile object with complete configuration
            profile_name: Name of template profile to load
            
            # Legacy parameters (deprecated, use profile instead):
            resolution_m: Resolution in meters per pixel
            structure_radius_m: Expected structure radius in meters
            structure_type: Type of structure to detect
            max_workers: Maximum number of parallel workers for feature modules
            enable_refinement: Whether to enable recursive refinement
            max_refinement_attempts: Maximum number of refinement attempts
        """
        # Initialize profile manager
        self.profile_manager = DetectorProfileManager()
        
        # Handle profile resolution with three initialization modes
        self.profile = self._resolve_profile(
            profile, profile_name, 
            resolution_m, structure_radius_m, structure_type, 
            max_workers, enable_refinement, max_refinement_attempts
        )
        
        # Check if profile was successfully created
        if self.profile is None:
            raise RuntimeError("Could not create a valid detector profile. Please provide a profile explicitly.")
        
        # Extract core parameters from profile
        self.resolution_m = self.profile.geometry.resolution_m
        self.structure_radius_m = self.profile.geometry.structure_radius_m
        self.structure_type = self.profile.structure_type.value
        self.detection_threshold = self.profile.thresholds.detection_threshold
        
        # Extract G₂ specific parameters from profile
        self.max_workers = self.profile.max_workers
        self.enable_refinement = self.profile.enable_refinement
        self.max_refinement_attempts = self.profile.max_refinement_attempts
        
        # Initialize feature modules using profile configuration
        enabled_features = self.profile.get_enabled_features()
        
        # Map profile feature names to actual module names for weights
        feature_name_mapping = {
            'histogram': 'ElevationHistogram',
            'entropy': 'ElevationEntropy', 
            'dropoff': 'DropoffSharpness',
            'volume': 'Volume',
            'compactness': 'Compactness',
            'planarity': 'Planarity'
        }
        
        module_weights = {}
        for profile_name, config in enabled_features.items():
            module_name = feature_name_mapping.get(profile_name, profile_name)
            module_weights[module_name] = config.weight
        
        self.feature_modules = feature_registry.get_all_modules(module_weights)
        
        # Set parameters for all modules using profile geometry
        structure_radius_px = self.profile.geometry.get_structure_radius_px()
        for module in self.feature_modules.values():
            module.set_parameters(self.resolution_m, structure_radius_px)
        
        # Configure modules with their specific parameters from profile
        self._configure_modules_from_profile()
        
        # Extract polarity preferences from profile feature configurations
        polarity_preferences = {}
        for profile_name, feature_config in enabled_features.items():
            if feature_config.polarity_preference is not None:
                # Map profile feature name to actual module name using same mapping
                module_name = feature_name_mapping.get(profile_name, profile_name)
                polarity_preferences[module_name] = feature_config.polarity_preference
        
        # Initialize aggregator using profile thresholds and polarity preferences
        self.aggregator = StreamingDetectionAggregator(
            base_score=0.5,  # G₂ neutral starting score
            early_decision_threshold=self.profile.thresholds.early_decision_threshold,
            min_modules_for_decision=self.profile.thresholds.min_modules_for_decision,
            polarity_preferences=polarity_preferences
        )
        
        logger.info(f"G₂ detector initialized with profile '{self.profile.name}' - "
                   f"{len(self.feature_modules)} feature modules enabled")
    
    def _resolve_profile(self, 
                        profile: Optional[DetectorProfile], 
                        profile_name: Optional[str],
                        resolution_m: Optional[float],
                        structure_radius_m: Optional[float], 
                        structure_type: Optional[str],
                        max_workers: Optional[int],
                        enable_refinement: Optional[bool],
                        max_refinement_attempts: Optional[int]) -> DetectorProfile:
        """
        Resolve detector profile using three initialization modes:
        1. Use provided profile directly
        2. Load profile by name  
        3. Create profile from legacy parameters or use default
        """
        # Mode 1: Use existing profile
        if profile is not None:
            logger.info(f"Using provided profile: '{profile.name}'")
            return profile
        
        # Mode 2: Load profile by name
        if profile_name is not None:
            try:
                profile = self.profile_manager.load_template(f"{profile_name}.json")
                logger.info(f"Loaded profile template: '{profile_name}'")
                return profile
            except FileNotFoundError:
                logger.warning(f"Profile template '{profile_name}' not found, creating default")
        
        # Mode 3: Create from legacy parameters or use default
        if any(param is not None for param in [resolution_m, structure_radius_m, structure_type]):
            # Issue deprecation warning for legacy usage
            warnings.warn(
                "Using individual parameters is deprecated. "
                "Please use DetectorProfile for better configuration management.",
                DeprecationWarning,
                stacklevel=3
            )
            
            # Create profile from legacy parameters
            profile = self._create_profile_from_legacy_params(
                resolution_m, structure_radius_m, structure_type,
                max_workers, enable_refinement, max_refinement_attempts
            )
            logger.info("Created profile from legacy parameters")
            return profile
        
        # No parameters provided - create a basic default profile
        try:
            from .detector_profile import DetectorProfile, StructureType
            profile = DetectorProfile(
                name="Basic Default Profile",
                description="Auto-generated basic profile for G₂ detector",
                structure_type=StructureType.GENERIC
            )
            logger.info("Created basic default profile")
            return profile
        except Exception as e:
            logger.error(f"Failed to create default profile: {e}")
            # If even basic profile creation fails, provide minimal fallback
            logger.warning("Using minimal fallback configuration")
            return None
    
    def _create_profile_from_legacy_params(self,
                                          resolution_m: Optional[float],
                                          structure_radius_m: Optional[float], 
                                          structure_type: Optional[str],
                                          max_workers: Optional[int],
                                          enable_refinement: Optional[bool],
                                          max_refinement_attempts: Optional[int]) -> DetectorProfile:
        """Create a DetectorProfile from legacy individual parameters"""
        from .detector_profile import GeometricParameters, DetectionThresholds, FeatureConfiguration, DetectorProfile, StructureType
        
        # Start with basic default profile and override with provided parameters
        profile = DetectorProfile(
            name="Legacy Parameter Profile",
            description="Created from individual constructor parameters",
            structure_type=StructureType.GENERIC
        )
        
        # Override geometry parameters
        if resolution_m is not None:
            profile.geometry.resolution_m = resolution_m
        if structure_radius_m is not None:
            profile.geometry.structure_radius_m = structure_radius_m
            
        # Override structure type
        if structure_type is not None:
            try:
                profile.structure_type = StructureType(structure_type)
            except ValueError:
                logger.warning(f"Unknown structure type '{structure_type}', using GENERIC")
                profile.structure_type = StructureType.GENERIC
        
        # Override execution parameters
        if max_workers is not None:
            profile.max_workers = max_workers
        if enable_refinement is not None:
            profile.enable_refinement = enable_refinement
        if max_refinement_attempts is not None:
            profile.max_refinement_attempts = max_refinement_attempts
            
        return profile
    
    def run_feature_modules_streaming(self, elevation_patch: np.ndarray,
                                     callback=None) -> Dict[str, FeatureResult]:
        """
        Run feature modules with streaming aggregation - results processed as they complete
        
        Args:
            elevation_patch: 2D elevation data array
            callback: Optional callback function called for each completed module
                     Signature: callback(module_name, result, streaming_aggregation)
            
        Returns:
            Dictionary mapping module names to their results
        """
        results = {}
        completed_modules = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all feature computations
            future_to_module = {
                executor.submit(module.compute, elevation_patch): name
                for name, module in self.feature_modules.items()
            }
            
            logger.info(f"🚀 Started {len(future_to_module)} feature modules in parallel")
            
            # Process results as they complete
            for future in as_completed(future_to_module):
                module_name = future_to_module[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per module
                    results[module_name] = result
                    completed_modules.append(module_name)
                    
                    # Add evidence to aggregator immediately
                    if module_name in self.feature_modules:
                        weight = self.feature_modules[module_name].weight
                        self.aggregator.add_evidence(module_name, result, weight)
                    
                    # Perform streaming aggregation
                    streaming_result = self.aggregator.streaming_aggregate(
                        available_modules=completed_modules,
                        total_modules=len(self.feature_modules)
                    )
                    
                    logger.info(f"✅ Module {module_name} completed: score={result.score:.3f}")
                    logger.info(f"📊 Streaming: {streaming_result.completion_percentage:.1%} complete, "
                               f"confidence={streaming_result.streaming_confidence:.3f}")
                    
                    # Call callback if provided
                    if callback:
                        callback(module_name, result, streaming_result)
                    
                    # Check for early decision
                    if streaming_result.early_decision_possible:
                        logger.info(f"🎯 Early decision possible! "
                                   f"Score: {streaming_result.final_score:.3f}, "
                                   f"Confidence: {streaming_result.streaming_confidence:.3f}")
                        
                        # Could optionally cancel remaining futures here for even faster execution
                        # for remaining_future in future_to_module:
                        #     if not remaining_future.done():
                        #         remaining_future.cancel()
                    
                except Exception as e:
                    logger.warning(f"❌ Module {module_name} failed: {e}")
                    results[module_name] = FeatureResult(
                        score=0.0,
                        valid=False,
                        reason=f"Computation failed: {str(e)}"
                    )
        
        logger.info(f"🏁 All feature modules completed: {len(results)} results")
        return results
    
    def detect_structure(self, elevation_patch: ElevationPatch) -> G2DetectionResult:
        """
        Perform G₂-level structure detection with recursive reasoning
        
        Args:
            elevation_patch: ElevationPatch object with elevation data
            
        Returns:
            G2DetectionResult with comprehensive detection analysis
        """
        logger.info("Starting G₂ structure detection")
        
        # Reset aggregator for new detection
        self.aggregator.reset()
        
        # In the new G₂ system, we start with neutral base score
        # Feature modules (especially histogram) drive the detection
        base_score = 0.5
        logger.info(f"G₂ base score: {base_score:.3f} (feature-driven detection)")
        
        # Step 2: Run feature modules in parallel  
        feature_results = self.run_feature_modules_parallel(elevation_patch.elevation_data)
        
        # Step 3: Set up streaming aggregator and add evidence
        self.aggregator.set_expected_modules(len(self.feature_modules))
        
        for name, result in feature_results.items():
            if name in self.feature_modules:
                weight = self.feature_modules[name].weight
                self.aggregator.add_evidence(name, result, weight)
        
        # Use streaming aggregation (compatible with both streaming and batch modes)
        aggregation_result = self.aggregator.aggregate_streaming()
        logger.info(f"Initial aggregation: score={aggregation_result.final_score:.3f}, confidence={aggregation_result.confidence:.3f}")
        
        # Step 4: Recursive refinement if enabled and needed
        refinement_attempts = 0
        refinement_history = [aggregation_result]
        
        # Recursive refinement is supported by RecursiveDetectionAggregator
        if self.enable_refinement:
            logger.info("Starting recursive refinement process")
        
        # Step 5: Make final detection decision
        detection_threshold = self.detection_threshold
        confidence_threshold = self.profile.thresholds.confidence_threshold
        detected = (aggregation_result.final_score >= detection_threshold and 
                   aggregation_result.confidence >= confidence_threshold)
        
        # Generate comprehensive result
        result = G2DetectionResult(
            detected=detected,
            confidence=aggregation_result.confidence,
            final_score=aggregation_result.final_score,
            base_score=base_score,
            aggregation_result=aggregation_result,
            feature_results=feature_results,
            refinement_attempts=refinement_attempts,
            refinement_history=refinement_history,
            reason=f"G₂ detection: {aggregation_result.reason}",
            metadata={
                "base_score": base_score,
                "detection_threshold": detection_threshold,
                "confidence_threshold": confidence_threshold,
                "structure_type": self.structure_type,
                "feature_module_count": len(self.feature_modules)
            }
        )
        
        logger.info(f"G₂ detection completed: detected={detected}, confidence={aggregation_result.confidence:.3f}")
        return result
    
    def _simulate_refinement(self, base_result: AggregationResult, strategy: Dict[str, Any]) -> Optional[AggregationResult]:
        """
        Simulate refinement process (placeholder for actual implementation)
        
        In a full implementation, this would:
        - Adjust detection parameters based on strategy
        - Re-run detection with new parameters
        - Return new aggregation result
        
        For the skeleton, we simulate slight improvements
        """
        # Simulate small improvement in confidence
        simulated_score = base_result.final_score + np.random.normal(0, 0.05)
        simulated_confidence = base_result.confidence + np.random.normal(0.02, 0.03)
        
        # Clamp values
        simulated_score = max(0.0, min(1.0, simulated_score))
        simulated_confidence = max(0.0, min(1.0, simulated_confidence))
        
        return AggregationResult(
            final_score=simulated_score,
            confidence=simulated_confidence,
            phi0_contribution=base_result.phi0_contribution,
            feature_contribution=base_result.feature_contribution,
            evidence_count=base_result.evidence_count,
            reason=f"Refined: {base_result.reason}",
            metadata={**base_result.metadata, "refinement_applied": strategy}
        )
    
    def get_module_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded feature modules"""
        return {
            name: {
                "class": module.__class__.__name__,
                "weight": module.weight,
                "resolution_m": module.resolution_m,
                "structure_radius_px": module.structure_radius_px
            }
            for name, module in self.feature_modules.items()
        }
    
    def configure_module_weights(self, weights: Dict[str, float]):
        """Configure weights for feature modules"""
        for name, weight in weights.items():
            if name in self.feature_modules:
                self.feature_modules[name].weight = weight
                logger.info(f"Updated weight for {name}: {weight}")
    
    def configure_aggregator(self, phi0_weight: float = None, feature_weight: float = None):
        """Configure aggregator weights"""
        if phi0_weight is not None:
            self.aggregator.phi0_weight = phi0_weight
        if feature_weight is not None:
            self.aggregator.feature_weight = feature_weight
        
        # Ensure weights sum to 1.0
        total = self.aggregator.phi0_weight + self.aggregator.feature_weight
        if total > 0:
            self.aggregator.phi0_weight /= total
            self.aggregator.feature_weight /= total
        
        logger.info(f"Aggregator weights: φ⁰={self.aggregator.phi0_weight:.2f}, features={self.aggregator.feature_weight:.2f}")
    
    def register_feature_module(self, name: str, module_class, weight: float = 1.0):
        """
        Dynamically register a new feature module
        
        Args:
            name: Unique name for the module
            module_class: Class inheriting from BaseFeatureModule
            weight: Weight for this module
        """
        feature_registry.register(name, module_class, weight)
        # Refresh the feature modules
        self.feature_modules[name] = feature_registry.get_module(name, weight)
        
        # Set parameters for the new module
        structure_radius_px = int(self.phi0_detector.structure_radius_m / self.phi0_detector.resolution_m)
        self.feature_modules[name].set_parameters(self.phi0_detector.resolution_m, structure_radius_px)
        
        logger.info(f"Registered and loaded feature module: {name}")
    
    def unregister_feature_module(self, name: str):
        """
        Unregister a feature module
        
        Args:
            name: Name of the module to unregister
        """
        feature_registry.unregister(name)
        if name in self.feature_modules:
            del self.feature_modules[name]
        logger.info(f"Unregistered feature module: {name}")
    
    def list_available_modules(self) -> List[str]:
        """List all available feature modules"""
        return feature_registry.list_modules()
    
    
    def detect_structure_streaming(self, elevation_patch: ElevationPatch, 
                                  progress_callback=None) -> G2DetectionResult:
        """
        Perform G₂-level structure detection with real-time streaming aggregation
        
        Args:
            elevation_patch: ElevationPatch object with elevation data
            progress_callback: Optional callback for streaming progress updates
                             Signature: callback(module_name, result, streaming_aggregation)
            
        Returns:
            G2DetectionResult with comprehensive detection analysis
        """
        logger.info("🌊 Starting G₂ streaming structure detection")
        
        # Reset aggregator for new detection
        self.aggregator.reset()
        
        # Set expected modules count for streaming aggregator
        self.aggregator.set_expected_modules(len(self.feature_modules))
        
        # NOTE: In the new G₂ system, we don't need a separate φ⁰ score
        # The histogram module provides the core pattern matching functionality
        # Set base score to neutral (0.5) - let feature modules drive the decision
        base_score = 0.5
        logger.info(f"G₂ base score: {base_score:.3f} (feature-driven detection)")
        self.aggregator.set_phi0_score(base_score)
        
        # Step 2: Run feature modules with streaming aggregation
        streaming_results = []
        
        def streaming_callback(module_name, result, streaming_agg):
            """Internal callback to collect streaming results"""
            streaming_results.append({
                'module': module_name,
                'result': result,
                'aggregation': streaming_agg,
                'timestamp': len(streaming_results)
            })
            
            # Call user callback if provided
            if progress_callback:
                progress_callback(module_name, result, streaming_agg)
        
        feature_results = self.run_feature_modules_streaming(
            elevation_patch.elevation_data, 
            callback=streaming_callback
        )
        
        # Step 3: Final aggregation
        final_aggregation = self.aggregator.aggregate()
        logger.info(f"Final aggregation: score={final_aggregation.final_score:.3f}, "
                   f"confidence={final_aggregation.confidence:.3f}")
        
        # Step 4: Make detection decision
        detection_threshold = self.detection_threshold
        confidence_threshold = self.profile.thresholds.confidence_threshold
        detected = (final_aggregation.final_score >= detection_threshold and 
                   final_aggregation.confidence >= confidence_threshold)
        # Generate comprehensive result
        result = G2DetectionResult(
            detected=detected,
            confidence=final_aggregation.confidence,
            final_score=final_aggregation.final_score,
            base_score=base_score,  # G₂ base score instead of phi0_score
            aggregation_result=final_aggregation,
            feature_results=feature_results,
            refinement_attempts=0,  # No refinement in streaming mode
            refinement_history=[final_aggregation],
            reason=f"G₂ streaming detection: {final_aggregation.reason}",
            metadata={
                "base_score": base_score,
                "detection_threshold": detection_threshold,
                "confidence_threshold": confidence_threshold,
                "structure_type": self.structure_type,
                "feature_module_count": len(self.feature_modules),
                "streaming_results": streaming_results,
                "early_decision_points": [sr for sr in streaming_results 
                                        if sr['aggregation'].early_decision_possible]
            }
        )
        
        logger.info(f"🌊 G₂ streaming detection completed: detected={detected}, "
                   f"confidence={final_aggregation.confidence:.3f}")
        return result
    
    def run_feature_modules_parallel(self, elevation_patch: np.ndarray) -> Dict[str, FeatureResult]:
        """
        Run all feature modules in parallel (traditional batch mode)
        
        Args:
            elevation_patch: 2D elevation data array
            
        Returns:
            Dictionary mapping module names to their results
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all feature computations
            future_to_module = {
                executor.submit(module.compute, elevation_patch): name
                for name, module in self.feature_modules.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_module):
                module_name = future_to_module[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per module
                    results[module_name] = result
                    logger.debug(f"Module {module_name} completed: score={result.score:.3f}")
                except Exception as e:
                    logger.warning(f"Module {module_name} failed: {e}")
                    results[module_name] = FeatureResult(
                        score=0.0,
                        valid=False,
                        reason=f"Computation failed: {str(e)}"
                    )
        
        return results
    
    def _configure_modules_from_profile(self):
        """Configure feature modules with their specific parameters from the profile"""
        enabled_features = self.profile.get_enabled_features()
        
        for module_name, config in enabled_features.items():
            if module_name in self.feature_modules and config.parameters:
                module = self.feature_modules[module_name]
                
                # Special handling for histogram module - load reference kernel if specified
                if module_name == 'histogram' and 'reference_kernel_path' in config.parameters:
                    try:
                        import pickle
                        from pathlib import Path
                        
                        kernel_path = config.parameters['reference_kernel_path']
                        full_path = Path(kernel_path)
                        
                        if full_path.exists():
                            with open(full_path, 'rb') as f:
                                kernel_data = pickle.load(f)
                            
                            if 'elevation_kernel' in kernel_data:
                                reference_kernel = kernel_data['elevation_kernel']
                                module.set_reference_kernel(reference_kernel)
                                logger.info(f"Loaded reference kernel from {kernel_path} for histogram module")
                            else:
                                logger.warning(f"No elevation_kernel found in {kernel_path}")
                        else:
                            logger.warning(f"Reference kernel file not found: {kernel_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load reference kernel: {e}")
                
                # Call the module's configure method with the profile parameters
                try:
                    logger.info(f"Configuring {module_name} module with parameters: {list(config.parameters.keys())}")
                    module.configure(**config.parameters)
                    logger.info(f"Successfully configured {module_name} module with {len(config.parameters)} parameters")
                except Exception as e:
                    logger.error(f"Failed to configure {module_name} module: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with other modules even if one fails
