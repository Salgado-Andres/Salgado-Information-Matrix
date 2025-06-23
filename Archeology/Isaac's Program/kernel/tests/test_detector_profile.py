#!/usr/bin/env python3
"""
Test the Detector Profile System

Demonstrates creating, configuring, and using detector profiles
for different archaeological detection scenarios.
"""

import sys
import os
import numpy as np

# Add the kernel directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_profile import (
    DetectorProfile, 
    DetectorProfileManager,
    StructureType,
    PatchShape,
    GeometricParameters,
    FeatureConfiguration,
    DetectionThresholds
)

def test_profile_creation():
    """Test creating and configuring detector profiles"""
    print("🎯 Testing Detector Profile Creation")
    print("=" * 50)
    
    # Create a custom profile for Amazon windmill detection
    amazon_profile = DetectorProfile(
        name="Amazon Windmill Optimized",
        description="Specially tuned for Amazonian windmill structures",
        structure_type=StructureType.WINDMILL
    )
    
    # Customize geometry for our specific use case
    amazon_profile.geometry.resolution_m = 0.5
    amazon_profile.geometry.structure_radius_m = 8.0
    amazon_profile.geometry.patch_size_m = (20.0, 20.0)
    amazon_profile.geometry.patch_shape = PatchShape.SQUARE
    
    # Optimize feature weights based on our experience
    amazon_profile.features["histogram"].weight = 1.6  # Boost histogram - it's very effective
    amazon_profile.features["volume"].weight = 1.4     # Volume is important for windmills
    amazon_profile.features["compactness"].weight = 1.3 # Windmills are quite compact
    amazon_profile.features["entropy"].weight = 0.8    # Lower entropy importance
    
    # Adjust thresholds for better performance
    amazon_profile.thresholds.detection_threshold = 0.55
    amazon_profile.thresholds.confidence_threshold = 0.65
    
    print(f"✅ Created profile: {amazon_profile.name}")
    print(f"   Structure type: {amazon_profile.structure_type.value}")
    print(f"   Patch size: {amazon_profile.geometry.patch_size_m}")
    print(f"   Resolution: {amazon_profile.geometry.resolution_m}m/px")
    print(f"   Enabled features: {len(amazon_profile.get_enabled_features())}")
    print(f"   Total weight: {amazon_profile.get_total_feature_weight():.2f}")
    
    # Validate the profile
    issues = amazon_profile.validate()
    if issues:
        print(f"❌ Validation issues: {issues}")
    else:
        print("✅ Profile validation passed")
    
    return amazon_profile

def test_profile_optimization():
    """Test automatic optimization for different structure types"""
    print("\n🔧 Testing Profile Optimization")
    print("=" * 50)
    
    structure_types = [StructureType.WINDMILL, StructureType.SETTLEMENT, StructureType.EARTHWORK]
    
    for struct_type in structure_types:
        profile = DetectorProfile(
            name=f"Optimized {struct_type.value.title()}",
            structure_type=struct_type
        )
        
        # Store original weights
        original_weights = profile.get_feature_weights().copy()
        
        # Apply optimization
        profile.optimize_for_structure_type()
        optimized_weights = profile.get_feature_weights()
        
        print(f"\n{struct_type.value.upper()} OPTIMIZATION:")
        print(f"  Structure radius: {profile.geometry.structure_radius_m}m")
        print(f"  Detection threshold: {profile.thresholds.detection_threshold}")
        
        # Show weight changes
        for feature, new_weight in optimized_weights.items():
            old_weight = original_weights.get(feature, 0)
            change = new_weight - old_weight
            change_str = f"({change:+.1f})" if change != 0 else ""
            print(f"  {feature}: {new_weight:.1f} {change_str}")

def test_feature_configuration():
    """Test detailed feature configuration options"""
    print("\n⚙️  Testing Feature Configuration")
    print("=" * 50)
    
    # Create a profile with custom feature configurations
    custom_profile = DetectorProfile(
        name="Custom Feature Test",
        description="Testing advanced feature configuration"
    )
    
    # Configure histogram module with specific parameters
    custom_profile.features["histogram"] = FeatureConfiguration(
        enabled=True,
        weight=1.8,
        parameters={
            "similarity_method": "correlation",
            "bin_count": 64,
            "normalize": True
        },
        confidence_threshold=0.1
    )
    
    # Configure volume module 
    custom_profile.features["volume"] = FeatureConfiguration(
        enabled=True,
        weight=1.5,
        parameters={
            "volume_method": "simpson",
            "baseline_method": "edge_mean"
        }
    )
    
    # Disable some features for speed
    custom_profile.features["entropy"].enabled = False
    custom_profile.features["planarity"].enabled = False
    
    enabled_features = custom_profile.get_enabled_features()
    print(f"Enabled features: {list(enabled_features.keys())}")
    
    for name, config in enabled_features.items():
        print(f"  {name}:")
        print(f"    Weight: {config.weight}")
        print(f"    Parameters: {config.parameters}")
        print(f"    Confidence threshold: {config.confidence_threshold}")

def test_profile_persistence():
    """Test saving and loading profiles"""
    print("\n💾 Testing Profile Persistence")
    print("=" * 50)
    
    # Create profile manager
    manager = DetectorProfileManager("test_profiles")
    
    # Create a test profile
    test_profile = DetectorProfile(
        name="Test Persistence",
        description="Testing save/load functionality",
        structure_type=StructureType.WINDMILL
    )
    test_profile.optimize_for_structure_type()
    
    # Save the profile
    filename = manager.save_profile(test_profile)
    print(f"✅ Saved profile to: {filename}")
    
    # Load the profile back
    loaded_profile = manager.load_profile("test_persistence.json")
    print(f"✅ Loaded profile: {loaded_profile.name}")
    
    # Verify the data matches
    assert loaded_profile.name == test_profile.name
    assert loaded_profile.structure_type == test_profile.structure_type
    assert loaded_profile.geometry.resolution_m == test_profile.geometry.resolution_m
    
    print("✅ Profile persistence test passed")

def test_irregular_shapes():
    """Test support for irregular patch shapes"""
    print("\n🔷 Testing Irregular Patch Shapes")
    print("=" * 50)
    
    shapes_to_test = [
        (PatchShape.SQUARE, (20.0, 20.0)),
        (PatchShape.RECTANGLE, (30.0, 15.0)),
        (PatchShape.CIRCLE, (20.0, 20.0)),  # diameter
        (PatchShape.IRREGULAR, (25.0, 18.0))  # bounding box
    ]
    
    for shape, dimensions in shapes_to_test:
        profile = DetectorProfile(
            name=f"{shape.value.title()} Test",
            description=f"Testing {shape.value} patch shape"
        )
        
        profile.geometry.patch_shape = shape
        profile.geometry.patch_size_m = dimensions
        
        patch_px = profile.geometry.get_patch_size_px()
        
        print(f"{shape.value.upper()}:")
        print(f"  Dimensions: {dimensions[0]}×{dimensions[1]}m")
        print(f"  Pixels: {patch_px[0]}×{patch_px[1]}px")
        print(f"  Area: {dimensions[0] * dimensions[1]:.1f}m²")

def test_preset_profiles():
    """Test the preset profile creation"""
    print("\n📋 Testing Preset Profiles")
    print("=" * 50)
    
    manager = DetectorProfileManager("preset_profiles")
    manager.create_preset_profiles()
    
    available_profiles = manager.list_profiles()
    print(f"Created {len(available_profiles)} preset profiles:")
    
    for profile_file in available_profiles:
        profile = manager.load_profile(profile_file)
        print(f"  • {profile.name}")
        print(f"    Type: {profile.structure_type.value}")
        print(f"    Resolution: {profile.geometry.resolution_m}m/px")
        print(f"    Enabled features: {len(profile.get_enabled_features())}")

def main():
    """Run all profile system tests"""
    print("🧪 Detector Profile System Test Suite")
    print("="*60)
    
    # Run all tests
    amazon_profile = test_profile_creation()
    test_profile_optimization()
    test_feature_configuration()
    test_profile_persistence()
    test_irregular_shapes()
    test_preset_profiles()
    
    print("\n🎉 All profile system tests completed successfully!")
    print("\n💡 Next steps:")
    print("  1. Integrate profiles with G₂ core detector")
    print("  2. Add profile optimization based on performance metrics")
    print("  3. Create UI for profile editing and management")
    print("  4. Implement crystallized Ψ optimization algorithms")

if __name__ == "__main__":
    main()
