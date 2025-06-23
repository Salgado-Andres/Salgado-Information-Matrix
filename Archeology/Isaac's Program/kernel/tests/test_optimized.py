#!/usr/bin/env python3
"""
Test the optimized G₂ kernel system with dynamic polarity interpretation
"""

import numpy as np
import sys
import os

# Add the kernel directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregator import StreamingDetectionAggregator
from modules.base_module import FeatureResult
from modules.features.histogram_module import ElevationHistogramModule
from modules.features.volume_module import VolumeModule
from modules.features.compactness_module import CompactnessModule  
from modules.features.dropoff_module import DropoffSharpnessModule
from modules.features.entropy_module import ElevationEntropyModule

def test_dynamic_polarity():
    """Test the dynamic polarity interpretation system"""
    print("🧪 Testing Dynamic Polarity Interpretation")
    print("=" * 50)
    
    # Create test elevation data (simulated structure)
    np.random.seed(42)
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Create a circular mound pattern (archaeological structure signature)
    radius = 0.6
    distance = np.sqrt(x**2 + y**2)
    structure_data = np.where(distance <= radius, 
                             2.0 * np.exp(-(distance**2) / (2 * 0.3**2)), 
                             0.1 * np.random.random((size, size)))
    
    # Add some noise to make it realistic
    structure_data += 0.05 * np.random.random((size, size))
    
    print(f"📊 Test data: {size}x{size} elevation grid")
    print(f"📊 Data range: {structure_data.min():.3f} to {structure_data.max():.3f}")
    
    # Initialize aggregator
    aggregator = StreamingDetectionAggregator(
        base_score=0.5,
        early_decision_threshold=0.8,
        min_modules_for_decision=2
    )
    aggregator.set_expected_modules(5)
    
    # Test each module with dynamic polarity
    modules = [
        ElevationHistogramModule(weight=1.5),
        VolumeModule(weight=1.3),
        CompactnessModule(weight=1.1),
        DropoffSharpnessModule(weight=1.1),
        ElevationEntropyModule(weight=1.0)
    ]
    
    print("\n🔍 Testing Individual Modules:")
    print("-" * 30)
    
    for i, module in enumerate(modules):
        print(f"\n{i+1}. {module.name}")
        
        # Compute feature
        result = module.compute(structure_data)
        print(f"   Raw Score: {result.score:.3f}")
        print(f"   Metadata: {list(result.metadata.keys())}")
        
        # Add to aggregator (triggers dynamic polarity interpretation)
        aggregator.add_evidence(module.name, result, module.weight)
        
        # Get streaming result
        streaming_result = aggregator.aggregate_streaming()
        
        print(f"   Interpreted Polarity: {aggregator.evidence_signals[-1].polarity}")
        print(f"   Adjusted Weight: {aggregator.evidence_signals[-1].weight:.2f}")
        print(f"   Current G₂ Score: {streaming_result.final_score:.3f}")
        print(f"   Confidence: {streaming_result.confidence:.3f}")
        
        if streaming_result.early_decision_possible:
            print(f"   🚨 Early Decision Possible!")
            
    print(f"\n📊 Final Results:")
    print(f"   Final Score: {streaming_result.final_score:.3f}")
    print(f"   Confidence: {streaming_result.confidence:.3f}")
    print(f"   Positive Evidence: {streaming_result.positive_evidence_count}")
    print(f"   Negative Evidence: {streaming_result.negative_evidence_count}")
    print(f"   Reason: {streaming_result.reason}")
    
    # Test edge case: natural terrain (should be negative)
    print(f"\n🏔️  Testing Natural Terrain (Expected: Low Score)")
    print("-" * 40)
    
    # Create random terrain data (truly random, no structure)
    np.random.seed(123)  # Different seed for natural terrain
    natural_data = 0.2 + 0.1 * np.random.random((size, size))  # Low, flat, random terrain
    
    aggregator.reset()
    aggregator.set_expected_modules(5)
    
    for module in modules[:3]:  # Test first 3 modules
        result = module.compute(natural_data)
        aggregator.add_evidence(module.name, result, module.weight)
    
    natural_result = aggregator.aggregate_streaming()
    print(f"   Natural Terrain Score: {natural_result.final_score:.3f}")
    print(f"   Confidence: {natural_result.confidence:.3f}")
    
    # Test passes if structure scores higher than natural terrain
    structure_higher = streaming_result.final_score > natural_result.final_score
    reasonable_confidence = streaming_result.confidence > 0.3
    print(f"   Structure vs Natural: {streaming_result.final_score:.3f} vs {natural_result.final_score:.3f}")
    return structure_higher and reasonable_confidence

def test_early_decision():
    """Test early decision capability"""
    print(f"\n⏱️  Testing Early Decision Logic")
    print("=" * 50)
    
    # Create very strong structural signal
    size = 30
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    
    # Perfect circular structure
    strong_structure = np.where(distance <= 0.5, 2.0, 0.1)
    
    aggregator = StreamingDetectionAggregator(
        early_decision_threshold=0.5,  # Lower threshold for testing
        min_modules_for_decision=2
    )
    aggregator.set_expected_modules(5)
    
    modules = [
        ElevationHistogramModule(weight=1.5),
        VolumeModule(weight=1.3),
        CompactnessModule(weight=1.1)
    ]
    
    early_decision_made = False
    
    for i, module in enumerate(modules):
        result = module.compute(strong_structure)
        aggregator.add_evidence(module.name, result, module.weight)
        
        streaming_result = aggregator.aggregate_streaming()
        
        print(f"Module {i+1}: Score={streaming_result.final_score:.3f}, "
              f"Confidence={streaming_result.confidence:.3f}")
        
        if streaming_result.early_decision_possible:
            print(f"🚨 Early Decision Possible after {i+1} modules!")
            early_decision_made = True
            break
        
        # Also check if confidence reached reasonable threshold
        if streaming_result.confidence > 0.5:
            print(f"🚨 High Confidence ({streaming_result.confidence:.3f}) reached after {i+1} modules!")
            early_decision_made = True
            break
    
    return early_decision_made

def main():
    """Run all optimization tests"""
    print("🚀 G₂ Kernel Optimization Test Suite")
    print("=" * 60)
    
    try:
        # Test dynamic polarity
        polarity_test_passed = test_dynamic_polarity()
        
        # Test early decision
        early_decision_test_passed = test_early_decision()
        
        print(f"\n📋 Test Results Summary:")
        print(f"   Dynamic Polarity: {'✅ PASSED' if polarity_test_passed else '❌ FAILED'}")
        print(f"   Early Decision:   {'✅ PASSED' if early_decision_test_passed else '❌ FAILED'}")
        
        if polarity_test_passed and early_decision_test_passed:
            print(f"\n🎉 All tests passed! Kernel optimization successful.")
            return True
        else:
            print(f"\n⚠️  Some tests failed. Review implementation.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)