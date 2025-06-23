#!/usr/bin/env python3
"""
Simple test for the DetectorProfile system
"""

import os
import sys
import numpy as np

# Add kernel to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_profile import DetectorProfile, StructureType, PatchShape, profile_manager

def test_basic_profile():
    """Test basic profile creation and configuration"""
    print("🧪 Basic Profile Test")
    print("=" * 30)
    
    # Create a simple windmill profile
    profile = DetectorProfile(
        name="SimpleWindmill",
        description="Basic windmill detection profile"
    )
    
    print(f"✅ Profile created: {profile.name}")
    print(f"   Features: {list(profile.features.keys())}")
    print(f"   Enabled features: {len(profile.get_enabled_features())}")
    print(f"   Resolution: {profile.geometry.resolution_m}m")
    print(f"   Patch size: {profile.geometry.patch_size_m}")
    print(f"   Detection threshold: {profile.thresholds.detection_threshold}")
    
    # Test feature weight modification
    profile.features["histogram"].weight = 2.0
    print(f"   Histogram weight: {profile.features['histogram'].weight}")
    
    # Test feature disable
    profile.features["planarity"].enabled = False
    print(f"   Planarity enabled: {profile.features['planarity'].enabled}")
    
    return profile

def test_persistence():
    """Test saving and loading profiles"""
    print("\n💾 Persistence Test")
    print("=" * 30)
    
    # Create and configure a profile
    profile = DetectorProfile(name="TestProfile")
    profile.features["histogram"].weight = 1.8
    profile.features["entropy"].enabled = False
    
    # Save profile
    profile_path = profile_manager.save_profile(profile, "test_simple.json")
    print(f"✅ Profile saved to: {profile_path}")
    
    # Load profile
    loaded = profile_manager.load_profile("test_simple.json")
    print(f"✅ Profile loaded: {loaded.name}")
    print(f"   Histogram weight: {loaded.features['histogram'].weight}")
    print(f"   Entropy enabled: {loaded.features['entropy'].enabled}")
    
    return loaded

def main():
    """Run basic tests"""
    print("🧪 DetectorProfile Basic Test")
    print("=" * 40)
    
    # Test basic functionality
    basic_profile = test_basic_profile()
    
    # Test persistence
    loaded_profile = test_persistence()
    
    print("\n✅ All basic tests completed!")

if __name__ == "__main__":
    main()
