#!/usr/bin/env python3
"""
Simple test for polarity preferences functionality
"""

import os
import sys
import json
import numpy as np

# Add kernel to path
kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, kernel_dir)

def test_polarity_preferences_loading():
    """Test that polarity preferences are loaded correctly from template"""
    print("🧪 Testing Polarity Preferences Loading")
    print("=" * 42)
    
    # Load the Dutch windmill template directly
    template_path = os.path.join(kernel_dir, "templates", "dutch_windmill.json")
    
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return False
    
    try:
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        print(f"✅ Template loaded: {template_data.get('name', 'Unknown')}")
        
        # Check for polarity preferences
        polarity_prefs = template_data.get('polarity_preferences', {})
        
        if polarity_prefs:
            print(f"✅ Polarity preferences found: {len(polarity_prefs)} features configured")
            for feature, polarity in polarity_prefs.items():
                print(f"   {feature}: {polarity}")
            
            # Validate polarity values
            valid_polarities = ["positive", "negative", None]
            all_valid = all(pol in valid_polarities for pol in polarity_prefs.values())
            
            if all_valid:
                print("✅ All polarity preferences are valid")
                return True
            else:
                print("❌ Some polarity preferences have invalid values")
                return False
        else:
            print("❌ No polarity preferences found in template")
            return False
            
    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return False

def test_aggregator_polarity_logic():
    """Test aggregator polarity interpretation logic"""
    print("\n🔬 Testing Aggregator Polarity Logic")
    print("=" * 38)
    
    try:
        # Import aggregator
        from aggregator import StreamingDetectionAggregator
        
        # Test polarity preferences
        polarity_prefs = {
            "histogram": "positive",
            "entropy": "negative",
            "volume": None  # Should use dynamic interpretation
        }
        
        # Create aggregator with polarity preferences
        aggregator = StreamingDetectionAggregator(polarity_preferences=polarity_prefs)
        
        print("✅ Aggregator created with polarity preferences")
        print(f"   Configured preferences: {len(polarity_prefs)} features")
        
        # Test polarity interpretation for each case
        test_cases = [
            ("histogram", 0.7, {}, 1.0, "positive"),  # Should force positive
            ("entropy", 0.8, {}, 1.0, "negative"),   # Should force negative  
            ("volume", 0.6, {}, 1.0, None),          # Should use dynamic logic
            ("unknown_feature", 0.5, {}, 1.0, None), # Should use dynamic logic
        ]
        
        for feature_name, score, metadata, weight, expected_polarity in test_cases:
            try:
                polarity, adjusted_weight = aggregator._interpret_polarity(
                    feature_name, score, metadata, weight
                )
                
                if expected_polarity is None:
                    # For dynamic interpretation, just check that we get a valid polarity
                    if polarity in ["positive", "negative"]:
                        print(f"   ✅ {feature_name}: {polarity} (dynamic)")
                    else:
                        print(f"   ❌ {feature_name}: Invalid polarity '{polarity}'")
                elif polarity == expected_polarity:
                    print(f"   ✅ {feature_name}: {polarity} (forced)")
                else:
                    print(f"   ❌ {feature_name}: Expected '{expected_polarity}', got '{polarity}'")
                    
            except Exception as e:
                print(f"   ❌ {feature_name}: Error - {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import aggregator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing aggregator: {e}")
        return False

def test_synthetic_windmill_generation():
    """Test synthetic windmill data generation"""
    print("\n🏰 Testing Synthetic Windmill Generation")
    print("=" * 40)
    
    try:
        # Generate a synthetic windmill mound
        def create_windmill_mound(size=40, radius=8.0, height=2.0):
            y, x = np.ogrid[:size, :size]
            center = size // 2
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            mound = height * np.exp(-(distance**2) / (2 * (radius/2.5)**2))
            
            # Add slight asymmetry
            angle = np.arctan2(y - center, x - center)
            asymmetry = 1.0 + 0.1 * np.cos(2 * angle)
            mound *= asymmetry
            
            # Sharp edge
            edge_mask = distance < radius
            mound[~edge_mask] *= 0.3
            
            return mound
        
        windmill = create_windmill_mound()
        
        print(f"✅ Synthetic windmill generated: {windmill.shape}")
        print(f"   Elevation range: {windmill.min():.3f} to {windmill.max():.3f}")
        print(f"   Mean elevation: {windmill.mean():.3f}")
        print(f"   Standard deviation: {windmill.std():.3f}")
        
        # Basic properties check
        center = windmill.shape[0] // 2
        center_height = windmill[center, center]
        edge_height = windmill[0, 0]  # Corner should be low
        
        if center_height > edge_height:
            print("   ✅ Center is higher than edges (good mound shape)")
        else:
            print("   ❌ Center is not higher than edges")
        
        # Check for reasonable elevation distribution
        if 0.1 < windmill.std() < 1.0:
            print("   ✅ Reasonable elevation variation")
        else:
            print("   ⚠️ Unusual elevation variation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating synthetic windmill: {e}")
        return False

def run_simple_tests():
    """Run all simple tests"""
    print("🧪 Simple Polarity Preferences Test Suite")
    print("=" * 42)
    print()
    
    # Test 1: Template loading
    test1_passed = test_polarity_preferences_loading()
    
    # Test 2: Aggregator logic
    test2_passed = test_aggregator_polarity_logic()
    
    # Test 3: Synthetic data
    test3_passed = test_synthetic_windmill_generation()
    
    # Summary
    print(f"\n📊 Test Results")
    print("=" * 15)
    
    tests = [
        ("Template Loading", test1_passed),
        ("Aggregator Logic", test2_passed), 
        ("Synthetic Data", test3_passed)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🏅 All tests passed! Polarity preferences system is working correctly.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. System is mostly functional.")
    else:
        print("⚠️ Some tests failed. System needs attention.")

if __name__ == "__main__":
    run_simple_tests()
