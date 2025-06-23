#!/usr/bin/env python3
"""
Test the G₂ kernel system with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the kernel directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregator import StreamingDetectionAggregator
from modules.features.histogram_module import ElevationHistogramModule
from modules.features.volume_module import VolumeModule
from modules.features.compactness_module import CompactnessModule  
from modules.features.dropoff_module import DropoffSharpnessModule
from modules.features.entropy_module import ElevationEntropyModule

def create_test_data():
    """Create test elevation data - structure vs natural terrain"""
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Structure data (circular mound)
    np.random.seed(42)
    radius = 0.6
    distance = np.sqrt(x**2 + y**2)
    structure_data = np.where(distance <= radius, 
                             2.0 * np.exp(-(distance**2) / (2 * 0.3**2)), 
                             0.1 * np.random.random((size, size)))
    structure_data += 0.05 * np.random.random((size, size))
    
    # Natural terrain data (random)
    np.random.seed(123)
    natural_data = 0.2 + 0.1 * np.random.random((size, size))
    
    return structure_data, natural_data

def test_and_visualize():
    """Test the G₂ system and create visualization"""
    print("🎨 G₂ Kernel Visualization Test")
    print("=" * 50)
    
    # Create test data
    structure_data, natural_data = create_test_data()
    
    # Initialize modules
    modules = [
        ("ElevationHistogram", ElevationHistogramModule(weight=1.5)),
        ("Volume", VolumeModule(weight=1.3)),
        ("Compactness", CompactnessModule(weight=1.1)),
        ("DropoffSharpness", DropoffSharpnessModule(weight=1.1)),
        ("ElevationEntropy", ElevationEntropyModule(weight=1.0))
    ]
    
    # Test both datasets
    datasets = [
        ("Archaeological Structure", structure_data),
        ("Natural Terrain", natural_data)
    ]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for dataset_idx, (dataset_name, data) in enumerate(datasets):
        print(f"\n🔍 Testing: {dataset_name}")
        print("-" * 30)
        
        # Initialize aggregator
        aggregator = StreamingDetectionAggregator(base_score=0.5)
        aggregator.set_expected_modules(len(modules))
        
        # Collect results for visualization
        module_scores = []
        module_polarities = []
        module_weights = []
        module_names = []
        
        # Process each module
        for name, module in modules:
            result = module.compute(data)
            aggregator.add_evidence(name, result, module.weight)
            
            # Get interpreted polarity and weight
            last_signal = aggregator.evidence_signals[-1]
            
            module_scores.append(result.score)
            module_polarities.append(last_signal.polarity)
            module_weights.append(last_signal.weight)
            module_names.append(name)
            
            print(f"  {name}: {result.score:.3f} → {last_signal.polarity} (weight: {last_signal.weight:.2f})")
        
        # Get final aggregation
        final_result = aggregator.aggregate_streaming()
        
        print(f"  Final Score: {final_result.final_score:.3f}")
        print(f"  Confidence: {final_result.confidence:.3f}")
        print(f"  Evidence: {final_result.positive_evidence_count}+ / {final_result.negative_evidence_count}-")
        
        # Visualization Row 1: Elevation Data
        ax_elev = axes[dataset_idx, 0]
        im = ax_elev.imshow(data, cmap='terrain', origin='lower')
        ax_elev.set_title(f'{dataset_name}\\nElevation Data')
        plt.colorbar(im, ax=ax_elev, shrink=0.8)
        
        # Visualization Row 2: Module Scores
        ax_scores = axes[dataset_idx, 1]
        colors = ['green' if p == 'positive' else 'red' for p in module_polarities]
        bars = ax_scores.bar(range(len(module_names)), module_scores, color=colors, alpha=0.7)
        ax_scores.set_xlabel('Feature Modules')
        ax_scores.set_ylabel('Raw Scores')
        ax_scores.set_title(f'{dataset_name}\\nFeature Scores by Polarity')
        ax_scores.set_xticks(range(len(module_names)))
        ax_scores.set_xticklabels([name[:8] for name in module_names], rotation=45)
        
        # Add score labels
        for bar, score in zip(bars, module_scores):
            height = bar.get_height()
            ax_scores.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Visualization Row 3: Evidence Summary
        ax_summary = axes[dataset_idx, 2]
        pos_count = final_result.positive_evidence_count
        neg_count = final_result.negative_evidence_count
        
        # Evidence pie chart
        if pos_count + neg_count > 0:
            sizes = [pos_count, neg_count]
            labels = [f'Positive ({pos_count})', f'Negative ({neg_count})']
            colors_pie = ['lightgreen', 'lightcoral']
            ax_summary.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%')
        
        ax_summary.set_title(f'{dataset_name}\\nFinal: {final_result.final_score:.3f}\\nConf: {final_result.confidence:.3f}')
    
    plt.suptitle('G₂ Kernel: Dynamic Polarity Detection System', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('/media/im3/plus/lab4/RE/re_archaeology/kernel/g2_visualization_test.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return True

if __name__ == "__main__":
    try:
        test_and_visualize()
        print("\\n✅ Visualization test completed successfully!")
        print("📁 Saved: g2_visualization_test.png")
    except Exception as e:
        print(f"\\n❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()