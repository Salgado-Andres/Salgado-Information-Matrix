# Dutch Windmill Detection Profile

## Overview

The Dutch Windmill Detector is a specialized G₂ kernel profile optimized for identifying historical windmill mound structures in the Netherlands and similar European landscapes. This profile represents a refined approach to detecting small, circular, elevated earthworks characteristic of traditional windmill foundations.

## Profile: Dutch Windmill Detector

### Description
Detects circular windmill mound structures using modular feature configuration with high precision geometric analysis.

### Historical Context
- **Period**: Medieval to Early Modern (1200-1900 CE)
- **Function**: Windmill foundations, mill mounds, industrial heritage sites
- **Typical Characteristics**:
  - Height: 2-8 meters
  - Diameter: 8-35 meters
  - Shape: Nearly perfect circular
  - Construction: Artificial earthwork with stone/brick foundations

### Detection Parameters
- **Scale**: 20m analysis patches (small, focused detection)
- **Resolution**: 0.5m (high precision for detailed features)
- **Thresholds**: Moderate (0.5/0.5) for well-preserved features
- **Early Decision**: High confidence (0.85) for efficient processing

## Technical Specifications

### Geometry Configuration
```json
{
  "resolution_m": 0.5,           // High resolution for precise detection
  "structure_radius_m": 8.0,     // Typical windmill mound radius
  "min_structure_size_m": 4.0,   // Minimum detectable size
  "max_structure_size_m": 40.0,  // Maximum expected size
  "patch_size_m": [20.0, 20.0],  // Compact analysis window
  "aspect_ratio_tolerance": 0.25  // Strict circularity requirement
}
```

### Feature Analysis Methods

#### **Histogram Analysis**
- **Method**: Correlation-based similarity
- **Bins**: 25 (moderate granularity)
- **Enhancements**: Edge enhancement, adaptive binning, noise reduction
- **Polarity**: Positive (elevated features)
- **Variation Threshold**: 0.25 (moderate contrast requirement)

#### **Volume Analysis**
- **Method**: Adaptive volume calculation
- **Normalization**: 40.0m³ base volume, 4.0m prominence
- **Border Factor**: 0.3 (30% border consideration)
- **Scaling**: Adaptive with concentration bonus (1.2x)
- **Polarity**: Neutral (flexible elevation assessment)

#### **Dropoff Analysis**
- **Method**: Gradient-based edge detection
- **Smoothing**: 0.8m radius (fine-scale analysis)
- **Factors**: Inner 0.7σ, Outer 1.3σ (steep gradient detection)
- **Enhancements**: Directional analysis, edge enhancement
- **Polarity**: Positive (outward slope from center)

#### **Compactness Analysis**
- **Method**: Circularity measurement
- **Angles**: 45 (8-degree resolution)
- **Samples**: Minimum 12 points
- **Symmetry**: 0.85 factor (high circularity requirement)
- **Analysis**: Fourier analysis, edge detection
- **Polarity**: Positive (circular shape preference)

#### **Entropy Analysis**
- **Method**: Shannon entropy
- **Bins**: 16 spatial bins
- **Edge Weight**: 1.3 (emphasize boundaries)
- **Normalization**: Local context
- **Polarity**: Negative (structured vs. random patterns)

#### **Planarity Analysis**
- **Method**: Least squares plane fitting
- **Outlier Threshold**: 2.0σ (moderate tolerance)
- **Edge Weight**: 0.7 (reduced edge influence)
- **Smoothing**: 1.2m radius
- **Polarity**: Negative (deviation from flat plane)

## Feature Weighting Strategy

### Weight Distribution
```json
{
  "histogram": 1.0,    // Standard elevation pattern matching
  "volume": 0.8,       // Moderate volume emphasis
  "dropoff": 1.0,      // Standard edge definition
  "compactness": 1.0,  // Standard shape analysis
  "entropy": 3.0,      // HIGH - Key discriminator
  "planarity": 0.6     // Reduced - allows for mound structure
}
```

### Rationale
- **Entropy (3.0)**: Primary discriminator between artificial mounds and natural features
- **Equal weights (1.0)**: Histogram, dropoff, and compactness provide balanced geometric analysis
- **Reduced planarity (0.6)**: Allows for natural mound curvature
- **Moderate volume (0.8)**: Balances size detection without over-emphasis

## Environmental Context

### Landscape Characteristics
- **Terrain**: Flat to gently rolling agricultural land
- **Vegetation**: Managed grassland, crop fields
- **Soil**: Clay-rich, well-drained soils
- **Drainage**: Controlled drainage systems, polders
- **Preservation**: Generally excellent due to stable environment

### Modern Challenges
- **Agricultural Activity**: Plowing, field leveling
- **Urban Development**: Suburban expansion
- **Infrastructure**: Roads, utilities crossing sites
- **Tourism**: Recreational modifications

## Detection Performance

### Optimal Conditions
- **Open Fields**: Minimal vegetation interference
- **Winter/Spring**: Reduced crop coverage
- **High-Resolution Data**: LiDAR or photogrammetric DEMs
- **Stable Weather**: Consistent illumination for optical data

### Challenging Conditions
- **Dense Vegetation**: Summer crop coverage
- **Recent Cultivation**: Fresh plowing artifacts
- **Modern Modifications**: Landscaping, construction
- **Flooding**: Seasonal water level changes

## Validation and Quality Control

### Reference Sites
- **Known Windmills**: Historical windmill registers
- **Archaeological Records**: Cultural heritage databases
- **Historical Maps**: 18th-19th century cartographic sources
- **Aerial Photography**: Historical aerial survey archives

### False Positive Patterns
- **Natural Features**: Small hills, erosion mounds
- **Agricultural Features**: Silage clamps, manure piles
- **Modern Features**: Building foundations, utility structures
- **Historical Features**: Burial mounds, signal hills

### Verification Methods
1. **Historical Cross-Reference**: Check against windmill databases
2. **Morphological Analysis**: Verify typical windmill mound characteristics
3. **Contextual Assessment**: Evaluate landscape position and accessibility
4. **Field Validation**: Ground-truthing for high-confidence detections

## Processing Recommendations

### Workflow Integration
1. **Preprocessing**: Ensure high-quality DEM with vegetation removal
2. **Initial Scan**: Run windmill detector with standard parameters
3. **Refinement**: Apply up to 3 refinement iterations for borderline cases
4. **Post-Processing**: Filter results by historical plausibility
5. **Validation**: Cross-reference with cultural heritage databases

### Parameter Tuning
- **High Precision Mode**: Increase resolution to 0.3m, reduce patch size to 15m
- **Survey Mode**: Decrease resolution to 0.8m, increase patch size to 30m
- **Conservative Mode**: Increase all thresholds by 0.1
- **Sensitive Mode**: Decrease detection threshold to 0.4

### Performance Optimization
- **Parallel Processing**: 6 workers optimal for typical datasets
- **Memory Management**: 20m patches minimize memory usage
- **Early Decision**: 0.85 threshold provides 40% processing speedup
- **Streaming Aggregation**: Efficient for large-area surveys

## Cultural Heritage Applications

### Archaeological Significance
- **Industrial Heritage**: Documentation of wind-powered industry
- **Landscape Evolution**: Understanding historical land use
- **Settlement Patterns**: Relationship to villages and trade routes
- **Technological Development**: Evolution of wind power technology

### Management Applications
- **Heritage Protection**: Identifying sites requiring preservation
- **Planning Support**: Development impact assessment
- **Tourism Development**: Heritage trail planning
- **Research Support**: Academic and historical research

### Integration with Other Datasets
- **Historical Maps**: Overlay with 18th-19th century cartography
- **Cadastral Data**: Property boundary relationships
- **Geological Maps**: Soil and substrate correlations
- **Hydrological Data**: Drainage pattern associations

## Future Enhancements

### Planned Improvements
- **Multi-temporal Analysis**: Change detection over time
- **Material Classification**: Stone vs. earth construction detection
- **Associated Features**: Detection of mill races, access roads
- **Condition Assessment**: Preservation state evaluation

### Research Priorities
- **Regional Variations**: Adaptation for different windmill traditions
- **Period Specificity**: Temporal classification capabilities
- **Functional Classification**: Post mills vs. tower mills vs. smock mills
- **Landscape Context**: Integration with broader settlement analysis

---

*This profile represents optimized parameters for Dutch windmill detection based on extensive testing and validation against known archaeological sites. Regular updates incorporate new discoveries and methodological improvements.*
