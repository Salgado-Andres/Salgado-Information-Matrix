# Amazon Archaeological Detection Profiles

## Overview

This document describes three specialized detection profiles designed for identifying pre-Columbian archaeological features in the Amazon basin using the G₂ kernel system. Each profile is optimized for different types of cultural earthworks and settlements.

## Profile 1: Amazon Civilization Mounds

### Description
Detects artificial earthwork mounds and residential platforms created by pre-Columbian Amazonian cultures.

### Archaeological Context
- **Cultural Period**: Pre-Columbian (1000 BCE - 1500 CE)
- **Function**: Residential platforms, ceremonial mounds, burial sites
- **Typical Characteristics**:
  - Height: 1-8 meters
  - Diameter: 15-100 meters
  - Shape: Irregular to sub-circular
  - Often associated with terra preta (anthropogenic soils)

### Detection Parameters
- **Scale**: 60m analysis patches
- **Resolution**: 1.0m
- **Thresholds**: Relaxed (0.45/0.4) for subtle features
- **Key Features**: Volume analysis for elevated features, adaptive shape detection

### Environmental Challenges
- Dense forest canopy obscuring features
- Erosion and bioturbation degradation
- Seasonal flooding effects
- Modern agricultural disturbance

## Profile 2: Amazon Citadels

### Description
Identifies large fortified settlements and defensive earthwork complexes.

### Archaeological Context
- **Cultural Period**: Late Pre-Columbian (800-1500 CE)
- **Function**: Fortified settlements, elite residences, defensive complexes
- **Typical Characteristics**:
  - Height: 2-15 meters
  - Diameter: 50-400 meters
  - Shape: Polygonal to irregular
  - Features: Walls, ditches, terraces, palisade foundations

### Detection Parameters
- **Scale**: 180m analysis patches (largest)
- **Resolution**: 0.8m (high detail)
- **Thresholds**: Conservative (0.4/0.35) for complex structures
- **Key Features**: Compound gradient analysis, hierarchical consensus

### Distinctive Features
- **Defensive Architecture**: Multi-component earthworks
- **Strategic Locations**: Elevated positions, river confluences
- **Complex Geometry**: Multiple interconnected features
- **Terra Preta Association**: Rich anthropogenic soils

## Profile 3: Amazon Geoglyphs (Ring Structures)

### Description
Detects geometric earthworks including ring ditches, circular enclosures, and astronomical alignments.

### Archaeological Context
- **Cultural Period**: Pre-Columbian Geometric Tradition (200-1500 CE)
- **Function**: Ceremonial enclosures, astronomical observatories, territorial markers
- **Typical Characteristics**:
  - Depth: 0.5-3.0 meters (ditches)
  - Diameter: 30-200 meters
  - Shape: Perfect circles, concentric rings
  - Features: Precise geometric construction, radial causeways

### Detection Parameters
- **Scale**: 120m analysis patches
- **Resolution**: 0.6m (high precision)
- **Thresholds**: Strict (0.55/0.5) for geometric precision
- **Key Features**: Ring-specific algorithms, circularity analysis

### Unique Characteristics
- **Geometric Precision**: Near-perfect circular construction
- **Negative Features**: Excavated ditches rather than raised mounds
- **Astronomical Alignments**: Cardinal directions, celestial events
- **Preservation**: Often excellently preserved in cleared areas

## Comparative Analysis

| Feature | Mounds | Citadels | Geoglyphs |
|---------|--------|----------|-----------|
| **Size** | Small-Medium | Large | Medium |
| **Complexity** | Simple | Complex | Geometric |
| **Preservation** | Variable | Good | Excellent |
| **Detection Difficulty** | High | Medium | Low |
| **False Positives** | Natural hills | Natural mesas | Cattle rings |

## Detection Strategy

### Polarity Preferences
- **Mounds**: Positive elevation features
- **Citadels**: Positive with compound analysis
- **Geoglyphs**: Negative (ditches) with geometric constraints

### Feature Weights
- **Mounds**: Balanced approach with emphasis on entropy
- **Citadels**: Volume and histogram weighted for large features
- **Geoglyphs**: Compactness and entropy for geometric precision

### Environmental Adaptation
Each profile includes specific parameters for:
- Vegetation density effects
- Soil type variations
- Drainage patterns
- Modern disturbance factors

## Validation and Known Sites

### Reference Locations
- **Mounds**: Marajoara culture sites, Llanos de Mojos
- **Citadels**: Monte Alegre complexes, Upper Xingu settlements
- **Geoglyphs**: Acre state geometric earthworks, Rondônia rings

### Common False Positives
- **Natural Features**: Hills, mesas, natural ponds
- **Modern Features**: Agricultural terraces, cattle rings, mining
- **Biological Features**: Termite mounds, tree fall mounds

## Usage Guidelines

### Site Selection
1. **Mounds**: Focus on floodplain margins, terra preta areas
2. **Citadels**: Target strategic highlands, river confluences
3. **Geoglyphs**: Survey deforested/pastoral areas with LiDAR

### Quality Control
- Cross-reference with known archaeological databases
- Validate against soil maps (terra preta indicators)
- Consider cultural landscape context
- Field verification recommended for high-confidence detections

### Processing Recommendations
- Use hierarchical approach: geoglyphs first (easiest), then citadels, finally mounds
- Apply multiple profiles to same area for comprehensive coverage
- Combine with environmental layers (soil, hydrology, vegetation)

## Future Development

### Planned Enhancements
- Multi-temporal analysis for change detection
- Integration with ethnographic territory maps
- Ceramic scatter correlation algorithms
- Carbon dating probability modeling

### Research Priorities
- Refine parameters based on field validation
- Develop landscape-scale pattern recognition
- Improve natural vs. artificial discrimination
- Enhance preservation state assessment

---

*These profiles represent current best practices in archaeological remote sensing for Amazonian contexts. Regular updates incorporate new archaeological discoveries and methodological advances.*
