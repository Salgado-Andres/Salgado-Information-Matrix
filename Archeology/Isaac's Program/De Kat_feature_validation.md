# Feature Validation Report: De Kat
Generated: 2025-06-19 18:02:12

## f0_Radial_Height_Prominence
**Status:** ⚠️ ISSUES
- Mean: 0.348911, Std: 0.558209
- Range: [0.000, 2.000]
- Unique values: 8115
**Issues found:**
- 🚨 SATURATED: >10% of pixels at max value

## f1_Circular_Symmetry
**Status:** ✅ OK
- Mean: 0.412584, Std: 0.274470
- Range: [0.000, 1.000]
- Unique values: 14344

## f2_Radial_Gradient_Consistency
**Status:** ⚠️ ISSUES
- Mean: 0.941486, Std: 0.234414
- Range: [0.000, 1.000]
- Unique values: 19
**Issues found:**
- 🚨 SATURATED: >10% of pixels at max value

## f3_Ring_Edge_Sharpness
**Status:** ✅ OK
- Mean: 0.265551, Std: 0.257085
- Range: [0.000, 1.000]
- Unique values: 15681

## f4_Hough_Circle_Response
**Status:** ✅ OK
- Mean: 0.057860, Std: 0.231891
- Range: [0.000, 1.000]
- Unique values: 11

## f5_Local_Planarity
**Status:** ✅ OK
- Mean: 0.058689, Std: 0.234987
- Range: [0.000, 1.000]
- Unique values: 153

## f6_Isolation_Score
**Status:** ⚠️ ISSUES
- Mean: 0.596508, Std: 0.490011
- Range: [0.000, 1.000]
- Unique values: 9499
**Issues found:**
- 🚨 SATURATED: >10% of pixels at max value

## f7_Geometric_Coherence
**Status:** ✅ OK
- Mean: 0.053367, Std: 0.221435
- Range: [0.000, 1.000]
- Unique values: 290

## Summary
**3 anomalies detected:**
- f0_Radial_Height_Prominence: Saturation
- f2_Radial_Gradient_Consistency: Saturation
- f6_Isolation_Score: Saturation