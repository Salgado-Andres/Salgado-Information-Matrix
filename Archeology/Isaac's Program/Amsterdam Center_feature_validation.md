# Feature Validation Report: Amsterdam Center
Generated: 2025-06-19 18:02:29

## f0_Radial_Height_Prominence
**Status:** ✅ OK
- Mean: 0.160196, Std: 0.486262
- Range: [0.000, 2.000]
- Unique values: 2322

## f1_Circular_Symmetry
**Status:** ✅ OK
- Mean: 0.578138, Std: 0.353291
- Range: [0.000, 1.000]
- Unique values: 15657

## f2_Radial_Gradient_Consistency
**Status:** ⚠️ ISSUES
- Mean: 0.941250, Std: 0.235120
- Range: [0.000, 1.000]
- Unique values: 6
**Issues found:**
- 🚨 SATURATED: >10% of pixels at max value

## f3_Ring_Edge_Sharpness
**Status:** ✅ OK
- Mean: 0.323459, Std: 0.261368
- Range: [0.000, 1.000]
- Unique values: 15682

## f4_Hough_Circle_Response
**Status:** ✅ OK
- Mean: 0.058034, Std: 0.232446
- Range: [0.000, 1.000]
- Unique values: 17

## f5_Local_Planarity
**Status:** ✅ OK
- Mean: 0.058739, Std: 0.235069
- Range: [0.000, 1.000]
- Unique values: 154

## f6_Isolation_Score
**Status:** ⚠️ ISSUES
- Mean: 0.590810, Std: 0.491234
- Range: [0.000, 1.000]
- Unique values: 9432
**Issues found:**
- 🚨 SATURATED: >10% of pixels at max value

## f7_Geometric_Coherence
**Status:** ✅ OK
- Mean: 0.457824, Std: 0.240221
- Range: [0.000, 1.299]
- Unique values: 16507

## Summary
**2 anomalies detected:**
- f2_Radial_Gradient_Consistency: Saturation
- f6_Isolation_Score: Saturation