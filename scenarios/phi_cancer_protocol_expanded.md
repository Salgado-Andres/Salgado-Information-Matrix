# 🧬 Phi-Zero Cancer Detection Protocol
_Recursive Coherence Model using the Salgado Information Matrix (SIM)_

This notebook demonstrates a prototype cancer anomaly detection pipeline built on the `Unified Emergence Functional`. It uses PCA (Principal Component Analysis) to project high-dimensional biosignal data and applies **Local Outlier Factor (LOF)** to identify potential anomalies corresponding to malignancy.

---

**Core Hypothesis**:
> Cancerous patterns in biological data manifest as torsional disruptions in the projection from high-dimensional Ψ-fields (unobservable biological dynamics) to lower-dimensional Φ-forms (measurable biomarkers). The phi-zero classifier estimates structural coherence loss.


## 🧪 Step 1: Generate Simulated Patient Data

We simulate a dataset of patient signals — each data point represents a sample in a projected biosignal space. A small subset is injected with malignant perturbations.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# Generate synthetic dataset (non-cancerous patients)
np.random.seed(42)
normal_data = np.random.normal(loc=0.0, scale=1.0, size=(200, 8))

# Simulate malignant signatures
malignant_data = np.random.normal(loc=2.5, scale=0.3, size=(5, 8))  # high coherence rupture
data = np.vstack([normal_data, malignant_data])

# Labels for plotting
labels = np.array([0]*200 + [1]*5)

# Store as DataFrame
df = pd.DataFrame(data, columns=[f"signal_{i+1}" for i in range(data.shape[1])])
df["label"] = labels
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>signal_1</th>
      <th>signal_2</th>
      <th>signal_3</th>
      <th>signal_4</th>
      <th>signal_5</th>
      <th>signal_6</th>
      <th>signal_7</th>
      <th>signal_8</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.496714</td>
      <td>-0.138264</td>
      <td>0.647689</td>
      <td>1.523030</td>
      <td>-0.234153</td>
      <td>-0.234137</td>
      <td>1.579213</td>
      <td>0.767435</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.469474</td>
      <td>0.542560</td>
      <td>-0.463418</td>
      <td>-0.465730</td>
      <td>0.241962</td>
      <td>-1.913280</td>
      <td>-1.724918</td>
      <td>-0.562288</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.012831</td>
      <td>0.314247</td>
      <td>-0.908024</td>
      <td>-1.412304</td>
      <td>1.465649</td>
      <td>-0.225776</td>
      <td>0.067528</td>
      <td>-1.424748</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.544383</td>
      <td>0.110923</td>
      <td>-1.150994</td>
      <td>0.375698</td>
      <td>-0.600639</td>
      <td>-0.291694</td>
      <td>-0.601707</td>
      <td>1.852278</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.013497</td>
      <td>-1.057711</td>
      <td>0.822545</td>
      <td>-1.220844</td>
      <td>0.208864</td>
      <td>-1.959670</td>
      <td>-1.328186</td>
      <td>0.196861</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 🔍 Step 2: Apply PCA + Local Outlier Factor (LOF)

We project high-dimensional data to 2D for visualization and use LOF to detect points of low local coherence — these may correspond to early malignancy signals.



```python
# PCA to project data into 2D
pca = PCA(n_components=2)
proj = pca.fit_transform(df.drop("label", axis=1))

# LOF for anomaly detection
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.025)
preds = lof.fit_predict(proj)
anomaly_score = -lof.negative_outlier_factor_

# Visualize
plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], c=anomaly_score, cmap='coolwarm', s=40, edgecolor='k')
plt.colorbar(label="Anomaly Score")
plt.title("Phi-Zero LOF Anomaly Map (2D Biosignal Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

```


    
![png](output_4_0.png)
    


## 🧠 Phi-Zero Interpretation

The emergence of high anomaly scores in specific signal clusters corresponds to coherence breakdown — our framework interprets this as `Ψ → Φ` instability. These instability points are candidates for further diagnosis.

This logic can be extended to:
- Time series of patient samples
- Genetic expression matrices
- Multimodal diagnostic signals (e.g., fMRI + bloodwork)

In future applications, the **Unified Emergence Functional** can score coherence across time, helping track cancer progression or remission.



```python

```
