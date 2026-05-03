# Feature Selection using Swarm Intelligence Techniques

A machine learning project that applies **Dispersive Flies Optimization (DFO)** and **Particle Swarm Optimization (PSO)** for feature selection on a high-dimensional swarm behaviour dataset, with the goal of reducing dimensionality while improving classification accuracy.

---

## Project Overview

High-dimensional datasets present a significant challenge in machine learning - too many features increase computational cost, introduce noise, and can degrade model performance. This project addresses that challenge by leveraging two nature-inspired **swarm intelligence** metaheuristic algorithms to intelligently select the most informative features before classification.

**Key Objectives:**
- Reduce the dimensionality of a large dataset (2,400 features)
- Improve classification accuracy across multiple classifiers
- Compare the effectiveness of DFO vs PSO for feature selection

---

## Project Structure

```
├── data/
│   └── Swarm_Behaviour.csv        # Dataset
├── DFO_Feature_Selection.ipynb    # Notebook: Dispersive Flies Optimization
├── PSO_Feature_Selection.ipynb    # Notebook: Particle Swarm Optimization
└── README.md
```

---

## Dataset

- **File:** `Swarm_Behaviour.csv`
- **Samples:** 23,309
- **Features:** 2,400 (position, velocity, acceleration, separation, cohesion vectors for 200 agents)
- **Target:** `Swarm_Behaviour` (Binary - `0`: No Swarm, `1`: Swarm)
- **Class Distribution:**
  - Class 0: 15,355 samples (65.9%)
  - Class 1: 7,954 samples (34.1%)

---

## Methodology

### Data Preprocessing
1. **Label Encoding** - Encode all feature columns using `LabelEncoder`
2. **Feature Scaling** - Normalize all values to [0, 1] using `MinMaxScaler`
3. **Train-Test Split** - 70% training / 30% testing (`random_state=23`)

### Swarm Intelligence Algorithms

#### Dispersive Flies Optimization (DFO)
- Population size: **50 flies**
- Dimensions: **2,400** (one per feature)
- Search bounds: **lb = 0, ub = 1** (aligned with MinMax scaled feature space)
- Initialization: Random uniform within [lb, ub]
- Disturbance threshold (δ): **0.005**
- Iterations: **50**
- Fitness function: Sphere function - minimizes the sum of squared position values
- Features selected by applying binary threshold (`thres = 0.7`) to the best fly's final position
- **Features selected: 335** (from 2,400)

#### Particle Swarm Optimization (PSO)
- Population size: **20 particles**
- Dimensions: **2,400**
- Inertia weight (w): **0.9**
- Cognitive factor (c1): **2**
- Social factor (c2): **2**
- Iterations: **100**
- Fitness function: Weighted combination of RMSE error rate and feature ratio (`α=0.99, β=0.01`)
- Binary conversion threshold: **0.5**
- **Features selected: 1,170** (from 2,400)

### Classifiers Evaluated
| Classifier | Library |
|---|---|
| Logistic Regression | `sklearn.linear_model` |
| Support Vector Machine (Linear) | `sklearn.svm` |
| Decision Tree | `sklearn.tree` |
| Random Forest | `sklearn.ensemble` |
| K-Nearest Neighbor | `sklearn.neighbors` |
| Neural Network (MLP) | `sklearn.neural_network` |

---

## Results

### DFO Feature Selection (2,400 → 335 features)

| Classifier | Without DFO | With DFO | Δ Accuracy |
|---|---|---|---|
| Logistic Regression | 88.16% | **89.99%** | **+1.83%** |
| Support Vector Machine | 87.70% | **89.68%** | **+1.98%** |
| Decision Tree | 87.23% | 87.19% | -0.04% |
| Random Forest | 87.16% | 87.10% | -0.06% |
| K-Nearest Neighbor | 88.85% | 88.87% | **+0.02%** |
| Neural Network | 89.69% | **89.78%** | **+0.09%** |

### PSO Feature Selection (2,400 → 1,170 features)

| Classifier | Without PSO | With PSO | Δ Accuracy |
|---|---|---|---|
| Logistic Regression | 88.00% | 88.97% | **+0.97%** |
| Support Vector Machine | 87.70% | 88.85% | **+1.15%** |
| Decision Tree | 87.20% | 87.20% | 0.00% |
| Random Forest | 87.00% | 87.14% | **+0.14%** |
| K-Nearest Neighbor | 88.85% | 88.89% | **+0.04%** |
| Neural Network | 89.43% | **89.12%** | -0.31% |

### Key Takeaways
- DFO with `thres=0.7` achieved the strongest dimensionality reduction (~86%), selecting only 335 features from 2,400
- PSO achieved ~51% reduction, selecting 1,170 features from 2,400
- DFO showed the largest accuracy gains for linear models (Logistic Regression: +1.83%, SVM: +1.98%)
- Tree-based models (Decision Tree, Random Forest) were largely unaffected by either algorithm
- PSO showed more consistent improvements across all classifiers with less aggressive reduction

---

## Algorithm Comparison

| Property | DFO | PSO |
|---|---|---|
| Population Size | 50 | 20 |
| Iterations | 50 | 100 |
| Search Bounds | lb=0, ub=1 (explicit) | lb=0, ub=1 (explicit) |
| Fitness Function | Sphere (minimization) | RMSE + feature ratio |
| Binary Threshold | 0.7 | 0.5 |
| Selected Features | 335 | 1,170 |
| Dimensionality Reduction | ~86% | ~51% |
| Complexity | Lower | Higher |
| Convergence Speed | Fast | Moderate |

---
## Research Papers & Resources
- The Dispersive Flies Optimisation (DFO) Algorithm - https://github.com/mohmaj/DFO
- Al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE - https://www.researchgate.net/publication/267514160_Dispersive_Flies_Optimisation

---
