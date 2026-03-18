# **Bayesian Reinforcement Learning Model of Multisensory Learning**  
**A Reanalysis of Bedi et al. (2025) using Bayesian Beta-Binomial RL model**

## **Overview**
Reanalysis of behavioral data from Bedi et al. (2025) using a Bayesian Beta-Binomial reinforcement learning model.
Examines whether tracking uncertainty via Beta distributions in multisensory learning:

### **H1 – Differential Fit Across Original Groups**
1. Do participants classified into different Q-learning groups (Basic/Asym/Transfer) show differential fit in the Bayesian model?
**Test:** Log-likelihood and BIC compared with descriptive stats & Cohen's d. No inferential stats due to circularity.


### **H2 – Predicting Original Group Labels**
Can Bayesian parameters predict original Q-learning group assignments?  
**Test:** Random forest classifier with temperature, LL, BIC, probability; 5-fold stratified CV.

---

## **Dataset**
– **Original Study:** Bedi et al. (2025) - Separable neurocomputational mechanisms underlying multisensory learning  
– **Data:** `model_ready_data.csv`and `bestfitting.tsv`, available at: github.com/ruffgroup/multlearn  
– **Participants:** 62 (in behavioral modeling), 58 (H1 & H2, required group assignment matching)  
– **Task:** 360 trials (3 images × 3 sounds/tactile patterns)  
– **Original Groups:** Basic (n=31), Asymmetric (n=11), Transfer (n=16)  

---

## **Methods**

### **1. Data Loading & Setup**
– Libraries: NumPy, Pandas, SciPy, Matplotlib, Scikit-learn  
– Loading `model_ready_data.csv` + `BestFitting.tsv`  
– Inspecting for missing values & structure

### **2. Bayesian Beta-Binomial Model**
| Function | Purpose |
|----------|---------|
| `expected_value()` | α / (α+β) reward probability |
| `update()` | Updates Beta parameters after outcome |
| `softmax()` | Converts expected values to choice probabilities |
| `calculate_log_likelihood()` | Computes model fit per participant |

### **3. Parameter Optimization**
– Temperature per participant (`minimize_scalar`, 0.01–20)  
– Extracting cumulative group log-likelihood (LL), per-trial LL, probability  
– BIC: `BIC = -2*LL + 1*log(n_trials)`

### **4. Group Analysis (H1)**
– Merging group labels with Bayesian results  
– Computing means & Cohen's d  
– Comparison to random baseline (LL = -242.6, 50% accuracy)

### **5. Random Forest Classification (H2)**
– Features: temperature, LL, BIC, probability  
– 5-fold CV, extracting feature importance, confusion matrix

### **6. Parameter Recovery**
– Simulated temps: [0.5, 1.0, 2.0, 5.0] ×10 simulations  
– Correlation: true vs recovered temperature

### **7. Learning Curve Analysis**
– Cumulative accuracy per participant  
– Group-level curves + SEM  
– 80% participant cutoff

---

## **Results**

### **Model Performance**
– Participants above chance: 41/62 (66%)  
– Mean prediction accuracy: 52.3% (median 51.4%)  
– Temperature: 0.01–4.94 (mean 1.47)  
– BIC: 168.1–505.0 (mean 463.2)

### **H1 – Group Differences**
| Group | Log-Likelihood | BIC | Effect Size |
|-------|---------------|-----|------------|
| Asym | -220.45 | 446.8 | d = -1.50 |
| Basic | -240.46 | 486.8 | reference |
| Transfer | -220.56 | 447.0 | d = -1.51 |

### **H2 – Classification**
– Accuracy: 49.7% (chance 33.3%)  
– Feature importance: Temp 33.7%, BIC 23.6%, Prob 22.6%, LL 20.1%  
– Per-group accuracy: Basic 74.2%, Asym 45.5%, Transfer 6.2%

### **Parameter Recovery**
– Correlation r = 0.99

### **Learning Curve**
– Accuracy improves across trials from 50% → ~65%  
– Basic underperforms Asym/Transfer

---

## **Visualizations**
1. Temperature distribution  
2. Model fit comparison by original group(LL & BIC)  
3. Confusion matrix classification performance 
4. Parameter recovery scatter
5. Learning curve over time

---

## **Limitations**
– No direct model comparison  
– Simplified reward structure  
– No hierarchical structure  
– Behavioral-only analysis  
– Single dataset

---

## **Reproducibility**
```bash
pip install -r requirements.txt
