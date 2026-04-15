# **Bayesian Reinforcement Learning Model of Multisensory Learning**  
**A Reanalysis of Bedi et al. (2025) using Bayesian Beta-Binomial RL model**

## **Overview**
Reanalysis of behavioral data from Bedi et al. (2025) using a Bayesian Beta-Binomial reinforcement learning model.
The goal is to evaluate whether a probabilistic model that explicitly represents uncertainty can capture individual differences in multisensory learning and relate to previously identified Q-learning strategies.

Two hypotheses are tested:

### **H1 – Differential Fit Across Original Groups**
1. Do participants classified into different Q-learning groups (Basic/Asym/Transfer) show differential fit in the Bayesian model?  
**Test:** Log-likelihood and BIC compared with descriptive stats & Cohen's d. No inferential statistics were performed due to non-independent group formation.


### **H2 – Predicting Original Group Labels**
2. Can Bayesian parameters predict original Q-learning group assignments?  
**Test:** Random forest classifier with inverse temperature (θ), log-likelihood (LL), BIC, probability; 5-fold stratified cross-validation.

---

## **Dataset**
– **Original Study:** Bedi et al. (2025) - Separable neurocomputational mechanisms underlying multisensory learning  
– **Files:**  
`model_ready_data.csv` (behavioral data)  
`bestfitting.tsv` (original group labels)  
Dataset access: [multlearn repository](https://github.com/ruffgroup/multlearn)  
– **Participants:**  
62 (in behavioral modeling)  
58 (group-based analyses)  
– **Task:**  
up to 360 trials (variable: 118–360 per participant)  
3 images × 3 sounds/tactile patterns  
– **Original Groups:** Basic (n=31), Asymmetric (n=11), Transfer (n=16)  

---

## **Methods**

### **1. Bayesian Beta-Binomial Model**
| Function | Purpose |
|----------|---------|
| `expected_value()` | α / (α+β) reward probability |
| `update()` | Updates α/β based on reward outcome |
| `softmax()` | Converts expected values to choice probabilities |
| `calculate_log_likelihood()` | Computes model fit per participant |

– Initial prior: α = 1, β = 1 (uniform)  
– Free parameter: inverse temperature (θ)  
Interpretation:  
Low θ → exploratory behavior
High θ → deterministic behavior

### **2. Parameter Optimization**
– Inverse temperature per participant (`minimize_scalar`, 0.01–20)  
– Extracting cumulative group log-likelihood (LL), per-trial LL, probability  
note: probability (exp(avg log-likelihood), interpreted as retrospective accuracy proxy
– BIC: `BIC = -2*LL + 1*log(n_trials)`

### **3. Group Analysis (H1)**
– Merging Bayesian results with original group labels  
– Comparison of:  
Mean LL  
Mean BIC  
– Computing means & Cohen's d  
– Comparison to reference baseline (LL = -242.6, 50% accuracy)

### **4. Random Forest Classification (H2)**
– Features: Inverse temperature, LL, BIC, probability  
– 5-fold stratified CV, extracting feature importance, confusion matrix

### **5. Parameter Recovery**
– Simulated temps: [0.5, 1.0, 2.0, 5.0] ×10 simulations per value  
– Correlation: true vs recovered temperature  

### **6. Learning Curve Analysis**
– Cumulative accuracy aggregated across participants, with inclusion threshold: ≥80% participants per trial  
– Mean ± SEM plotted over trials  

---

## **Results**

### **Model Performance**
– Participants above chance: 41/62 (66%)  
– Mean prediction accuracy: 52.3% (median 51.4%)  
– Inverse temperature:
Range: 0.01–4.94 
Mean: 1.47  
– BIC: 168.1–505.0 (mean 463.2)

### **H1 – Group Differences**
| Group | Log-Likelihood | BIC | Effect Size |
|-------|---------------|-----|------------|
| Asym | -220.45 | 446.8 | d = -1.50 |
| Basic | -240.46 | 486.8 | reference |
| Transfer | -220.56 | 447.0 | d = -1.51 |

Asym and Transfer show nearly identical fit.
Basic group performs near chance.

### **H2 – Classification**
– Accuracy: 49.7% +/- 0.098 (chance 33.3%, below majority baseline 53.4%)  
– Per-group classification recovery:  
Transfer: 1/16 = 6.2%  
Basic: 23/31 = 74.2%  
Asym: 5/11 = 45.5%  
– Feature importance: Temp 33.7%, BIC 23.6%, Prob 22.6%, LL 20.1%  


### **Parameter Recovery**
– Correlation r = 0.967.
Accurate recovery across full parameter range

### **Learning Curve**
– Accuracy improves modestly across trials, from 50% → ~60%  

---

## **Limitations**
– No direct comparison with original Q-learning BIC values  
– Simplified reward structure (assumed stationary)  
– No modeling of transfer/generalization (no ω parameter)  
– Single-parameter model limits expressiveness  
– Group definitions not independent of model fit  

---

## **Reproducibility**
```bash
pip install -r requirements.txt
```

---

## Acknowledgements
Data and experimental design from Bedi et al. (2025).
Special thanks to Gilles de Hollander for providing access to the behavioral dataset.  
Dataset available at [multlearn repository](https://github.com/ruffgroup/multlearn).

---

The Bayesian Beta-Binomial model captures meaningful individual differences in learning with minimal complexity (1 parameter), but fails to represent structured inference strategies observed in Transfer participants.  
Results support the view that human learning is heterogeneous, multiple computational strategies coexist, and that simple models capture only part of behavioral structure.
