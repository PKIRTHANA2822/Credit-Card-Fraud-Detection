# Project: 
- Credit Card Fraud Detection (with Imbalanced Learning)
![maxresdefault](https://github.com/user-attachments/assets/f11800b9-bbc2-4620-965e-43345e40c099)
# Objective
- Build a robust ML system that identifies fraudulent transactions (Class = 1) from normal ones (Class = 0) in creditcard.csv, maximizing detection of true frauds while minimizing false alarms.
# Why this project
- Business impact: Reduces financial loss, chargebacks, and reputational risk.
- Security: Supports real-time monitoring and case investigation.
- Data-science value: Classic highly imbalanced binary classification problem—great for applying resampling, cost-sensitive learning, and careful evaluation.
# Step-by-step approach
- Load & split
- Read creditcard.csv.
- Train/test split with stratification to preserve class ratio.
- Preprocess
- Scale numeric features (notably Time, Amount; PCA features V1..V28 are already scaled).
- Use pipelines to avoid leakage.
- Baseline models
- Logistic Regression and Random Forest (with and without class_weight).
- Handle imbalance
- Try Under-sampling (NearMiss), Over-sampling (RandomOverSampler), and Hybrid (SMOTETomek) on the training set only.
# Model selection
- Hyperparameter tuning with GridSearchCV and StratifiedKFold CV.
- Evaluate
- Focus on Recall (class 1), Precision (class 1), F1 (class 1), ROC-AUC, and especially PR-AUC.
- Inspect confusion matrix.
# Explainability & stability
- Coefficients (LogReg), Feature/importances (RF), threshold tuning via Precision-Recall trade-off.
- Finalize
- Choose best combo (model + resampling + threshold), export pipeline, document operating threshold.
# Exploratory Data Analysis (EDA)
- Target distribution: df.Class.value_counts() → quantify imbalance (fraud ≪ non-fraud).
- Feature overview: Summary stats for Time, Amount; distributions by class.
- Leakage check: Ensure no post-transaction info is included.
- Correlation/importance probes: Quick RF feature importances to spot signal in Amount/PCA components.
- Temporal drift: Plot fraud rate vs Time buckets.
# Feature Selection
- Since V1..V28 are PCA-derived, multicollinearity is reduced.
- Keep Amount and optionally engineered transforms (log/boxcox of Amount).
- Use model-based selection:
- Logistic Regression (L1) to induce sparsity.
- Random Forest importances to drop consistently uninformative features.
- Avoid exhaustive selection; prefer pipeline-based selection validated via CV.
# Feature Engineering
- Scaling: Standardize Amount (and Time if retained).
- Binning/time features (optional): Hour-of-day or segments from Time.
- Outlier-aware transforms: Amount_log = log1p(Amount).
- Class-prior features: None needed; rely on resampling/cost-sensitive learning.
# Model Training
- Logistic Regression
- Grid over C and penalty with a solver that supports L1 (liblinear or saga).
- Use class_weight='balanced' as a baseline.
- Random Forest
- Tune n_estimators, max_depth, min_samples_split, class_weight (balanced or custom {0:1, 1:100}).
- Resampling strategies (training only)
- NearMiss (under-sampling): NearMiss(sampling_strategy=0.8).
- RandomOverSampler: RandomOverSampler(sampling_strategy=0.5).
- SMOTETomek: SMOTETomek(sampling_strategy=0.5).
# Pipelines
- Pipeline([('scaler', StandardScaler()), ('sampler', SMOTETomek(...)), ('clf', LogisticRegression(...))])
- Tune sampler ratio + model params jointly via GridSearchCV.
- Cross-validation
- Use StratifiedKFold(n_splits=5, shuffle=True, random_state=...).
- Optimize for F1 (class 1) or average='macro', and also track PR-AUC.
# Model Testing (Evaluation)
- On the untouched test set:
- Confusion matrix to see TP/FP/FN/TN.
- Classification report: precision/recall/F1 per class.
- ROC-AUC and PR-AUC (more informative for imbalance).
- Threshold tuning: choose a probability cutoff that meets the business recall target (e.g., ≥90% recall on fraud) while keeping precision acceptable.
# Output (Deliverables)
<img width="544" height="295" alt="Screenshot 2025-08-16 153524" src="https://github.com/user-attachments/assets/d9963934-b893-4f5d-81ea-d17a178aef9c" />
