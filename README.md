# Bot Account Detection

A Kaggle‐style end-to-end pipeline for detecting bot accounts on social media using Python and scikit-learn (plus LightGBM & CatBoost).  

## 🚀 Project Overview

This project tackles a binary classification problem—identifying whether a social media account on twitter is a bot or human. It features:

- **Extensive feature engineering**  
  - Temporal features (weekday, hour)  
  - Activity ratios (statuses-per-day, favourites-per-day, friends/followers)  
  - Log-scaled counts for skewed distributions  
  - Text statistics from user descriptions (length, hashtags, mentions, uppercase ratio, etc.)  
  - Frequency encoding for `lang` & `location`  
  - TF–IDF vectorization of descriptions + Truncated SVD for dimensionality reduction  

- **Modeling & ensembling**  
  - Six calibrated base learners:  
    - SGD with ElasticNet penalty  
    - Scikit-learn’s HistGradientBoosting  
    - LightGBM (DART booster)  
    - CatBoost  
    - RandomForest  
    - AdaBoost  
  - Early stopping on boosting models  
  - Weighted blend of predictions  
  - Final stacking with logistic regression  

- **Robust evaluation**  
  - 15 % hold-out for early stopping  
  - Chronological 70/30 back-test with ROC-AUC  
  - Final submission on Kaggle achieving > 0.90 AUC  
