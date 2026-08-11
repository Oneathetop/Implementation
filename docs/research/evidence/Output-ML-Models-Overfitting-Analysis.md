(venv) C:\Implementation\backend>python -m ml.analyze_overfitting
============================================================
QR Phishing Detection - Overfitting Analysis
============================================================

Loading feature dataset...
Dataset loaded successfully.
Rows: 235370
Columns: 27

Preparing features and labels...
Features: 26
Samples: 235370

Cross-validation strategy:
Folds: 5
Strategy: Stratified K-Fold
Shuffle: True
Random state: 42


============================================================
OVERFITTING ANALYSIS - Logistic Regression
============================================================

Training vs Validation Performance:

ACCURACY
Training   : 0.9943 ± 0.0001
Validation : 0.9943 ± 0.0002
Gap        : 0.0000

PRECISION
Training   : 0.9907 ± 0.0001
Validation : 0.9907 ± 0.0003
Gap        : 0.0000

RECALL
Training   : 0.9994 ± 0.0001
Validation : 0.9994 ± 0.0002
Gap        : 0.0000

F1
Training   : 0.9950 ± 0.0001
Validation : 0.9950 ± 0.0002
Gap        : 0.0000

ROC_AUC
Training   : 0.9968 ± 0.0000
Validation : 0.9968 ± 0.0004
Gap        : 0.0000


============================================================
OVERFITTING ANALYSIS - Random Forest
============================================================

Training vs Validation Performance:

ACCURACY
Training   : 0.9977 ± 0.0000
Validation : 0.9955 ± 0.0001
Gap        : 0.0022

PRECISION
Training   : 0.9960 ± 0.0001
Validation : 0.9939 ± 0.0003
Gap        : 0.0021

RECALL
Training   : 0.9999 ± 0.0000
Validation : 0.9982 ± 0.0002
Gap        : 0.0017

F1
Training   : 0.9980 ± 0.0000
Validation : 0.9961 ± 0.0001
Gap        : 0.0019

ROC_AUC
Training   : 0.9996 ± 0.0000
Validation : 0.9969 ± 0.0001
Gap        : 0.0027


============================================================
OVERFITTING GAP COMPARISON
============================================================

Metric      Logistic Regression      Random Forest       
------------------------------------------------------------
Accuracy    0.0000                    0.0022
Precision   0.0000                    0.0021
Recall      0.0000                    0.0017
F1          0.0000                    0.0019
Roc_auc     0.0000                    0.0027

Interpretation:
A small training-validation gap indicates similar performance on training and validation data.
A large positive gap indicates that the model performs substantially better on training data than validation data.
The gap must be interpreted together with the cross-validation results and later domain-grouped evaluation.
