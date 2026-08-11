(venv) C:\Implementation\backend>python -m ml.cross_validate_models
============================================================
QR Phishing Detection - 5-Fold Cross-Validation
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
Evaluating Logistic Regression
============================================================

Fold Results:

Fold 1
Accuracy : 0.9944
Precision: 0.9911
Recall   : 0.9993
F1-Score : 0.9952
ROC-AUC  : 0.9966

Fold 2
Accuracy : 0.9940
Precision: 0.9903
Recall   : 0.9993
F1-Score : 0.9948
ROC-AUC  : 0.9962

Fold 3
Accuracy : 0.9942
Precision: 0.9908
Recall   : 0.9991
F1-Score : 0.9950
ROC-AUC  : 0.9971

Fold 4
Accuracy : 0.9945
Precision: 0.9908
Recall   : 0.9996
F1-Score : 0.9952
ROC-AUC  : 0.9972

Fold 5
Accuracy : 0.9942
Precision: 0.9903
Recall   : 0.9996
F1-Score : 0.9950
ROC-AUC  : 0.9967

Mean ± Standard Deviation:
Accuracy  : 0.9943 ± 0.0002
Precision : 0.9907 ± 0.0003
Recall    : 0.9994 ± 0.0002
F1        : 0.9950 ± 0.0002
Roc_auc   : 0.9968 ± 0.0004


============================================================
Evaluating Random Forest
============================================================

Fold Results:

Fold 1
Accuracy : 0.9957
Precision: 0.9945
Recall   : 0.9980
F1-Score : 0.9962
ROC-AUC  : 0.9972

Fold 2
Accuracy : 0.9955
Precision: 0.9940
Recall   : 0.9982
F1-Score : 0.9961
ROC-AUC  : 0.9968

Fold 3
Accuracy : 0.9955
Precision: 0.9937
Recall   : 0.9984
F1-Score : 0.9960
ROC-AUC  : 0.9970

Fold 4
Accuracy : 0.9953
Precision: 0.9938
Recall   : 0.9980
F1-Score : 0.9959
ROC-AUC  : 0.9969

Fold 5
Accuracy : 0.9956
Precision: 0.9938
Recall   : 0.9986
F1-Score : 0.9962
ROC-AUC  : 0.9968

Mean ± Standard Deviation:
Accuracy  : 0.9955 ± 0.0001
Precision : 0.9939 ± 0.0003
Recall    : 0.9982 ± 0.0002
F1        : 0.9961 ± 0.0001
Roc_auc   : 0.9969 ± 0.0001


============================================================
MODEL CROSS-VALIDATION COMPARISON
============================================================

Metric      Logistic Regression      Random Forest       
------------------------------------------------------------
Accuracy    0.9943 ± 0.0002        0.9955 ± 0.0001
Precision   0.9907 ± 0.0003        0.9939 ± 0.0003
Recall      0.9994 ± 0.0002        0.9982 ± 0.0002
F1          0.9950 ± 0.0002        0.9961 ± 0.0001
Roc_auc     0.9968 ± 0.0004        0.9969 ± 0.0001
