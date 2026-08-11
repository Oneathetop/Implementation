====================================================
QR Phishing Detection Logistic Regression Evaluation =====================================================

Loading feature dataset... 

Dataset loaded successfully. 

Rows: 235370 

Columns: 27 

Preparing features and labels... 

Features: 26 

Samples: 235370 

Splitting dataset... 
Dataset split completed. 

Training samples: 188296 

Testing samples: 47074 

Training Logistic Regression... 
Logistic Regression training completed. 

Evaluating Logistic Regression... 
============================================================ 
            LOGISTIC REGRESSION RESULTS             
============================================================ 

Accuracy : 0.9941 
Precision: 0.9904 
Recall : 0.9995 
F1-Score : 0.9949 
ROC-AUC : 0.9965 

Confusion Matrix: [[19842 262] 
                   [ 14 26956]] 

Classification Report: 
     
                precision recall f1-score support 
    Legitimate  1.00       0.99    0.99    20104 
    Phishing    0.99       1.00    0.99    26970 
    
    accuracy                       0.99    47074 
    macro avg   0.99       0.99    0.99    47074 
    weighted avg 0.99      0.99    0.99    47074 