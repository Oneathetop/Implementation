====================================================QR Phishing Detection - Random Forest Evaluation ====================================================

Loading feature dataset... 

Dataset loaded successfully. 

Rows: 235370 Columns: 27 

Preparing features and labels... 

Features: 26 

Samples: 235370 

Splitting dataset... 

Dataset split completed. 

Training samples: 188296 

Testing samples: 47074 

Training Random Forest... 

Random Forest training completed. 

Evaluating Random Forest... 

====================================================             RANDOM FOREST RESULTS ====================================================

Accuracy : 0.9956 
Precision: 0.9941 
Recall : 0.9983 
F1-Score : 0.9962 
ROC-AUC : 0.9974 

Confusion Matrix: [[19943 161] 
                   [ 45 26925]] 

Classification Report: 
            precision recall f1-score support
        
Legitimate  1.00       0.99     0.99    20104 Phishing    0.99       1.00     1.00    26970 

accuracy                        1.00    47074 
macro avg   1.00       1.00     1.00    47074 weighted avg 1.00      1.00     1.00    47074