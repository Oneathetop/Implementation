Windows PowerShell 5.1
Copyright (C) Microsoft Corporation. All rights reserved.

PS C:\Users\Asus> Set-Location C:\Implementation\backend
PS C:\Implementation\backend> .\venv\Scripts\python.exe -c "import joblib; from ml.config import FEATURE_NAMES; model=joblib.load('models/random_forest_model.joblib'); actual=list(model.feature_names_in_); print('Feature count:', len(actual)); print('Feature order matches:', actual == FEATURE_NAMES); print('Model features:', actual)"
Feature count: 26
Feature order matches: True
Model features: ['url_length', 'domain_length', 'path_length', 'slash_count', 'double_slash_in_path', 'suspicious_file_extension', 'subdomain_count', 'query_parameter_count', 'https', 'ip_address', 'dot_count', 'digit_count', 'max_consecutive_digits', 'hyphen_count', 'special_character_count', 'at_symbol_presence', 'suspicious_character_ratio', 'url_encoding_count', 'entropy', 'consecutive_special_character_count', 'keyword_count', 'brand_keyword_count', 'url_shortener', 'port_number', 'suspicious_tld', 'fragment_present']
PS C:\Implementation\backend> .\venv\Scripts\python.exe -c "import pandas as pd; from ml.config import FEATURE_NAMES, FEATURE_DATASET; df=pd.read_csv(FEATURE_DATASET); print('Missing columns:', [name for name in FEATURE_NAMES if name not in df.columns]); print('Column count:', len(df.columns)); print('Expected feature columns:', len(FEATURE_NAMES))"
Missing columns: []
Column count: 27
Expected feature columns: 26
PS C:\Implementation\backend> @'
>> import pandas as pd
>>
>> from ml.config import CLEAN_DATASET, FEATURE_DATASET, FEATURE_NAMES
>> from ml.feature_extractor import extract_features
>>
>> clean_df = pd.read_csv(CLEAN_DATASET)
>> saved_features = pd.read_csv(FEATURE_DATASET)
>>
>> current_features = pd.DataFrame(
>>     [
>>         extract_features(url)
>>         for url in clean_df["URL"]
>>     ]
>> )
>>
>> mismatches = []
>>
>> for feature_name in FEATURE_NAMES:
>>     saved_values = saved_features[feature_name]
>>     current_values = current_features[feature_name]
>>
>>     different_rows = saved_values != current_values
>>
>>     if different_rows.any():
>>         mismatches.append(
>>             {
>>                 "feature": feature_name,
>>                 "different_rows": int(different_rows.sum()),
>>             }
>>         )
>>
>> print("Rows checked:", len(clean_df))
>> print("Feature mismatches:", mismatches)
>>
>> if mismatches:
>>     print("RESULT: The saved feature dataset differs from the current extractor.")
>> else:
>>     print("RESULT: The saved feature dataset matches the current extractor.")
>> '@ | .\venv\Scripts\python.exe
Rows checked: 235370
Feature mismatches: [{'feature': 'keyword_count', 'different_rows': 52}]
RESULT: The saved feature dataset differs from the current extractor.
PS C:\Implementation\backend>
PS C:\Implementation\backend>
PS C:\Implementation\backend>
PS C:\Implementation\backend> .\venv\Scripts\python.exe -m ml.build_features

============================================================
QR Phishing Detection - Feature Engineering
============================================================

Loading cleaned dataset...
Dataset loaded successfully.
Rows: 235370
Columns: 2

Validating dataset...
Dataset validation passed.

Extracting features...
Feature extraction completed.
Rows: 235370
Columns: 27

Validating feature dataset...
Feature dataset validation passed.

Saving feature dataset...
Feature dataset saved to:
C:\Implementation\backend\datasets\processed\feature_dataset.csv
PS C:\Implementation\backend> .\venv\Scripts\python.exe -m ml.train_RF_model

============================================================
QR Phishing Detection - Random Forest Training
============================================================

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

Training Random Forest...
Random Forest training completed.

Saving trained model to models/random_forest_model.joblib...
Model saved successfully.
PS C:\Implementation\backend> @'
>> import pandas as pd
>>
>> from ml.config import CLEAN_DATASET, FEATURE_DATASET, FEATURE_NAMES
>> from ml.feature_extractor import extract_features
>>
>> clean_df = pd.read_csv(CLEAN_DATASET)
>> saved_features = pd.read_csv(FEATURE_DATASET)
>>
>> current_features = pd.DataFrame(
>>     [
>>         extract_features(url)
>>         for url in clean_df["URL"]
>>     ]
>> )
>>
>> mismatches = []
>>
>> for feature_name in FEATURE_NAMES:
>>     saved_values = saved_features[feature_name]
>>     current_values = current_features[feature_name]
>>
>>     different_rows = saved_values != current_values
>>
>>     if different_rows.any():
>>         mismatches.append(
>>             {
>>                 "feature": feature_name,
>>                 "different_rows": int(different_rows.sum()),
>>             }
>>         )
>>
>> print("Rows checked:", len(clean_df))
>> print("Feature mismatches:", mismatches)
>>
>> if mismatches:
>>     print("RESULT: Mismatches remain.")
>> else:
>>     print("RESULT: The saved feature dataset matches the current extractor.")
>> '@ | .\venv\Scripts\python.exe
Rows checked: 235370
Feature mismatches: []
RESULT: The saved feature dataset matches the current extractor.
PS C:\Implementation\backend> .\venv\Scripts\python.exe -m ml.evaluate_RF_model
============================================================
QR Phishing Detection - Random Forest Evaluation
============================================================

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

Training Random Forest...
Random Forest training completed.

Evaluating Random Forest...


============================================================
RANDOM FOREST RESULTS
============================================================
Accuracy : 0.9956
Precision: 0.9977
Recall   : 0.9920
F1-Score : 0.9949
ROC-AUC  : 0.9975

Confusion Matrix:
Rows = actual [Phishing, Legitimate]
Columns = predicted [Phishing, Legitimate]
[[19943   161]
 [   45 26925]]

Classification Report:
                precision    recall  f1-score   support

  Phishing (0)       1.00      0.99      0.99     20104
Legitimate (1)       0.99      1.00      1.00     26970

      accuracy                           1.00     47074
     macro avg       1.00      1.00      1.00     47074
  weighted avg       1.00      1.00      1.00     47074

PS C:\Implementation\backend> .\venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
INFO:     Started server process [16052]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [16052]
PS C:\Implementation\backend> .\venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
INFO:     Started server process [22076]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:56261 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:56261 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:63274 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:63274 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:64583 - "POST /predict HTTP/1.1" 200 OK
INFO:     127.0.0.1:64587 - "POST /predict HTTP/1.1" 200 OK