Windows PowerShell 5.1
Copyright (C) Microsoft Corporation. All rights reserved.

PS C:\Users\Asus> $body = @{
>>     url = "http://login-verify-account.example.xyz/update"
>> } | ConvertTo-Json
PS C:\Users\Asus>
PS C:\Users\Asus> Invoke-RestMethod `
>>     -Uri "http://127.0.0.1:8000/predict" `
>>     -Method Post `
>>     -ContentType "application/json" `
>>     -Body $body


url            : http://login-verify-account.example.xyz/update
prediction     : Malicious
risk_score     : 1.0
risk_level     : High Risk
risk_color     : red
explanation    : {The model estimated a 100.00% probability that this URL
                 is phishing., The URL does not use HTTPS. Observed value:
                 0.0. Model contribution toward phishing: 0.1663., The URL
                 uses a top-level domain associated with suspicious URLs.
                 Observed value: 1.0. Model contribution toward phishing:
                 0.1132., The URL contains an unusual number of path
                 separators. Observed value: 1.0. Model contribution
                 toward phishing: 0.0906....}
recommendation : Do not open this link. Verify the source before
                 continuing.



PS C:\Users\Asus> ^C
PS C:\Users\Asus> $body = @{
>>     url = "https://example.com"
>> } | ConvertTo-Json
PS C:\Users\Asus>
PS C:\Users\Asus> Invoke-RestMethod `
>>     -Uri "http://127.0.0.1:8000/predict" `
>>     -Method Post `
>>     -ContentType "application/json" `
>>     -Body $body


url            : https://example.com
prediction     : Malicious
risk_score     : 1.0
risk_level     : High Risk
risk_color     : red
explanation    : {The model estimated a 100.00% probability that this URL
                 is phishing., The URL contains multiple subdomains.
                 Observed value: 0.0. Model contribution toward phishing:
                 0.4127., The URL contains an unusual number of dots.
                 Observed value: 1.0. Model contribution toward phishing:
                 0.3591.}
recommendation : Do not open this link. Verify the source before
                 continuing.

