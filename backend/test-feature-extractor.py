from ml.feature_extractor import extract_features

TEST_URLS = [
    "https://google.com",
    "https://example.xyz",
    "https://login-secure.top/account",
    "https://malicious.zip",
    "https://university.edu",
]

for url in TEST_URLS:
    print("=" * 60)
    print(url)
    print(extract_features(url))