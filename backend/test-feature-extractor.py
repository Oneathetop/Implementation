from ml.feature_extractor import extract_features

TEST_URLS = [
    "https://google.com",

    "https://example.com/login#home",

    "https://paypal.com/account#verify",

    "https://example.com/#payment",
]

for url in TEST_URLS:
    print("=" * 60)
    print(url)
    print(extract_features(url))