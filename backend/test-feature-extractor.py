from ml.feature_extractor import extract_features

TEST_URLS = [
    "https://google.com",
    "https://example.com/login",
    "https://example.com/login?id=1",
    "https://example.com/login?id=1&token=abc",
    "https://example.com/login?id=1&token=abc&redirect=test"
]

for url in TEST_URLS:
    print("=" * 60)
    print(url)
    print(extract_features(url))