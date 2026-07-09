import math
import re
from urllib.parse import urlparse

def extract_url_length(url):
    """
    Return the total length of the URL.
    """

    return len(url)

def extract_https(url):
    """
    Check whether the URL uses HTTPS.

    Returns
    -------
    int
        1 if HTTPS is used, otherwise 0.
    """

    return int(url.lower().startswith("https://"))


def extract_features(url):
    """
    Extract lexical features from a URL.
    """

    parsed = urlparse(url)

    features = {
        "url_length": extract_url_length(url),
        "https": extract_https(url),
    }

    return features