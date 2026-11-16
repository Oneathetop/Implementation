from __future__ import annotations

from difflib import SequenceMatcher
from urllib.parse import urlparse

import tldextract


TRUSTED_ROOT_DOMAINS = {
    "apple.com",
    "google.com",
    "microsoft.com",
    "youtube.com",
}

BRAND_NAMES = {
    "apple",
    "facebook",
    "google",
    "microsoft",
    "youtube",
}


def get_root_domain(url: str) -> str:
    parsed = tldextract.extract(url)

    if parsed.domain and parsed.suffix:
        return f"{parsed.domain}.{parsed.suffix}".lower()

    return parsed.domain.lower()


def is_trusted_root_domain(url: str) -> bool:
    return get_root_domain(url) in TRUSTED_ROOT_DOMAINS


def find_brand_lookalike(url: str) -> str | None:
    root_domain = get_root_domain(url)
    root_label = root_domain.rsplit(".", 1)[0]
    tokens = [root_label] + [token for token in root_label.split("-") if token]

    for token in tokens:
        for brand in BRAND_NAMES:
            if token == brand:
                if root_domain not in TRUSTED_ROOT_DOMAINS:
                    return brand
                continue

            if len(token) >= 4 and SequenceMatcher(None, token, brand).ratio() >= 0.8:
                return brand

    return None
