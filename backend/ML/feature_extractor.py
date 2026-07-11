import math
import re
import ipaddress
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "bank",
    "password",
    "signin",
    "confirm",
    "wallet",
    "payment",
    "paypal",
    "microsoft",
    "apple",
    "amazon",
    "google",
    "facebook",
}

URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "tiny.cc",
    "rb.gy",
    "shorturl.at",
}

SUSPICIOUS_TLDS = {
    "xyz",
    "top",
    "click",
    "work",
    "gq",
    "tk",
    "cf",
    "ml",
    "ga",
    "zip",
    "country",
}

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

def extract_domain_length(parsed_url):
    """
    Return the length of the domain (netloc).
    """

    return len(parsed_url.netloc)

def extract_path_length(parsed_url):
    """
    Return the length of the URL path.
    """

    return len(parsed_url.path)

def extract_dot_count(url):
    """
    Count the number of dots in the URL.
    """

    return url.count(".")

def extract_digit_count(url):
    """
    Count numeric characters in the URL.
    """

    return sum(character.isdigit() for character in url)

def extract_hyphen_count(url):
    """
    Count hyphens in the URL.
    """

    return url.count("-")

def extract_special_character_count(url):
    """
    Count common special characters.
    """

    special_characters = "@?=&%_"

    return sum(url.count(character) for character in special_characters)

def extract_ip_address(parsed_url):
    """
    Check whether the URL uses an IP address instead of a domain.

    Returns
    -------
    int
        1 if the hostname is an IP address, otherwise 0.
    """

    hostname = parsed_url.hostname

    if hostname is None:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0

def extract_subdomain_count(parsed_url):
    """
    Count the number of subdomains.

    IP addresses are treated as having zero subdomains.

    Examples:
    google.com -> 0
    www.google.com -> 1
    login.mail.google.com -> 3
    """

    hostname = parsed_url.hostname

    if hostname is None:
        return 0
    
    # IP addresses do not have subdomains
    if extract_ip_address(parsed_url):
        return 0

    parts = hostname.split(".")

    if len(parts) <= 2:
        return 0

    return len(parts) - 2

def extract_keyword_count(url):
    """
    Count suspicious phishing-related keywords appearing in the URL.
    """

    url = url.lower()

    return sum(
        keyword in url
        for keyword in SUSPICIOUS_KEYWORDS
    )

def extract_url_shortener(parsed_url):
    """
    Check whether the URL belongs to a known URL shortening service.

    Returns
    -------
    int
        1 if the domain is a known URL shortener, otherwise 0.
    """

    hostname = parsed_url.hostname

    if hostname is None:
        return 0

    hostname = hostname.lower()

    return int(any(
        hostname == shortener or hostname.endswith("." + shortener)
        for shortener in URL_SHORTENERS
    ))

def extract_port_number(parsed_url):
    """
    Detect whether the URL explicitly specifies a port.

    Returns
    -------
    int
        1 if a port is explicitly specified, otherwise 0.
    """

    return int(parsed_url.port is not None)

def extract_query_parameter_count(parsed_url):
    """
    Count the number of query parameters in the URL.

    Example:
    ?id=1&token=abc

    returns 2
    """

    if not parsed_url.query:
        return 0

    return len(parsed_url.query.split("&"))

def extract_url_encoding_count(url):
    """
    Count URL-encoded character sequences.

    Examples:
    %20
    %2F
    %3A
    """

    return len(re.findall(r"%[0-9A-Fa-f]{2}", url))

def extract_entropy(url):
    """
    Calculate Shannon entropy of a URL.

    Higher entropy indicates a more random-looking URL.
    """

    if not url:
        return 0.0

    entropy = 0.0

    for character in set(url):
        probability = url.count(character) / len(url)
        entropy -= probability * math.log2(probability)

    return round(entropy, 4)

def extract_suspicious_tld(parsed_url):
    """
    Check whether the URL uses a suspicious top-level domain.

    Returns
    -------
    int
        1 if the TLD is in the suspicious list, otherwise 0.
    """

    hostname = parsed_url.hostname

    if hostname is None:
        return 0

    parts = hostname.lower().split(".")

    if len(parts) < 2:
        return 0

    tld = parts[-1]

    return int(tld in SUSPICIOUS_TLDS)

def extract_features(url):
    """
    Extract lexical features from a URL.
    """

    parsed = urlparse(url)

    features = {

        # URL Structure Features
        "url_length": extract_url_length(url),
        "domain_length": extract_domain_length(parsed),
        "path_length": extract_path_length(parsed),
        "subdomain_count": extract_subdomain_count(parsed),
        "query_parameter_count": extract_query_parameter_count(parsed),

        # Security Protocol Features
        "https": extract_https(url),
        "ip_address": extract_ip_address(parsed),

        # Character Count Features
        "dot_count": extract_dot_count(url),
        "digit_count": extract_digit_count(url),
        "hyphen_count": extract_hyphen_count(url),
        "special_character_count": extract_special_character_count(url),
        "url_encoding_count": extract_url_encoding_count(url),
        "entropy": extract_entropy(url),

        # Phishing Indicators
        "keyword_count": extract_keyword_count(url),
        "url_shortener": extract_url_shortener(parsed),
        "port_number": extract_port_number(parsed),
        "suspicious_tld": extract_suspicious_tld(parsed),
    }

    return features