import unittest

from services.domain_policy import (
    find_brand_lookalike,
    get_root_domain,
    is_trusted_root_domain,
)


class DomainPolicyTests(unittest.TestCase):
    def test_recognizes_trusted_subdomain(self):
        self.assertEqual(
            get_root_domain("https://docs.google.com/forms"),
            "google.com",
        )
        self.assertTrue(is_trusted_root_domain("https://docs.google.com/forms"))

    def test_rejects_trusted_brand_in_attacker_domain(self):
        url = "https://google.com.attacker.example/login"

        self.assertFalse(is_trusted_root_domain(url))
        self.assertEqual(find_brand_lookalike(url), None)

    def test_detects_lookalike_brand_domain(self):
        self.assertEqual(
            find_brand_lookalike("https://facebok-login.com"),
            "facebook",
        )


if __name__ == "__main__":
    unittest.main()
