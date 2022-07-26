"""Unit tests for the creation of a dict from BIO tags and transforming tokens into raw text

Created May, 2021

@author:JRCasanova
"""

import unittest
from DatasetParsers.process_tokens_and_bio_tags import create_diction
from DatasetParsers.process_tokens_and_bio_tags import tokens_to_raw_text


class TestTokensAndBioTagsProcessor(unittest.TestCase):
    """Tests the creation of a dict with acronyms as keys and expansions as values from BIO tags as well as transforming tokens into raw text"""

    def test_bio_tags_to_dict(self):
        """Tests the create_diction function"""

        # Just one acronym with expansion in text
        # one of the parenthesis does not have an acronym inside
        tokens = [
            "Quantum",
            "key",
            "distribution",
            "(",
            "QKD",
            ")",
            "at",
            "telecom",
            "wavelengths",
            "(",
            "1260",
            "-",
            "1625nm",
            ")",
            "has",
            "the",
            "potential",
            "for",
            "fast",
            "deployment",
            "due",
            "to",
            "existing",
            "optical",
            "fibre",
            "infrastructure",
            "and",
            "mature",
            "telecom",
            "technologies",
            ".",
        ]

        labels = [
            "B-long",
            "I-long",
            "I-long",
            "O",
            "B-short",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ]

        correct_dict = {"QKD": "Quantum key distribution"}

        predicted_dict = create_diction(tokens, labels)

        self.assertEqual(predicted_dict, correct_dict)

        # Two acronyms with expansions in text
        # One of them appears several times
        # Another appears in a different phrase than the expansion
        tokens = [
            "The",
            "Central",
            "Intelligence",
            "Agency",
            "(",
            "CIA",
            ")",
            "is",
            "a",
            "civilian",
            "foreign",
            "intelligence",
            "service",
            "of",
            "the",
            "federal",
            "government",
            "of",
            "the",
            "United",
            "States",
            ".",
            "The",
            "CIA",
            "reports",
            "to",
            "the",
            "Director",
            "of",
            "National",
            "Intelligence",
            ".",
            "The",
            "CIA",
            "as",
            "transfered",
            "some",
            "of",
            "its",
            "powers",
            "to",
            "the",
            "DNI",
            ".",
        ]

        labels = [
            "O",
            "B-long",
            "I-long",
            "I-long",
            "O",
            "B-short",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-short",
            "O",
            "O",
            "O",
            "B-long",
            "I-long",
            "I-long",
            "I-long",
            "O",
            "O",
            "B-short",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-short",
            "O",
        ]

        correct_dict = {
            "CIA": "Central Intelligence Agency",
            "DNI": "Director of National Intelligence",
        }

        predicted_dict = create_diction(tokens, labels)

        self.assertEqual(predicted_dict, correct_dict)

        # One acronym with expansion in text
        # Another without expansion in text
        tokens = [
            "An",
            "internet",
            "service",
            "provider",
            "(",
            "ISP",
            "for",
            "short",
            ")",
            "provides",
            "access",
            "to",
            "the",
            "WWW",
            ".",
        ]

        labels = [
            "O",
            "B-long",
            "I-long",
            "I-long",
            "O",
            "B-short",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-short",
            "O",
        ]

        correct_dict = {"ISP": "internet service provider", "WWW": None}

        predicted_dict = create_diction(tokens, labels)

        self.assertEqual(predicted_dict, correct_dict)

        # One acronym that has several tokens
        tokens = [
            "Goh",
            "Keng",
            "Swee",
            "Command",
            "and",
            "Staff",
            "College",
            "(",
            "GKS",
            "CSC",
            ")",
            ".",
        ]

        labels = [
            "B-long",
            "I-long",
            "I-long",
            "I-long",
            "I-long",
            "I-long",
            "I-long",
            "O",
            "B-short",
            "I-short",
            "O",
            "O",
        ]

        correct_dict = {"GKS CSC": "Goh Keng Swee Command and Staff College"}

        predicted_dict = create_diction(tokens, labels)

        self.assertEqual(predicted_dict, correct_dict)

    def test_tokens_to_raw_text(self):

        tokens = [
            "Yesterday",
            "we",
            "had",
            "class",
            "at",
            "Instituto",
            "Superior",
            "Técnico",
            "(",
            "IST",
            ")",
            ".",
        ]

        correct_sentence = "Yesterday we had class at Instituto Superior Técnico (IST)."

        predicted_sentence = tokens_to_raw_text(tokens)

        self.assertEqual(correct_sentence, predicted_sentence)

        tokens = [
            "When",
            "I",
            "went",
            "to",
            "class",
            "I",
            "heard",
            "someone",
            "say",
            ":",
            '"',
            "Tomorrow",
            "is",
            "going",
            "to",
            "rain",
            "!",
            '"',
            ".",
        ]

        correct_sentence = (
            'When I went to class I heard someone say: "Tomorrow is going to rain!".'
        )

        predicted_sentence = tokens_to_raw_text(tokens)

        self.assertEqual(correct_sentence, predicted_sentence)

        tokens = [
            "When",
            "I",
            "went",
            "to",
            "class",
            "I",
            "heard",
            "someone",
            "say",
            ":",
            "'",
            "Tomorrow",
            "is",
            "going",
            "to",
            "rain",
            "!",
            "'",
            ".",
        ]

        correct_sentence = (
            "When I went to class I heard someone say: 'Tomorrow is going to rain!'."
        )

        predicted_sentence = tokens_to_raw_text(tokens)

        self.assertEqual(correct_sentence, predicted_sentence)


if __name__ == "__main__":
    unittest.main()
