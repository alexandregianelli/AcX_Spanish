import unittest

from acrodisam.TextTools import getCleanedWords


class TestTextTools(unittest.TestCase):

    def test_getCleanedWords_noStemming_Punctuations(self):
        words = getCleanedWords(
            "Purpose, of; his. life: with\r\n \"productive\" achievement\n (as) <his> noblest! activity?", stem_words=False, removeNumbers=False)
        self.assertTrue(len(words) == 6)
        self.assertEquals(
            words, ['purpose', 'life', 'productive', 'achievement', 'noblest', 'activity'])

    def test_getCleanedWords_blankString(self):
        words = getCleanedWords(
            "", stem_words=False, removeNumbers=False)
        self.assertEquals(words, [])

    def test_getCleanedWords_noStemming_stopwordsWithCapitals(self):
        words = getCleanedWords(
            "This Is A Test String", stem_words=False, removeNumbers=False)
        self.assertEquals(words, ["test", "string"])

    def test_getCleanedWords_noStemming_stopwords(self):
        words = getCleanedWords(
            "this is a test string", stem_words=False, removeNumbers=False)
        self.assertEquals(words, ["test", "string"])

    def test_getCleanedWords_removeNumbersAndPercentageAndFormulas(self):
        inputText = "This is a sentence with 01 2 10 numbers and 10% and 1+2=5."
        words = getCleanedWords(
            inputText, stem_words=False, removeNumbers=True)
        self.assertEquals(words, ["sentence", "numbers"])
