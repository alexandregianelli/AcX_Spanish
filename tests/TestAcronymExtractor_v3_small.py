import unittest
from acrodisam.AcronymExtractors.AcronymExtractor_v3_small import AcronymExtractor_v3_small
from acrodisam.DataCreators import ArticleDB
from acrodisam.string_constants import file_msh_articleDB


class TestAcronymExtractor_v3_small(unittest.TestCase):
    
    mshArticleDB = ArticleDB.load(file_msh_articleDB)
        
    def test_SomeCases(self):
        text = "A Aa tPa tPA First eCG USA Ala"
        acronyms = AcronymExtractor_v3_small().get_acronyms(text)
        self.assertTrue("ECG" in acronyms)
        self.assertTrue("TPA" in acronyms)
        self.assertTrue("USA" in acronyms)
        self.assertFalse("Ala" in acronyms)
        self.assertEqual(len(acronyms), 3)
    
    def test_MSH_articleID20118811(self):
        text = TestAcronymExtractor_v3_small.mshArticleDB["20118811"]
        acronyms = AcronymExtractor_v3_small().get_acronyms(text)
        self.assertTrue("DVT" in acronyms)
        self.assertTrue("TPA" in acronyms)
        self.assertEqual(len(acronyms), 2)
        