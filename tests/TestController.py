import unittest

from acrodisam.AcronymDisambiguator import AcronymDisambiguator

class TestAcronymDisambiguator(unittest.TestCase):
    
    def test_processChoices_basic(self):
        
        disambiguator = AcronymDisambiguator(text_extractor=None, acronym_extractor=None, expanders=None, articleDB=None, acronymDB=None)
        
        choices = ["nuclear pore complexes",
                   "nuclear pore complex",
                   "polychlorinated biphenyl",
                   "t replacing factor",
                   "this supernatant factor",
                   "polychlorinated biphenyls"]
        
        labels, classToExpansion = disambiguator._processChoices(choices)
        
        self.assertNotEquals(labels, [0,0,2,3,4,2])
        self.assertNotEquals(classToExpansion, {0:"nuclear pore complexes", 2:"polychlorinated biphenyl", 3:"t replacing factor", 4:"this supernatant factor"})
        
        self.assertEquals(labels, [0,0,1,2,3,1])
        self.assertEquals(classToExpansion, {0:"nuclear pore complexes", 1:"polychlorinated biphenyl", 2:"t replacing factor", 3:"this supernatant factor"})

    def test_processChoices_blank(self):
        disambiguator = AcronymDisambiguator(text_extractor=None, acronym_extractor=None, expanders=None, articleDB=None, acronymDB=None)
        labels, classToExpansion = disambiguator._processChoices([])
        
        self.assertEquals(labels, [])
        self.assertEquals(classToExpansion, {})
