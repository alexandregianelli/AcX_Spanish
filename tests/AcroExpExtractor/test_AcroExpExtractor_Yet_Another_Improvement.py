'''
Created on Jul 19, 2019

@author: jpereira
'''
import unittest
from AcroExpExtractors.AcroExpExtractor_Yet_Another_Improvement import AcroExpExtractor_Yet_Another_Improvement

class Test_AcroExpExtractor_Yet_Another_Improvement(unittest.TestCase):
    someSentence = "Test sentence blabla more text some more."
    
    sentenceParenthesis1 = "Test sentence EXP (Expansion xpto palalalalal) bla bla some more text."
    sentenceParenthesis2 = "Test sentence expansion (EXP) bla bla some more text."

    sentenceHyphen = "Test sentence EXP - Expansion bla bla some more text."
    sentenceColon = "Test sentence EXP:Expansion bla bla some more text."
    
    validAcronyms = ["AAA", "LDA", "IST", "ACRO", "BG", "myc-glu-glu", "MPB", "T3S", "Asf1", "SSAT", "IPCs", "MPB", "PDMAEMA", "[(18)F]FLT", "AA AA"]
    
    invalidAcronyms = ["THAT", "11233", "sdodssoddsosd", "A", "", "AA AA AA"]

    def test_select_expansion(self):
        acroExp = AcroExpExtractor_Yet_Another_Improvement()
        
        # Suppose we found the acronym on the right of the expansion
        newExpansion = acroExp.select_expansion("EXP", "expannot Test sentence expansion a")
        self.assertEqual(newExpansion, "expansion")
        
        newExpansion = acroExp.select_expansion("EXP", "Expansion bla bla some more text")
        self.assertEqual(newExpansion, "Expansion")

        newExpansion = acroExp.select_expansion("5-HIAA", "bla dssdd  5--hydroxy-indole-acetic acid")
        self.assertEqual(newExpansion, "5--hydroxy-indole-acetic acid")
        
        newExpansion = acroExp.select_expansion("nNOS", "ndsdd nervours NOS")
        self.assertEqual(newExpansion, "nervours NOS")
        
        newExpansion = acroExp.select_expansion("2AAF", "2-acetylaminofluorene")
        self.assertEqual(newExpansion, "2-acetylaminofluorene")
        
        newExpansion = acroExp.select_expansion("PDMAEMA", "poly2-(dimethylamino) poly[2-(dimethylamino) ethyl methacrylate]")
        self.assertEqual(newExpansion, "poly[2-(dimethylamino) ethyl methacrylate]")

        newExpansion = acroExp.select_expansion("2’5’-Oas", "2’5’-oligoadenyllate synthetase")
        self.assertEqual(newExpansion, "2’5’-oligoadenyllate synthetase")
        
    #    newExpansion = acroExp.select_expansion("ωτ", "wildtype")
    #    self.assertEqual(newExpansion, "wildtype")
        
        newExpansion = acroExp.select_expansion("GNAT", "Gcn5 Gcn5-related N-acetyltransferase")
        self.assertEqual(newExpansion, "Gcn5-related N-acetyltransferase")        

    #    newExpansion = acroExp.select_expansion("[(18)F]FLT", "3-deoxy-3-(18)F-fluoro-l-thymidine")
    #    self.assertEqual(newExpansion, "3-deoxy-3-(18)F-fluoro-l-thymidine")   

        newExpansion = acroExp.select_expansion("SSAT", "sdddsd sdds spermidine/spermine N(1)-acetyltransferase")
        self.assertEqual(newExpansion, "spermidine/spermine N(1)-acetyltransferase")   

        
        newExpansion = acroExp.select_expansion("BG", "sd Beta-1, 4-glucosidase")
        self.assertEqual(newExpansion, "Beta-1, 4-glucosidase")
        
        newExpansion = acroExp.select_expansion("myc-glu-glu", "mycosporine mycosporine-glutaminol-glucoside")
        self.assertEqual(newExpansion, "mycosporine-glutaminol-glucoside")

        newExpansion = acroExp.select_expansion("MPB", "dsda 3-N-maleimidyl-propionyl-biocytin")
        self.assertEqual(newExpansion, "3-N-maleimidyl-propionyl-biocytin")
        
        newExpansion = acroExp.select_expansion("T3S", "tsddd type III secretion")
        self.assertEqual(newExpansion, "type III secretion")
        
        #newExpansion = acroExp.select_expansion("Asf1", "anti-silencing function")
        #self.assertEqual(newExpansion, "anti-silencing function")

        # non acceptable cases
        newExpansion = acroExp.select_expansion("EXPANSION", "Test sentence expansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EX", "esddsdsdpão? ssddssdsd sdsdsdsd sdsdd xpansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EX", "Test sentence xexpansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EXP", "Test sentence epansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EXP3", "Test sentence expansion2")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("3EXP", "Test sentence 2expansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EXP", "e a x b c d pansion")
        self.assertEqual(newExpansion, None)
        
        
        # Suppose we found the acronym on the left of the expansion
        newExpansion = acroExp.select_expansion("EXP", "expannot Test sentence expansion", False)
        self.assertEqual(newExpansion, "expannot")
        
        newExpansion = acroExp.select_expansion("EXP", "Test sentence expansion", False)
        self.assertEqual(newExpansion, "expansion")
        
        newExpansion = acroExp.select_expansion("EXP", "Expansion bla bla some more text", False)
        self.assertEqual(newExpansion, "Expansion")

        newExpansion = acroExp.select_expansion("5-HIAA", "bla dssdd  5--hydroxy-indole-acetic acid", False)
        self.assertEqual(newExpansion, "5--hydroxy-indole-acetic acid")
        
        newExpansion = acroExp.select_expansion("nNOS", "aa nervours NOS sdsdd", False)
        self.assertEqual(newExpansion, "nervours NOS")
        
        newExpansion = acroExp.select_expansion("2AAF", "2-acetylaminofluorene", False)
        self.assertEqual(newExpansion, "2-acetylaminofluorene")
        
        newExpansion = acroExp.select_expansion("PDMAEMA", "poly[2-(dimethylamino) ethyl methacrylate] ethyl methacrylate]", False)
        self.assertEqual(newExpansion, "poly[2-(dimethylamino) ethyl methacrylate]")

        newExpansion = acroExp.select_expansion("2’5’-Oas", "2’5’-oligoadenyllate synthetase", False)
        self.assertEqual(newExpansion, "2’5’-oligoadenyllate synthetase")
        
    #    newExpansion = acroExp.select_expansion("ωτ", "wildtype")
    #    self.assertEqual(newExpansion, "wildtype")
        
        newExpansion = acroExp.select_expansion("GNAT", "Gcn5-related N-acetyltransferase acetyltransferase", False)
    #    self.assertEqual(newExpansion, "Gcn5-related N-acetyltransferase")    
        self.assertEqual(newExpansion, "Gcn5-related")       

    #    newExpansion = acroExp.select_expansion("[(18)F]FLT", "3-deoxy-3-(18)F-fluoro-l-thymidine")
    #    self.assertEqual(newExpansion, "3-deoxy-3-(18)F-fluoro-l-thymidine")   

        newExpansion = acroExp.select_expansion("SSAT", "spermidine/spermine N(1)-acetyltransferase atsdddsd sdds ", False)
        self.assertEqual(newExpansion, "spermidine/spermine N(1)-acetyltransferase")   

        
        newExpansion = acroExp.select_expansion("BG", "sb Beta-1, 4-glucosidase", False)
        self.assertEqual(newExpansion, "Beta-1, 4-glucosidase")
        
        newExpansion = acroExp.select_expansion("myc-glu-glu", "mycosporine-glutaminol-glucoside glucoside", False)
        self.assertEqual(newExpansion, "mycosporine-glutaminol-glucoside")

        newExpansion = acroExp.select_expansion("MPB", "dsda 3-N-maleimidyl-propionyl-biocytin", False)
        self.assertEqual(newExpansion, "3-N-maleimidyl-propionyl-biocytin")
        
        newExpansion = acroExp.select_expansion("T3S", "type bb III aa secretion secretion", False)
        self.assertEqual(newExpansion, "type bb III aa secretion")
        
        #newExpansion = acroExp.select_expansion("Asf1", "anti-silencing function")
        #self.assertEqual(newExpansion, "anti-silencing function")

        # non acceptable cases
        newExpansion = acroExp.select_expansion("EXPANSION", "Test sentence expansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EX", "esddsdsdpão? ssddssdsd sdsdsdsd sdsdd xpansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EX", "Test sentence xexpansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EXP", "Test sentence epansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EXP3", "Test sentence expansion2")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("3EXP", "Test sentence 2expansion")
        self.assertEqual(newExpansion, None)
        
        newExpansion = acroExp.select_expansion("EXP", "e a x b c d pansion")
        self.assertEqual(newExpansion, None)
        
    def test_extract_candidates(self):
        acroExp = AcroExpExtractor_Yet_Another_Improvement()
        
        self.assertEqual(len(list(acroExp.extract_candidates(self.someSentence))),0)
        
        acroExpanList = list(acroExp.extract_candidates(self.sentenceParenthesis1))
        self.assertEqual(len(acroExpanList), 1)
        self.assertEqual(acroExpanList[0], ("Test sentence EXP","Expansion xpto palalalalal", False))
        
        acroExpanList = list(acroExp.extract_candidates(self.sentenceParenthesis2))
        self.assertEqual(len(acroExpanList), 1)
        self.assertEqual(acroExpanList[0], ("EXP", "Test sentence expansion", True))
        
        acroExpanList = list(acroExp.extract_candidates(self.sentenceHyphen))
        self.assertEqual(len(acroExpanList), 1)
        self.assertEqual(acroExpanList[0], ("EXP", "Expansion bla bla some more text", False))
        
        acroExpanList = list(acroExp.extract_candidates(self.sentenceColon))
        self.assertEqual(len(acroExpanList), 1)
        self.assertEqual(acroExpanList[0], ("EXP", "Expansion bla bla some more text", False))


    def test_conditions(self):
        acroExp = AcroExpExtractor_Yet_Another_Improvement()
        
        for acronym in self.validAcronyms:
            self.assertTrue(acroExp.conditions(acronym))
            
        for acronym in self.invalidAcronyms:
            self.assertFalse(acroExp.conditions(acronym))
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()