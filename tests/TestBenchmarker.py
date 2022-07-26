import unittest

from acrodisam.Benchmarker.Benchmark import Benchmarker


class TestBenchmarker(unittest.TestCase):

    bm = Benchmarker()

    def test_areExpansionsSimilar_same(self):
        str1 = u"premature ventricular contractions"
        str2 = u"premature ventricular contractions"
        self.assertTrue(TestBenchmarker.bm._Benchmarker__areExpansionsSimilar(str1, str2))

    def test_areExpansionsSimilar_underscored(self):
        str1 = u"diamond-blackfan anemia"
        str2 = u"diamond blackfan anemia"
        self.assertTrue(TestBenchmarker.bm._Benchmarker__areExpansionsSimilar(str1, str2))

    def test_areExpansionsSimilar_joinedWordsWithUnderscore(self):
        str1 = u"anti-lymphocyte serum"
        str2 = u"antilymphocyte serum"
        self.assertTrue(TestBenchmarker.bm._Benchmarker__areExpansionsSimilar(str1, str2))

    def test_areExpansionsSimilar_spellingDifference(self):
        str1 = u"primary ciliary dyskinesia"
        str2 = u"primary ciliary diskinesia"
        self.assertTrue(TestBenchmarker.bm._Benchmarker__areExpansionsSimilar(str1, str2))

    def test_areExpansionsSimilar_differentExpansions(self):
        self.assertFalse(TestBenchmarker.bm._Benchmarker__areExpansionsSimilar(
            u"osteochondritis dissecans", u"osteochondral defects"))
