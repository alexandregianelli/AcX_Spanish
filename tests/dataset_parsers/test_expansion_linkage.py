"""
Created on Mar 25, 2021

@author: jpereira
"""
import unittest
from DatasetParsers.expansion_linkage import _resolve_exp_for_acronym


class ExpansionLinkageTest(unittest.TestCase):
    def test_resolve_exp_for_acronym_for_article_duplicates(self):
        acronym = "ATE"
        exp_article_pairs = [
            ("Acronym Test Expansion", "0"),
            ("Acronym Test Expansions", "0"),
            ("A very diff exp", "1"),
            ("Acronym Test Expansion", "2"),
        ]
        item = (acronym, exp_article_pairs)
        (
            acronym_out,
            new_exp_article_pairs,
            exp_changes_articles,
        ) = _resolve_exp_for_acronym(item)
        self.assertEqual(acronym, acronym_out)

        expected_new_exp_article_pairs = set([
            ("Acronym Test Expansion", "0"),
            ("A very diff exp", "1"),
            ("Acronym Test Expansion", "2"),
            ]
        )
        self.assertEqual(expected_new_exp_article_pairs, set(new_exp_article_pairs))

        expected_exp_changes_articles = {
            "0": set([("Acronym Test Expansions", "Acronym Test Expansion")])
        }
        self.assertEqual(expected_exp_changes_articles, exp_changes_articles)


        exp_article_pairs = [
            ("bla bla", "0"),
            ("Acronym Test Expansion", "0"),
            ("A very diff exp", "1"),
            ("Acronym Test Expansion", "2"),
        ]
        item = (acronym, exp_article_pairs)
        (
            acronym_out,
            new_exp_article_pairs,
            exp_changes_articles,
        ) = _resolve_exp_for_acronym(item)
        
        self.assertEqual(acronym, acronym_out)

        expected_new_exp_article_pairs = set([
            ("Acronym Test Expansion", "0"),
            ("A very diff exp", "1"),
            ("Acronym Test Expansion", "2"),
            ]
        )
        self.assertEqual(expected_new_exp_article_pairs, set(new_exp_article_pairs))

        expected_exp_changes_articles = {
            "0": set([("bla bla", "Acronym Test Expansion")])}
        
        self.assertEqual(expected_exp_changes_articles, exp_changes_articles)
        
        
        
        exp_article_pairs = [
            ("Acronym Test Expansion", "0"),
            ("bla bla", "0"),
            ("A very diff exp", "1"),
        ]
        item = (acronym, exp_article_pairs)
        (
            acronym_out,
            new_exp_article_pairs,
            exp_changes_articles,
        ) = _resolve_exp_for_acronym(item)
        
        self.assertEqual(acronym, acronym_out)

        expected_new_exp_article_pairs = set([
            ("Acronym Test Expansion", "0"),
            ("A very diff exp", "1"),
            ]
        )
        self.assertEqual(expected_new_exp_article_pairs, set(new_exp_article_pairs))

        expected_exp_changes_articles = {
            "0": set([("bla bla", "Acronym Test Expansion")])}
        
        self.assertEqual(expected_exp_changes_articles, exp_changes_articles)
#if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
#    unittest.main()
