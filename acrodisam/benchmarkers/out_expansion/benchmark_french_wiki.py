"""Benchmark of the out-expansion techniques on the French Wikipedia"""


import logging
import _pickle as pickle
import sys

from Logger import logging

from benchmarkers.out_expansion.benchmark import Benchmarker
from string_constants import  FRENCH_WIKIPEDIA_DATASET, file_report_csv, FOLDER_GENERATED_FILES
from helper import get_acronym_db_path, getArticleDBPath, getArticleAcronymDBPath,\
                   getDatasetGeneratedFilesPath
from DataCreators import ArticleAcronymDB, ArticleDB, AcronymDB
from TextExtractors.Extract_PdfMiner import Extract_PdfMiner
from AcronymExtractors.AcronymExtractor import AcronymExtractor
LOGGER = logging.getLogger(__name__)

class BenchmarkerFrenchWiki(Benchmarker):
    """The French Wikipedia Benchmarker class grouping the required functions.
    """
    doc2vecArgs = ['100', 'CBOW', '25', '2']
    def __init__(self, args, persistent_articles=None):
        self.datasetName = FRENCH_WIKIPEDIA_DATASET
        self.acronymDBPath = get_acronym_db_path(FRENCH_WIKIPEDIA_DATASET)
        self.acronymExtractor = AcronymExtractorFrenchWiki(self)
        self.argv = args
        self.articleAcronymDBPath = getArticleAcronymDBPath(FRENCH_WIKIPEDIA_DATASET)
        self.articleAcronymDb = ArticleAcronymDB.load(path=self.articleAcronymDBPath, storageType="SQLite")
        self.articleDBPath = getArticleDBPath(FRENCH_WIKIPEDIA_DATASET)
        self.articleIDToAcronymExpansions = self.articleAcronymDb
        self.persistentArticles = persistent_articles
        self.expandersToUse = [self._getAcronymExpander(FRENCH_WIKIPEDIA_DATASET, args)]
        self.textExtractor = Extract_PdfMiner()


    def _loadArticles(self,
                      train_articles_ids,
                      test_articles_ids,
                      train_articles={},
                      test_articles={}):
        LOGGER.info("Staring loading articles")
        article_db = ArticleDB.load(path=self.articleDBPath, storageType="SQLite")
        for key, value in article_db.items():
            if key in test_articles_ids:
                test_articles[key] = value
            elif key in train_articles_ids:
                train_articless[key] = value
            else:
                pass
        LOGGER.info("Finished putting articles into DBs")
        return train_articles, test_articles


    def _getRealignedAcronymDb(self, article_ids_to_keep):
        acronym_db = AcronymDB.load(path=self.acronymDBPath, storageType="SQLite")
        for acronym in acronym_db.keys():
            valid_entries = []
            for entry in acronym_db[acronym]:
                if entry[1] in article_ids_to_keep:
                    valid_entries.append(entry)
            acronym_db[acronym] = valid_entries
        return acronym_db


class AcronymExtractorFrenchWiki(AcronymExtractor):
    """A custom created for benchmark acronym extractor.
    """
    def __init__(self, benchmark_french_wiki):
        self.benchmark_french_wiki = benchmark_french_wiki


    def get_acronyms(self, text, article_id):
        acronyms_expansions = \
            self.benchmark_french_wiki.article_id_to_acronym_expansions[article_id] \
                                .keys()
        result = {}
        for acronym in acronyms_expansions:
            result[acronym] = []
        return result


if __name__ == "__main__":

    if len(sys.argv) > 1:
        ARGV = sys.argv[1:]
    else:
        ARGV = []

    GENERATED_FILES_FOLDER = getDatasetGeneratedFilesPath(FRENCH_WIKIPEDIA_DATASET)
    TRAIN_ARTICLES = pickle.load(open(GENERATED_FILES_FOLDER
                                      + 'train_articles.pickle',
                                      "rb"))
    TEST_ARTICLES = pickle.load(open(GENERATED_FILES_FOLDER + 'test_articles.pickle',
                                     "rb"))
    INDEXED_FOLDS = [("TrainData", TEST_ARTICLES, TRAIN_ARTICLES)]
    REPORT_FILE_NAME = str(file_report_csv).replace('.csv', 'french.csv')
    LOGGER.info("Args: %s", ' '.join(ARGV))
    BENCHMARKER = BenchmarkerFrenchWiki(ARGV)
    BENCHMARKER.run(INDEXED_FOLDS, REPORT_FILE_NAME)
