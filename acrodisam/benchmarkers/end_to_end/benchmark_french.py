"""End to end Benchmark for the French Disambiguator.

@author : Maxime Prieur
"""

import sys
import multiprocessing_logging
from sqlitedict import SqliteDict
from Logger import logging

from AcronymExpander import AcronymExpander
from AcronymExpanders.Expander_Cossim import Factory_Expander_Cossim
from AcronymExpanders.Expander_Logistic_Regression import Factory_Expander_Logistic_Regression
from AcronymExpanders.Expander_Random_Forest import Factory_Expander_Random_Forest
from AcronymExpanders.Expander_SVM import Factory_Expander_SVM
from DatasetParsers import french_wikipedia, FullWikipedia
from DataCreators import ArticleDB, AcronymDB, ArticleAcronymDB
from TextExtractors.Extract_PdfMiner import Extract_PdfMiner
from TextRepresentators.Representator_Concat import Factory_ConcatRepresentators
from TextRepresentators.Representator_ContextVector import Factory_ContextVector
from TextRepresentators.Representator_Doc2Vec import Factory_Doc2Vec
from TextRepresentators.Representator_DocumentContextVector import Factory_DocumentContextVector
from TextRepresentators.Representator_LDA import Factory_LDA
from TextRepresentators.Representator_SBE import Factory_SBE
from TextRepresentators.Representator_TFIDF import Factory_TFIDF
from TextRepresentators.Representator_New_Locality import Factory_New_Locality
from benchmarkers.end_to_end.results_reporter import ResultsReporter
from benchmarkers.in_expansion.benchmark import Benchmarker
from helper import ExecutionTimeObserver, getArticleDBPath,\
     get_acronym_db_path, getArticleAcronymDBPath
from string_constants import FILE_REPORT_ENDTOEND_CSV, FRENCH_WIKIPEDIA_DATASET as FR_WIKI


DATASET_NAME = FR_WIKI
ARTICLE_DB_PATH = getArticleDBPath(DATASET_NAME)
ACRONYM_DB_PATH = get_acronym_db_path(DATASET_NAME)
ARTICLE_DB = ArticleDB.load(path=ARTICLE_DB_PATH, storageType=french_wikipedia.STORAGE_TYPE)
ACRONYM_DB = AcronymDB.load(path=ACRONYM_DB_PATH, storageType=french_wikipedia.STORAGE_TYPE)

LOGGER = logging.getLogger(__name__)

class EndToEndBenchmarker(Benchmarker):
    """Benchmarker for end-to-end benchmark.
    """
    def __init__(self, args, persistent_articles=None):
        LOGGER.info("Args: %s", ' '.join(args))
        super().__init__(args[:2])
        self.persistent_articles = 'SQLITE'
        self.text_extractor = Extract_PdfMiner()
        self.text_pre_process = FullWikipedia.preprocess_text
        self.expanders = [self._get_expander(args[2:])]
        self.labelled_article_db_path = self.article_db_path
        self.labelled_acronym_db_path = self.article_acronym_db_path
        self.article_db_path = getArticleDBPath(self.dataset_name)
        self.acronym_db_path = get_acronym_db_path(self.dataset_name)
        self.article_acronym_db_path = getArticleAcronymDBPath(self.dataset_name)
    def _get_expander(self, args):
        """Initialize the expander
            Args :
                args (list) : name of the text representator and expander
        """
        if len(args) > 1:
            if args[0] == 'Doc2Vec':
                if len(args) > 2:
                    doc_2_vec_args = args[2:6]
                else:
                    doc_2_vec_args = ['50', 'CBOW', '25', '8']
                representator_fact = Factory_Doc2Vec(doc_2_vec_args,
                                                     datasetName=DATASET_NAME,
                                                     saveAndLoad=True,
                                                     persistentArticles=True)
            elif args[0] == 'ContextVector':
                representator_fact = Factory_ContextVector()
            elif args[0] == 'NewLocality':
                representator_fact = Factory_New_Locality(['2', '2'],
                                                          datasetName=DATASET_NAME)
            elif args[0] == 'TFIDF':
                representator_fact = Factory_TFIDF(['0-0.50-5', '3-3'],
                                                   datasetName=DATASET_NAME,
                                                   saveAndLoad=True)
            elif args[0] == 'SBE':
                representator_fact = Factory_SBE([5, 1, 200, 5, 3])
            elif args[0] == 'TFIDF':
                representator_fact = Factory_TFIDF(['0-0.50-5', '3-3'],
                                                   datasetName=DATASET_NAME,
                                                   saveAndLoad=True)
            elif args[0] == 'Concat1':
                doc2vec_representator = Factory_Doc2Vec(['100', 'CBOW', '25', '2'],
                                                        datasetName=DATASET_NAME,
                                                        saveAndLoad=True,
                                                        persistentArticles=True)
                context_vector_representator = Factory_ContextVector()
                representators = [context_vector_representator, doc2vec_representator]
                representator_fact = Factory_ConcatRepresentators(representators)
            elif args[0] == 'Concat2':
                doc2vec_representator = Factory_Doc2Vec(['100', 'CBOW', '25', '2'],
                                                        datasetName=DATASET_NAME,
                                                        saveAndLoad=True,
                                                        persistentArticles=True)
                document_context_vector_representator = Factory_DocumentContextVector()
                representators = [document_context_vector_representator, doc2vec_representator]
                representator_fact = Factory_ConcatRepresentators(representators)
            elif args[0] == 'LDA':
                representator_fact = Factory_LDA(['100', 'log(nub_distinct_words)+1'],
                                                 datasetName=DATASET_NAME,
                                                 saveAndLoad=True,
                                                 persistentArticles=True)
            if args[1] == 'LR':
                if len(args) > 2:
                    lr_args = args[6:8]
                else:
                    lr_args = ['l2', '0.1']
                disam_exp_factory = Factory_Expander_Logistic_Regression(representator_fact,
                                                                         lr_args)
            elif args[1] == 'SVM':
                disam_exp_factory = Factory_Expander_SVM(representator_fact,
                                                         ['l2', '1'])
            elif args[1] == 'Cossim':
                disam_exp_factory = Factory_Expander_Cossim(representator_fact)
            elif args[1] == 'RF':
                disam_exp_factory = Factory_Expander_Random_Forest(representator_fact,
                                                                   ['100', "auto"])
        return disam_exp_factory.getExpander(ARTICLE_DB)


    def run(self, indexed_folds, report_file_name, num_processes=1):
        """Run the in-expansion benchmark.
            Args :
                indexed_folds (list) :
                report_file_name : Path of the results file
                num_processes : number of process to use in the benchmark
        """
        save_results_to_file = len(indexed_folds) < 2
        indexed_folds = indexed_folds[0]
        multiprocessing_logging.install_mp_handler(LOGGER)
        with ResultsReporter(datasetName=self.dataset_name,
                             argv=self.argv,
                             reportFileName=report_file_name,
                             saveResultsToFile=save_results_to_file) as results_reporter:
            self._proxy_function(indexed_folds, results_reporter)

    def _get_articles_and_acronyms(self):
        if self.persistent_articles is None and self.lang != 'FR':
            labelled_article_db = ArticleDB.load(path=self.article_db_path, storageType="SQLite")
            labelled_acronym_db = ArticleAcronymDB.load(path=self.article_acronym_db_path, storageType="SQLite")
        elif self.lang == 'FR':
            labelled_article_db = SqliteDict(self.labelled_article_db_path, flag='r')
            labelled_acronym_db = SqliteDict(self.labelled_acronym_db_path, flag='r')
            article_db = SqliteDict(self.article_db_path, flag='r')
            acronym_db = SqliteDict(self.acronym_db_path, flag='r')
            article_acronym_db = SqliteDict(self.article_acronym_db_path, flag='r')
        elif self.persistent_articles == "SQLITE":
            raise ValueError("Not implemented")
        else:
            raise ValueError("persitentArticles value unkown: " + str(self.persistent_articles))
        return labelled_article_db, labelled_acronym_db, article_db, acronym_db, article_acronym_db

    def _evaluate(self, fold_num, results_reporter):
        """ Process the benchmark and use the results_reporter
        to write the results.
            Args :
                fold_num (int):
                results_reporter (ResultReporter)
            Return :
                (bool)
        """
        labelled_article_db, labelled_acronym_db, article_db, acronym_db, article_acronym_db =\
             self._get_articles_and_acronyms()
        try:
            acro_disam = AcronymExpander(text_extractor=self.text_extractor,
                                         textPreProcess=self.text_pre_process,
                                         acroExpExtractor=self.acronym_extractor(),
                                         expanders=self.expanders,
                                         articleDB=article_db,
                                         acronymDB=acronym_db)
            LOGGER.critical("evaluating test performance")
            for article_id, article in labelled_article_db.items():
                LOGGER.debug("article_id: %s", article_id)
                try:
                    test_instance_execution_time = ExecutionTimeObserver()
                    test_instance_execution_time.start()
                    predicted_expansions = acro_disam.processText(article)
                    test_instance_execution_time.stop()
                    if self.lang == 'FR':
                        actual_expansions, predicted_expansions =\
                            self.clean_results(labelled_acronym_db.get(article_id, []),
                                               predicted_expansions, all_acronyms=True)
                    else:
                        actual_expansions = article_acronym_db.get(article_id, [])

                    results_reporter.addTestResult(fold_num,
                                                   article_id,
                                                   actual_expansions,
                                                   predicted_expansions,
                                                   test_instance_execution_time)
                except Exception as inst:
                    LOGGER.exception("skipping articleID: %s, error details:", article_id)
                    results_reporter.addTestError(fold_num, article_id, inst)
        finally:
            if self.persistent_articles == "SQLITE":
                acronym_db.close()
                article_db.close()
                labelled_article_db.close()
                labelled_acronym_db.close()
        return True


if __name__ == "__main__":
    # Args are :
    # - The Dataset Name / lang
    # - The AcroExp extractor name
    # - The text representator name
    # - The expander name
    # - Or onyl the 2 in one expander name
    if len(sys.argv) > 2:
        benchmarker = EndToEndBenchmarker(sys.argv[1:])
        benchmarker.run(["TrainData"], report_file_name=FILE_REPORT_ENDTOEND_CSV)
    else:
        print("NO VALID MODEL SELECTED")
