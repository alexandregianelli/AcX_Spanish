"""Bennchmark for in-expansion.
@author : jpereira
"""

from itertools import cycle
import sys
from concurrent.futures import ProcessPoolExecutor
from sqlitedict import SqliteDict
import multiprocessing_logging

from benchmarkers.in_expansion.results_reporter import ResultsReporter
from DataCreators import ArticleDB, ArticleAcronymDB
from Logger import logging

from helper import (
    getArticleDBPath,
    get_labelled_acronyms_path,
    getArticleAcronymDBPath,
    ExecutionTimeObserver,
    get_labelled_articles_path,
)

from string_constants import (
    FILE_REPORT_EXTRACTION_CSV,
    FRENCH_WIKIPEDIA_DATASET as FR_WIKI,
    FR_MAPPING,
)

from AcroExpExtractors.AcroExpExtractor_Schwartz_Hearst import (
    AcroExpExtractor_Schwartz_Hearst,
)
from AcroExpExtractors.AcroExpExtractor_Yet_Another_Improvement2 import (
    AcroExpExtractor_Yet_Another_Improvement,
)
from AcroExpExtractors.AcroExpExtractor_Original_Schwartz_Hearst import (
    AcroExpExtractor_Original_Schwartz_Hearst,
)
from AcroExpExtractors.AcroExpExtractor_FR import AcroExpExtractor_FR

LOGGER = logging.getLogger(__name__)


class Benchmarker:
    """Benchmarker of the in-expansion AcroExpExtractor"""

    def __init__(self, args, persistent_articles=None):
        LOGGER.info("Args: " + " ".join(args))
        self.lang = args[0]
        if self.lang == "FR":
            self.dataset_name = FR_WIKI
            self.article_acronym_db_path = get_labelled_acronyms_path(self.dataset_name)
            self.article_db_path = get_labelled_articles_path(self.dataset_name)
        else:
            self.dataset_name = args[0]
            self.article_db_path = getArticleDBPath(self.dataset_name)
            self.article_acronym_db_path = getArticleAcronymDBPath(self.dataset_name)
        self.argv = args[2:]
        self.persistent_articles = persistent_articles
        self.acronym_extractor = self._get_acro_exp_extractor(args[1], self.argv[1:])

    def _get_acro_exp_extractor(self, extractor_name, args=None):
        """Load the AcroExpExtractor
        Args :
            extractor_name (str) : Name of the AcroExpExtractor
        Returns :
            (AcroExpExtractor)
        """
        if extractor_name == "Schwartz_Hearst":
            return AcroExpExtractor_Schwartz_Hearst
        elif extractor_name == "Orig_SH":
            return AcroExpExtractor_Original_Schwartz_Hearst
        elif extractor_name == "Ours":
            return AcroExpExtractor_Yet_Another_Improvement
        elif extractor_name == "FR":
            return AcroExpExtractor_FR
        else:
            raise Exception("No extractor found with name: " + extractor_name)

    def _proxy_function(self, fold, results_reporter):
        logger = logging.getLogger(__name__ + "._proxyfunction")
        try:
            self._evaluate(fold, results_reporter)
        except BaseException as exception:
            logger.critical("Fatal error in subprocess", exc_info=True)
            logger.critical(exception.args)
            raise exception

    def _get_articles_and_acronyms(self):
        if self.persistent_articles is None and self.lang != "FR":
            article_db = ArticleDB.load(path=self.article_db_path, storageType="SQLite")
            article_acronym_db = ArticleAcronymDB.load(
                path=self.article_acronym_db_path, storageType="SQLite"
            )
        elif self.lang == "FR":
            article_db = SqliteDict(self.article_db_path, flag="r")
            article_acronym_db = SqliteDict(self.article_acronym_db_path, flag="r")
        elif self.persistent_articles == "SQLITE":
            raise ValueError("Not implemented")
        else:
            raise ValueError(
                "persitentArticles value unkown: " + str(self.persistent_articles)
            )

        return article_db, article_acronym_db

    def _evaluate(self, fold_num, results_reporter):
        """Process the benchmark and use the results_reporter
        to write the results.
            Args :
                fold_num (int):
                results_reporter (ResultReporter)
            Return :
                (bool)
        """
        extractor = self.acronym_extractor()
        article_db, article_acronym_db = self._get_articles_and_acronyms()

        LOGGER.critical("evaluating test performance")
        for article_id, article in article_db.items():
            LOGGER.debug("article_id: %s" % article_id)
            try:
                test_instance_execution_time = ExecutionTimeObserver()
                test_instance_execution_time.start()
                predicted_expansions = extractor.get_acronym_expansion_pairs(article)
                test_instance_execution_time.stop()
                if self.lang == "FR":
                    actual_expansions, predicted_expansions = self.clean_results(
                        article_acronym_db.get(article_id, []), predicted_expansions
                    )
                else:
                    actual_expansions = article_acronym_db.get(article_id, [])

                results_reporter.addTestResult(
                    fold_num,
                    article_id,
                    actual_expansions,
                    predicted_expansions,
                    test_instance_execution_time,
                )

            except Exception as inst:
                LOGGER.exception(
                    "skipping articleID: %s, error details:" % (article_id)
                )
                results_reporter.addTestError(fold_num, article_id, inst)
        return True

    def run(self, indexed_folds, report_file_name, num_processes=1):
        """Run the in-expansion benchmark.
        Args :
            indexed_folds (list) :
            report_file_name : Path of the results file
            num_processes : number of process to use in the benchmark
        """
        save_results_to_file = len(indexed_folds) < 2
        multiprocessing_logging.install_mp_handler(LOGGER)
        with ResultsReporter(
            datasetName=self.dataset_name,
            argv=self.argv,
            reportFileName=report_file_name,
            saveResultsToFile=save_results_to_file,
        ) as results_reporter:
            # with Pool(processes=num_processes, maxtasksperchild=1) as process_pool:
            with ProcessPoolExecutor(num_processes) as process_pool:
                process_pool.map(
                    self._proxy_function,
                    zip(indexed_folds, cycle([results_reporter])),
                    chunksize=1,
                )

    def clean_results(self, labelled_acronyms, predicted_acronyms, all_acronyms=False):
        """Clean the acronyms and expansion to compare them.
        Args :
            labelled_acronyms (dict) : the labelled acronym/expansions
            predicted_acronyms (dict) : the predicted pairs of acronym/expansion
        Returns :
            actual_expansions (dict) :  the cleaned labelled pairs
            cleaned_predicted_acronyms (dict) : the cleaned predicted pairs
        """
        actual_expansions = {}
        cleaned_predicted_acronyms = {}
        for key in labelled_acronyms:
            if not all_acronyms and labelled_acronyms[key][1]:
                actual_expansions[key] = labelled_acronyms[key][0]
            elif all_acronyms:
                actual_expansions[key] = labelled_acronyms[key][0:2]
        actual_expansions = self.french_clean_labelled_acronym(
            actual_expansions, all_acronyms
        )
        for key in predicted_acronyms:
            if not predicted_acronyms[key] is None:
                cleaned_predicted_acronyms[
                    key.upper().replace(".", "")
                ] = self.french_clean_exp(predicted_acronyms[key])
        return actual_expansions, cleaned_predicted_acronyms

    def french_clean_exp(self, exp):
        """Formant the input expansion for comparaison.
        Args :
            exp (str) : the acronym's expansion
        """
        new_exp = ""
        for letter in exp:
            new_exp += FR_MAPPING.get(letter, letter)
        parts = new_exp.lower().replace("-", " ").split(" ")
        for i, part in enumerate(parts):
            if len(part) > 0 and part[-1] == "s":
                parts[i] = part[:-1]
        return " ".join(parts)

    def french_clean_labelled_acronym(self, tuples, all_acronyms):
        """Remove duplicate pairs (acronym, expansion) and format them.
        Args :
            tuples (list) : lsit of pairs (acronym, expansion, exp_is_in_text)
        Returns :
            cleaned_tuples (list) : formated tuples
            real_case_acro (dict) : for a formated acronym, give is original form
        """
        cleaned_tuples = {}
        for acro in tuples:
            cleaned_acro = acro.upper().replace(".", "")
            if not all_acronyms:
                cleaned_exp = self.french_clean_exp(tuples[acro])
            else:
                cleaned_exp = self.french_clean_exp(tuples[acro][0])
            if (
                cleaned_exp != ""
                and cleaned_exp is not None
                and cleaned_acro not in cleaned_tuples
            ):
                if all_acronyms:
                    cleaned_tuples[cleaned_acro] = [cleaned_exp, tuples[acro][1]]
                else:
                    cleaned_tuples[cleaned_acro] = cleaned_exp
        return cleaned_tuples


if __name__ == "__main__":
    argv = sys.argv[1:]
    LOGGER.info("Args: " + " ".join(argv))
    # The args required are :
    # - The Dataset Name / lang
    # - The extractor Name
    benchmarker = Benchmarker(argv)
    REPORT_FILE_NAME = FILE_REPORT_EXTRACTION_CSV
    benchmarker.run(["TrainData"], REPORT_FILE_NAME, num_processes=1)
