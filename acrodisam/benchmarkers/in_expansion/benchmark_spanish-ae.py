import pickle
from helper import create_configargparser, getDatasetGeneratedFilesPath
from Logger import logging
from benchmarkers.in_expansion.results_reporter import ResultsReporter
from benchmarkers.in_expansion.benchmark import Benchmarker

from string_constants import (
    SDU_AAAI_AI_DATASET,
    AB3P_DATASET,
    SH_DATASET,
    BIOADI_DATASET,
    MEDSTRACT_DATASET,
    USERS_WIKIPEDIA,
    SPANISH_ACRONYM_EXPANSION,
)

test_dataset_name = SPANISH_ACRONYM_EXPANSION

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = create_configargparser(
        in_expander=True,
        external_data=True,
        results_db_config=True,
    )
    args = parser.parse_args()

    if args.external_data:
        in_expander_train_dataset_name = [AB3P_DATASET, SH_DATASET, BIOADI_DATASET, MEDSTRACT_DATASET, SDU_AAAI_AI_DATASET, USERS_WIKIPEDIA, SPANISH_ACRONYM_EXPANSION]
    else:
        in_expander_train_dataset_name = SPANISH_ACRONYM_EXPANSION

    benchmarker = Benchmarker(
        in_expander_name=args.in_expander,
        in_expander_args=args.in_expander_args,
        in_expander_train_dataset_names=in_expander_train_dataset_name,
        in_expander_test_dataset_name=test_dataset_name,
        results_report=ResultsReporter,
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(test_dataset_name)

    test_articles = pickle.load(
        open(generatedFilesFolder + "test_articles.pickle", "rb")
    )

    # no actual need to specify the train articles
    # if we choose the same dataset for train and test
    # the system will just remove the articles that are in test_articles
    # from the train dataset
    train_articles = pickle.load(
        open(generatedFilesFolder + "train_articles.pickle", "rb")
    )

    indexedFolds = [(test_articles, train_articles)]

    benchmarker.run(
        indexedFolds,
        report_db_config=args.results_database_configuration,
        num_processes=len(indexedFolds),
    )
