"""
Used for the Cross-Training and Additional Data experiments
Runs experiments for Reuters with the option to add wikipedia to the training data
"""
import pickle

from DataCreators import ArticleDB
from Logger import logging
from acronym_expander import RunConfig
from benchmarkers.out_expansion.benchmark import Benchmarker
from helper import (
    getArticleDBPath,
    getDatasetGeneratedFilesPath,
    create_configargparser,
)
from string_constants import FULL_WIKIPEDIA_DATASET, REUTERS_DATASET


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = create_configargparser(
        crossvalidation=True, out_expander=True, save_and_load=True
    )
    parser.add_argument(
        "--use-wikipedia",
        "-wiki",
        action="store_true",
        help="additionally uses wikipedia as training data",
    )
    args = parser.parse_args()

    experiment_name = REUTERS_DATASET  #pylint: disable=invalid-name

    if args.use_wikipedia:
        experiment_name += "_wiki"
        train_dataset_name = [REUTERS_DATASET, FULL_WIKIPEDIA_DATASET]
    else:
        train_dataset_name = REUTERS_DATASET  #pylint: disable=invalid-name

    run_config = RunConfig(
        name=experiment_name,
        save_and_load=args.save_and_load,
        persistent_articles="SQLITE",
    )

    benchmarker = Benchmarker(
        run_config=run_config,
        train_dataset_name=train_dataset_name,
        test_dataset_name=REUTERS_DATASET,
        out_expander_name=args.out_expander,
        out_expander_args=args.out_expander_args,
        expansion_linkage=True,
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(REUTERS_DATASET)
    if args.crossvalidation:
        FOLDS_NUM = 5
        foldsFilePath = (
            generatedFilesFolder + str(FOLDS_NUM) + "-cross-validation.pickle"
        )
        folds = pickle.load(open(foldsFilePath, "rb"))
        # Adds index to folds list
        indexedFolds = [(fold[0], fold[1], idx) for idx, fold in enumerate(folds)]
    else:
        reuters_train_articles = pickle.load(
            open(generatedFilesFolder + "train_articles.pickle", "rb")
        )
        reuters_test_articles = pickle.load(
            open(generatedFilesFolder + "test_articles.pickle", "rb")
        )
        if args.use_wikipedia:
            wiki_train_articles = list(
                ArticleDB.load(path=getArticleDBPath(FULL_WIKIPEDIA_DATASET)).keys()
            )
            indexedFolds = [
                (reuters_test_articles, [reuters_train_articles, wiki_train_articles])
            ]
        else:
            indexedFolds = [(reuters_test_articles, reuters_train_articles)]

    benchmarker.run(indexedFolds, num_processes=len(indexedFolds))
