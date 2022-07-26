import json
import logging
import pickle
import re

from nltk.tokenize import word_tokenize

from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from DataCreators.ArticleDB import get_preprocessed_article_db
from helper import (
    getDatasetPath,
    getDatasetGeneratedFilesPath,
    get_acronym_db_path,
    getArticleAcronymDBPath,
    getCrossValidationFolds,
    get_raw_article_db_path,
    get_preprocessed_article_db_path,
)
from string_constants import UAD_DATASET
from text_preparation import (
    get_expansion_without_spaces,
    stop_words,
)


logger = logging.getLogger(__name__)


def _process_data_file(file_path, article_db, acronym_db, articles_id_start):
    logger.info("Processing file %s", file_path)

    sample_list = [
        json.loads(line)
        for line in open(file_path, "U")
        if line != "" and not line.isspace()
    ]

    test_articles_ids = []
    article_id = articles_id_start
    for sample in sample_list:

        sentence = " ".join(sample["sentence"])

        acronym = sample["short-form"]
        expansion = sample["long-form"]
        expansion_without_spaces = get_expansion_without_spaces(expansion)

        sentence_with_expansion = sentence.replace("__ABB__", expansion_without_spaces)

        if acronym not in acronym_db:
            acronym_db[acronym] = []
        acronym_db[acronym].append([expansion, article_id])

        article_db[article_id] = sentence_with_expansion
        test_articles_ids.append(article_id)
        article_id += 1
    return test_articles_ids


def _create_article_and_acronym_db(dataset_name):
    dataset_path = getDatasetPath(dataset_name)

    article_db = {}
    acronym_db = {}

    test_raw_file_path = (
        dataset_path + "manually_verified_dataset/manually_labeled.json"
    )
    test_articles = _process_data_file(
        test_raw_file_path,
        article_db=article_db,
        acronym_db=acronym_db,
        articles_id_start=0,
    )

    train_articles = []
    # we skip fold 0 because it has the same text as the manually labeled dataset
    articles_id_start = test_articles[-1] + 1
    for fold in range(1, 10):
        train_raw_file_path = f"{dataset_path}wikipedia/test_{fold}.txt"
        fold_articles = _process_data_file(
            train_raw_file_path,
            article_db=article_db,
            acronym_db=acronym_db,
            articles_id_start=articles_id_start,
        )
        train_articles.extend(fold_articles)
        articles_id_start = fold_articles[-1] + 1
    return acronym_db, article_db, train_articles, test_articles


def _pre_processor(text, expansions_without_spaces=None):
    tokens = word_tokenize(text)

    final_tokens = []
    for token in tokens:
        if expansions_without_spaces and token in expansions_without_spaces:
            final_tokens.append(token)
            continue
        lower_token = token.lower()
        if lower_token not in stop_words:
            final_tokens.append(lower_token)

    return " ".join(final_tokens)


def make_dbs():
    foldsNum = 5
    dataset_name = UAD_DATASET

    logger.info("Creating DBs")
    acronymDB, articleDB, train_ids, test_ids = _create_article_and_acronym_db(
        dataset_name
    )

    logger.info("Creating Article-Acronym DB")
    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(acronymDB)

    logger.info("Storing DBs")
    pickle.dump(
        articleDB, open(get_raw_article_db_path(dataset_name), "wb"), protocol=2
    )
    pickle.dump(acronymDB, open(get_acronym_db_path(dataset_name), "wb"), protocol=2)
    pickle.dump(
        articleIDToAcronymExpansions,
        open(getArticleAcronymDBPath(dataset_name), "wb"),
        protocol=2,
    )

    logger.info("Saving train and test set ids")
    generatedFilesFolder = getDatasetGeneratedFilesPath(dataset_name)

    pickle.dump(
        train_ids,
        open(generatedFilesFolder + "train_articles.pickle", "wb"),
        protocol=2,
    )
    pickle.dump(
        test_ids, open(generatedFilesFolder + "test_articles.pickle", "wb"), protocol=2
    )

    logger.info("Creating and saving Preprocessed Articles DB")

    (
        preprocessed_artible_db,
        train_exec_time,
        test_avg_exec_time,
    ) = get_preprocessed_article_db(
        articleDB,
        articleIDToAcronymExpansions,
        train_ids,
        test_ids,
        _pre_processor,
    )
    pickle.dump(
        preprocessed_artible_db,
        open(get_preprocessed_article_db_path(dataset_name), "wb"),
        protocol=2,
    )

    logger.info("Creating and saving cross-validation set ids")
    folds_file_path = generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"

    folds = getCrossValidationFolds(train_ids, foldsNum)
    pickle.dump(folds, open(folds_file_path, "wb"), protocol=2)


if __name__ == "__main__":
    make_dbs()
