import json
import logging
import pickle

from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from DataCreators.ArticleDB import get_preprocessed_article_db
from helper import (
    getDatasetPath,
    getDatasetGeneratedFilesPath,
    get_acronym_db_path,
    getArticleAcronymDBPath,
    getCrossValidationFolds,
    getTrainTestData,
    get_preprocessed_article_db_path,
    get_raw_article_db_path,
)
from string_constants import SPANISH_ACRONYM_DESAMBIGUATION
from text_preparation import full_text_preprocessing
from text_preparation import get_expansion_without_spaces


logger = logging.getLogger(__name__)


def _create_test_db(dataset_name):
    article_db = {}
    acronym_db = {}
    article_acronym_db = {}

    
    create_acronym_expansions_db = True
    acronym_expansion_db = {}
    

    dataset_path = getDatasetPath(dataset_name)
    test_data = json.load(open(dataset_path + "/test.json"))
    if create_acronym_expansions_db:
        diction_data = json.load(open(dataset_path + "/diction.json"))

    for test_sample in test_data:
        pmid = test_sample["ID"]
        text = test_sample["sentence"]
        acronym = test_sample["acronym"]
        # we do this so the expansion is considered a single token, after a tokenization

        if acronym not in acronym_db:
            acronym_db[acronym] = []
        acronym_db[acronym].append(pmid)

        article_db[pmid] = text

        article_acronym_db[pmid] = {}
        article_acronym_db[pmid][acronym] = get_expansion_without_spaces(acronym)

        if create_acronym_expansions_db:
            acronym_expansion_db[acronym] = diction_data[acronym]
        
    return article_db, acronym_db, acronym_expansion_db, article_acronym_db


def make_test_dbs(dataset_name):

    article_db, acronym_db, acronym_expansion_db, article_acronym_db = _create_test_db(
        dataset_name
    )

    dataset_test_name = dataset_name + "_test"
    pickle.dump(
        article_db, open(get_raw_article_db_path(dataset_test_name), "wb"), protocol=2
    )
    pickle.dump(
        acronym_db, open(get_acronym_db_path(dataset_test_name), "wb"), protocol=2
    )
    pickle.dump(
        article_acronym_db,
        open(getArticleAcronymDBPath(dataset_test_name), "wb"),
        protocol=2,
    )
    if acronym_expansion_db is not None:
        pickle.dump(
            acronym_expansion_db,
            open(
                getDatasetGeneratedFilesPath(dataset_test_name)
                + "acronym_expansion_db.pickle",
                "wb",
            ),
            protocol=2,
        )

    (preprocessed_artible_db, _, test_avg_exec_time,) = get_preprocessed_article_db(
        article_db,
        article_acronym_db,
        [],
        article_db.keys(),
        full_text_preprocessing,
    )

    pickle.dump(
        preprocessed_artible_db,
        open(get_preprocessed_article_db_path(dataset_test_name), "wb"),
        protocol=2,
    )


def _create_article_and_acronym_db(dataset_name):
    dataset_path = getDatasetPath(dataset_name)

    raw_article_db = {}
    acronym_db = {}
    train_articles = []
    dev_articles = []

    train_data = json.load(open(dataset_path + "/train.json"))
    dev_data = json.load(open(dataset_path + "/dev.json"))

    for train_sample in train_data:
        pmid = "TR-"+train_sample["ID"]
        if pmid=="TR-88":
            print("here")
        
        text = train_sample["sentence"]
        acronym = train_sample["acronym"]
        expansion = train_sample["label"]
        # we do this so the expansion is considered a single token, after a tokenization
        exp_without_spaces = get_expansion_without_spaces(expansion)
        text_with_exp = text
        text_with_exp = text_with_exp.replace(acronym, " "+exp_without_spaces+" ")
        
        if acronym not in acronym_db:
            acronym_db[acronym] = []
        acronym_db[acronym].append([expansion, pmid])
        
        raw_article_db[pmid] = text_with_exp
        train_articles.append(pmid)

    for dev_sample in dev_data:
        pmid = "DV-"+dev_sample["ID"]
        text = dev_sample["sentence"]
        acronym = dev_sample["acronym"]
        expansion = dev_sample["label"]
        # we do this so the expansion is considered a single token, after a tokenization
        exp_without_spaces = get_expansion_without_spaces(expansion)
        text_with_exp = text
        text_with_exp = text_with_exp.replace(acronym, " "+exp_without_spaces+" ")
        
        if acronym not in acronym_db:
            acronym_db[acronym] = []
        acronym_db[acronym].append([expansion, pmid])

        raw_article_db[pmid] = text_with_exp
        dev_articles.append(pmid)
        
    
    return acronym_db, raw_article_db, train_articles, dev_articles


def make_dbs(dataset_name):

    folds_num = 5
    (
        acronym_db,
        article_db,
        train_articles,
        test_articles,
    ) = _create_article_and_acronym_db(dataset_name)

    article_acronym_db = create_article_acronym_db_from_acronym_db(acronym_db)
    

    pickle.dump(
        article_db, open(get_raw_article_db_path(dataset_name), "wb"), protocol=2
    )
    pickle.dump(acronym_db, open(get_acronym_db_path(dataset_name), "wb"), protocol=2)
    pickle.dump(
        article_acronym_db,
        open(getArticleAcronymDBPath(dataset_name), "wb"),
        protocol=2,
    )

    generated_files_folder = getDatasetGeneratedFilesPath(dataset_name)

    pickle.dump(
        train_articles,
        open(generated_files_folder + "train_articles.pickle", "wb"),
        protocol=2,
    )
    pickle.dump(
        test_articles,
        open(generated_files_folder + "test_articles.pickle", "wb"),
        protocol=2,
    )

    folds_file_path = (
        generated_files_folder + str(folds_num) + "-cross-validation.pickle"
    )

    folds = getCrossValidationFolds(train_articles, folds_num)
    pickle.dump(folds, open(folds_file_path, "wb"), protocol=2)

    # New train, test and folds
    new_train, new_test = getTrainTestData(article_db.keys(), 0.70)
    pickle.dump(
        new_train,
        open(generated_files_folder + "train_articles_new.pickle", "wb"),
        protocol=2,
    )
    pickle.dump(
        new_test,
        open(generated_files_folder + "test_articles_new.pickle", "wb"),
        protocol=2,
    )

    new_folds = getCrossValidationFolds(new_train, folds_num)

    folds_file_path = (
        generated_files_folder + str(folds_num) + "-cross-validation_new.pickle"
    )
    pickle.dump(new_folds, open(folds_file_path, "wb"), protocol=2)

    (
        preprocessed_artible_db,
        train_exec_time,
        test_avg_exec_time,
    ) = get_preprocessed_article_db(
        article_db,
        article_acronym_db,
        train_articles,
        test_articles,
        full_text_preprocessing,
    )

    pickle.dump(
        preprocessed_artible_db,
        open(get_preprocessed_article_db_path(dataset_name), "wb"),
        protocol=2,
    )

    make_test_dbs(dataset_name)


if __name__ == "__main__":
    # Uncomment to parse the original SDU AAAI dataset
    # make_dbs(SDU_AAAI_AD_DATASET)
    make_dbs(SPANISH_ACRONYM_DESAMBIGUATION)
