"""
Created on Sep 20, 2020

@author: jpereira
"""

import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sqlitedict import SqliteDict
import pandas as pd
import networkx as nx
from nltk.metrics.distance import edit_distance

from helper import (
    get_acronym_db_path,
    getArticleDBPath,
    getArticleAcronymDBPath,
    extend_dict_of_lists,
    AcronymExpansion,
    getDatasetGeneratedFilesPath,
)
from string_constants import (
    FULL_WIKIPEDIA_DATASET,
    CS_WIKIPEDIA_DATASET,
    MSH_SOA_DATASET,
    SCIENCE_WISE_DATASET,
    SDU_AAAI_AD_DATASET,
)
from Logger import logging
from DataCreators import AcronymDB, ArticleDB
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from text_preparation import get_expansion_without_spaces

logger = logging.getLogger(__name__)

aaai_expansion_dict = pickle.load(open(
    getDatasetGeneratedFilesPath(SDU_AAAI_AD_DATASET + "_test")
    + "acronym_expansion_db.pickle", "rb")
)


def aggregate_criteria_expansion(expansions):
    value_counts = expansions.value_counts()
    max_value_counts = value_counts[value_counts == value_counts.max()]
    if len(max_value_counts) < 2:
        return max_value_counts.index[0]

    index_largest_value = max_value_counts.index.map(len).argmax()

    return max_value_counts.index[index_largest_value]

def expansions_similar_dist(expansion_1, expansion_2):
        expansion_1 = expansion_1.lower().replace(u"-", u" ")
        expansion_2 = expansion_2.lower().replace(u"-", u" ")
        # numActualWords = len(expansion_1)
        # numPredictedWords = len(expansion_2)

        if (
            expansion_1 == expansion_2
            or AcronymExpansion.startsSameWay(expansion_1, expansion_2)
        ):  # max(numActualWords, numPredictedWords)):
            return 0
        dist = edit_distance(expansion_1, expansion_2)
        return dist

def _get_approximate_duplicates(exp_article_pairs, aaai_acronym_exp):
    #expansions = data_frame["expansion"]
    app_dup = []
    for i in range(0, len(exp_article_pairs)):
        for j in range(0, len(aaai_acronym_exp)):
            min_dist = 3
            dist = expansions_similar_dist(exp_article_pairs[i][0], aaai_acronym_exp[j])
            if dist < min_dist:
                min_dist = dist
                min_id = j
            if min_dist < 3:
                app_dup.append((exp_article_pairs[i][0], exp_article_pairs[i][1], aaai_acronym_exp[min_id]))
            #else:
            #    app_dup.append((exp_article_pairs[i][0], exp_article_pairs[i][1], None))
    return app_dup


def _resolve_exp_for_acronym(item):
    acronym = item[0]
    exp_article_pairs = item[1]
    exp_changes_articles = {}
    # exp_article_pairs = acronymDB[acronym]
    aaai_acronym = acronym
    if len(aaai_acronym) > 1 and aaai_acronym[-1] == 's':
        aaai_acronym = aaai_acronym[:-1]
    aaai_acronym = aaai_acronym.upper()
    aaai_acronym_exp = aaai_expansion_dict.get(aaai_acronym)
    if aaai_acronym_exp is not None and len(aaai_acronym_exp) > 0:
        #df = pd.DataFrame(exp_article_pairs, columns=["expansion", "article"])

        matches = _get_approximate_duplicates(exp_article_pairs, aaai_acronym_exp)

        new_exp_article_pairs = [(m[2], m[1])  for m in matches]
        # keep track changes for articles text
        # changes_to_apply = merged_consolidated_all[new_exp_indx]

        # add to dict
        for m in matches:
            old_exp = m[0]
            new_exp = m[2]
            if old_exp != new_exp:
                article_id = m[1]
                article_changes = exp_changes_articles.setdefault(article_id, set())
                article_changes.add((old_exp, new_exp))

        return acronym, new_exp_article_pairs, exp_changes_articles
    # acronym_db_new[acronym] = exp_article_pairs
    return acronym, [], None


def _resolve_exp_acronym_db(acronymDB, acronym_db_new):

    exp_changes_articles = {}

    tasksNum = len(acronymDB)
    with ProcessPoolExecutor() as process_pool:
        with tqdm(total=tasksNum) as pbar:
            for _, r in tqdm(
                enumerate(
                    process_pool.map(
                        _resolve_exp_for_acronym, acronymDB.items(), chunksize=1
                    )
                )
            ):
                acronym = r[0]
                exp_article_pairs = r[1]
                exp_changes_articles_new = r[2]
                acronym_db_new[acronym] = exp_article_pairs

                if exp_changes_articles_new is not None:
                    extend_dict_of_lists(exp_changes_articles, exp_changes_articles_new)
                pbar.update()

    return exp_changes_articles


def replace_exp_in_article(article_id, text, exp_changes_articles):
    if exp_changes_articles is None:
        return text

    exp_changes = exp_changes_articles.pop(article_id, None)
    new_text = text

    if exp_changes:
        for change in exp_changes:
            old_exp_token = get_expansion_without_spaces(change[0])
            new_exp_token = get_expansion_without_spaces(change[1])
            (new_string, number_of_subs_made) = re.subn(
                "\\b" + old_exp_token + "\\b",
                new_exp_token,
                new_text,
                flags=re.IGNORECASE,
            )
            new_text = new_string
            if number_of_subs_made < 1:
                logger.error(
                    "When replacing expansions, it was unable to find expansion: "
                    + change[0]
                    + " in article: "
                    + article_id
                )
    return new_text


def _replace_exp_articles(old_articles_db, new_articles_db, exp_changes_articles):
    for article_id, text in old_articles_db.items():
        new_articles_db[article_id] = replace_exp_in_article(
            article_id, text, exp_changes_articles
        )

    if len(exp_changes_articles) > 0:
        logger.error(
            "Unable to apply %d expansion replacements to articles.",
            len(exp_changes_articles),
        )


def resolve_approximate_duplicate_expansions_aaai(dataset_name, sqlite=False):
    new_dataset_name = dataset_name + "_res-dup-aaai"
    if sqlite:
        old_acronym_db = AcronymDB.load(get_acronym_db_path(dataset_name), "SQLite")

        with SqliteDict(
            get_acronym_db_path(new_dataset_name), flag="n", autocommit=True
        ) as new_acronym_db:
            exp_changes_articles = _resolve_exp_acronym_db(
                old_acronym_db, new_acronym_db
                )
                
            old_acronym_db.close()
            with SqliteDict(
                getArticleAcronymDBPath(new_dataset_name), flag="n", autocommit=True
                ) as new_article_acronym_db:
                    create_article_acronym_db_from_acronym_db(new_acronym_db, article_acronym_db=new_article_acronym_db)
        
        
        old_articles_db = ArticleDB.load(getArticleDBPath(dataset_name), "SQLite")
        with SqliteDict(
            getArticleDBPath(new_dataset_name), flag="n", autocommit=True
        ) as new_articles_db:
            _replace_exp_articles(old_articles_db, new_articles_db, exp_changes_articles)
        old_articles_db.close()
    else:
        old_acronym_db = pickle.load(open(get_acronym_db_path(dataset_name), "rb"))
        new_acronym_db = {}

        exp_changes_articles = _resolve_exp_acronym_db(old_acronym_db, new_acronym_db)

        pickle.dump(
            new_acronym_db, open(get_acronym_db_path(new_dataset_name), "wb"), protocol=2
        )

        articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
            new_acronym_db
        )
        pickle.dump(
            articleIDToAcronymExpansions,
            open(getArticleAcronymDBPath(new_dataset_name), "wb"),
            protocol=2,
        )

        old_articles_db = pickle.load(open(getArticleDBPath(dataset_name), "rb"))
        new_articles_db = {}
        _replace_exp_articles(old_articles_db, new_articles_db, exp_changes_articles)

        pickle.dump(
            new_articles_db, open(getArticleDBPath(new_dataset_name), "wb"), protocol=2
        )

        logger.info("End")


if __name__ == "__main__":
    # dataset_name = CS_WIKIPEDIA_DATASET
    # resolve_approximate_duplicate_expansions(dataset_name)
    dataset_name = FULL_WIKIPEDIA_DATASET
    resolve_approximate_duplicate_expansions_aaai(dataset_name, True)
