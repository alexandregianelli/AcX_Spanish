"""Allow to label documents from the French Wikipedia
to evaluate out-expansion extraction techniques.

Author : Maxime Prieur
Date : 07-15-2020
"""


from os import path
from sqlitedict import SqliteDict

from string_constants import FRENCH_WIKIPEDIA_DATASET, FOLDER_DATA
from helper import getDatasetGeneratedFilesPath


PATH_TO_TXT_FILE = FOLDER_DATA + "FrenchData/" + "labelled_articles.txt"
PATH_TO_DB = getDatasetGeneratedFilesPath(FRENCH_WIKIPEDIA_DATASET)+"labelled_acronyms.pickle"


def get_acronym_db():
    """Load the existing labelled articles DB.
    Returns :
        (SqliteDict) : the labelled_articles_db
    """
    return SqliteDict(PATH_TO_DB, autocommit=True, flag='r')


def display_label_per_articles():
    """Show the label par article in the DB
    """
    for key, value in get_acronym_db().items():
        print(key, value)


def add_labeled_article(doc_id, acro_exp_tuples):
    """ Add the labels for one doc to the db.
        Args :
            doc_id (int) : the article id corresponding to the document
                           in the parsed French_Wikipedia database.
            acro_exp_tuples (list): list of tuple (acronym, expansion), both str.
    """
    actual_value = LABELLED_ARTICLES_DB.setdefault(doc_id, [])
    if actual_value != acro_exp_tuples:
        actual_value = acro_exp_tuples
        LABELLED_ARTICLES_DB[doc_id] = actual_value


def update_labelled_db():
    """Parse the .txt with the label to update the groundthruth db.
    """
    doc_id = None
    labelled_file = open(PATH_TO_TXT_FILE, "r")
    for line in labelled_file:
        if "|" in line:
            acro, exp = line.split("|")
            # Expansions between [] are not present in text, otherise they are in-text expansions
            if exp[0] == "[":
                exp = exp[1:].replace(']', '')
                acro_exp_tuples[acro] = (exp.replace('\n', ''), False)
            else:
                acro_exp_tuples[acro] = (exp.replace('\n', ''), True)
        elif "doc_id :" in line:
            if not doc_id is None:
                add_labeled_article(doc_id, acro_exp_tuples)
            doc_id = ''.join([char for char in line if char.isdigit()])
            acro_exp_tuples = {}
    add_labeled_article(doc_id, acro_exp_tuples)

if __name__ == "__main__":
    if path.exists(PATH_TO_TXT_FILE):
        LABELLED_ARTICLES_DB = SqliteDict(PATH_TO_DB,
                                          flag='w',
                                          autocommit=True)
        update_labelled_db()
        display_label_per_articles()
    else:
        print('TXT PATH : "', PATH_TO_TXT_FILE, '" NOT VALID')
