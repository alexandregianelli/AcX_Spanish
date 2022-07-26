"""Functions and constant used in the different acronym scrapper.

Created on jun 7, 2020

@author: mprieur
"""


from sqlitedict import SqliteDict

from string_constants import FOLDER_GENERATED_FILES

ACRONYM_DB_PATH = FOLDER_GENERATED_FILES+"French/french_acronyms.pickle"

def format_acronym_db():
    """ Format the acronym db
    """
    acro_db = get_acronym_db()
    for key, _ in acro_db.items():
        _ = acro_db.setdefault(key, [])

def get_acronym_db():
    """Load the existing French Acronym DB.
    Returns :
        (SqliteDict) : the French Acronym DB
    """
    return SqliteDict(ACRONYM_DB_PATH, autocommit=True)

def show_acronym_db():
    """Print  the content of the French Acronym DB.
    """
    for key, value in get_acronym_db().items():
        print(key, value)

if __name__ == "__main__":
    show_acronym_db()
