"""Add the Acronym/Definition pairs from the orthogaffe website to the SqliteDict.

Created on jun 5, 2020

@author: mprieur
"""


from bs4 import BeautifulSoup

import requests

from utils_acronym_scrapping import get_acronym_db, show_acronym_db

LINK = "https://orthogaffe.wordpress.com/2012/12/26/sigles-courants/"


def add_db(acro, expansion):
    """ Add the (acronym,expansion) pair to the DataBase.
    Args :
        acro (str) :  the acronym
        expansion (str) : the expansion
    """
    new_dict_value = acro_db.setdefault(str(acro), [])
    if not str(expansion) in new_dict_value:
        new_dict_value.append(str(expansion))
    acro_db[acro] = new_dict_value

def process_elements(elems):
    """From a bs4.Tag process the (acronym,extansion) pairs.
    Args :
        elem (bs4.Tag) : the object containing the pairs.
    """
    acro_part = elems[0].contents[0]
    signification_part = elems[1].contents[0]
    if "/" in acro_part:
        acros = acro_part.split("/")
        significations = signification_part.split("/")
        for acro, sign in zip(acros, significations):
            add_db(acro, sign)
    elif "/" in signification_part:
        defs = signification_part.split("/")
        for sign in defs:
            add_db(acro_part, sign)
    else:
        add_db(acro_part, signification_part)

def parse_website():
    """Parse the website page to add the (new acronym,expansion) pairs.
    """
    response = requests.get(LINK).text
    table = BeautifulSoup(response, "html.parser").find('table')
    for line in table.findAll('tr'):
        elems = line.findAll('td')
        if len(elems) > 1:
            process_elements(elems)

if __name__ == "__main__":
    acro_db = get_acronym_db()
    parse_website()
    show_acronym_db()
