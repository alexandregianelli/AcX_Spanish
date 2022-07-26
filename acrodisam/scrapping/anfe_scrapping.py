"""Add the Acronym/Definition pairs from the anfe website to the SqliteDict.

Created on jun 7, 2020

@author: mprieur
"""


import re

from bs4 import BeautifulSoup

import requests

from utils_acronym_scrapping import get_acronym_db


LINK = "https://www.anfe.fr/liste-des-sigles"


def add_db(acro, expansion):
    """ Add the (acronym,expansion) pair to the DataBase.
    Args :
        acro (str) :  the acronym
        expansion (str) : the expansion
    """
    if expansion[0] == " ":
        expansion = expansion[1:]
    new_dict_value = acro_db.setdefault(str(acro), [])
    if not str(expansion) in new_dict_value:
        new_dict_value.append(str(expansion))
    acro_db[acro] = new_dict_value

def upper_word(expansion):
    """Return an upper word if  there is one in the expansion.
    Args :
        expansion (str) : the expansion
    Returns:
        Empty word or the upper word if found
    """
    words = expansion.split(' ')
    for word in words:
        if word.isupper() and len(word) > 1:
            return word
    return ""

def process_elements(pairs):
    """From a List containing the acronym / expansions pair non formatted
    process the (acronym,extansion) pairs.
    Args :
        pairs (list) : the lists containing the acronym and expansions
    """
    separation = pairs.split(":")
    acronym = separation[0].replace(u'\xa0', u'').replace(u"(^)", "")
    acronym = acronym[:-1] if acronym[-1] == " " else acronym
    subs = re.search("\((.*?)\)", acronym)
    if not subs is None:
        acronym = acronym.replace("("+subs.group(1)+")", "")
    for expansion in separation[1:]:
        upper = upper_word(expansion)
        if upper != "":
            expansion = expansion.replace(upper, "")
            add_db(acronym, expansion)
            acronym = upper
        else:
            expansion = expansion.replace(acronym, "")
            add_db(acronym, expansion)

def parse_website():
    """Parse the website page to add the (new acronym,expansion) pairs.
    """
    request = requests.get(LINK).text
    body = BeautifulSoup(request, "html.parser").find("div",
                                                      {"itemprop":"articleBody"}).findAll("p")
    for paragraph in body:
        definitions = paragraph.text.split("\n")
        for pairs in definitions:
            if ":" in pairs:
                process_elements(pairs)

if __name__ == "__main__":
    acro_db = get_acronym_db()
    parse_website()
