"""Add the Acronym/Definition pairs from the orthogaffe website to the Btb.

Created on jun 5, 2020

@author: mprieur
"""

import re

from bs4 import BeautifulSoup

import requests

from utils_acronym_scrapping import get_acronym_db, show_acronym_db


LINK = "https://www.btb.termiumplus.gc.ca/redac-chap?lang=fra&lettr=chapsect1&info0=1.4"


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

def getpair(acro_part, exp_part):
    """ Precess the pair in the given parts.
    Args :
        acro_part (str) : the acronyms parts
        exp_part (str) : the expansions parts
    """
    acro_part = re.split(', ', acro_part)
    for acro in acro_part:
        if len(acro) > 1 and acro.count(".") != 1 and not "/" in acro:
            exp = exp_part.replace("*", "")
            if acro[0] == " ":
                acro = acro[1:]
            if exp[0] == " ":
                exp = exp[1:]
            exp = exp.replace('\xa0', '')
            add_db(acro, exp)

def process_elements(elements):
    """From a bs4.Tag process the (acronym,extansion) pairs.
    Args :
        elements (bs4.Tag) : the object containing the pairs.
    """
    for elem in elements:
        if elem.contents[0] == ' ':
            acro_part = elem.contents[1].get_text()
            exp_part = elem.contents[3].get_text()
            if ";" in acro_part:
                acro_part = re.split(';', acro_part)
                exp_part = re.split(';', exp_part)
                for i in range(len(exp_part)):
                    getpair(acro_part[i], exp_part[i])
            elif ";" in exp_part:
                exp_part = re.split(';', exp_part)
                for i in range(len(exp_part)):
                    getpair(acro_part, exp_part[i])
            else:
                getpair(acro_part, exp_part)

def parse_website():
    """Parse the website page to add the (new acronym,expansion) pairs.
    """
    response = requests.get(LINK).text
    elements = BeautifulSoup(response, "html.parser").findAll('tr', class_='alignTop')
    process_elements(elements)

if __name__ == "__main__":
    acro_db = get_acronym_db()
    parse_website()
    show_acronym_db()
