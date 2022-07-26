"""Add the Acronym/Definition pairs from the file "151_Fichier.pdf" to the
SqliteDict.

Created on jun 5, 2020

@author: mprieur
"""

import pdftotext

from utils_acronym_scrapping import get_acronym_db
from string_constants import FOLDER_DATA

PATH = FOLDER_DATA+"/FrenchData/Abreviations/151_Fichier.pdf"


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


def all_caps_first_part(line):
    """Get the acronym part of the first part of the line.
    Args:
        line (str): the line to process
    Returns:
        acronym (str): the acronym
        index (int): the index separating acro/exp
    """
    index = line.index(" ")
    acronym = line[0:index]
    if " " in line[index+1:]:
        line = line.replace(acronym+" ", "")
        second_index = line.index(" ")
        second_part = line[:second_index]
        if second_part.isupper():
            return acronym+" "+second_part, index+second_index+1
    return acronym, index


def process_line(line):
    """From a line of the pdf file, extract the (acronym,extansion) pairs.
    Args :
        line (str) : the string containing the text to process
    """
    while len(line) > 0 and line[0] == " ":
        line = line[1:]
    if len(line) > 1 and line[1].isupper() and " " in line:
        acronym, index = all_caps_first_part(line)
        #index = line.index(" ")
        #acronym = line[0:index]
        expansion = line[index:]
        while len(expansion) > 0 and expansion[0] == " ":
            expansion = expansion[1:]
        if len(expansion) > 1:
            if expansion[-1] == ".":
                expansion = expansion[:-1]
            add_db(acronym, expansion)


def parse_file():
    """Parse the pdf file to add the (new acronym,expansion) pairs.
    """
    with open(PATH, "rb") as pdf_file:
        pdf = pdftotext.PDF(pdf_file)
    lines = pdf[2].split("\n")
    for line in lines[1:]:
        process_line(line)
    for page in range(3, len(pdf)):
        lines = pdf[page].split("\n")
        for line in lines:
            process_line(line)


if __name__ == "__main__":
    acro_db = get_acronym_db()
    parse_file()
    #show_acronym_db()
