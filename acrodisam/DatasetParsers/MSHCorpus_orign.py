"""
convert <e> to ( or ""
convert </e> to ) or ""
regex extend: BLM (Bloom's syndrome protein)
"""
from csv import DictReader
import os
import re

import arff

from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.Expander_fromText_v2 import Expander_fromText_v2
import logging
import text_preparation
import pickle
from helper import AcronymExpansion
from string_constants import folder_msh_arff, file_msh_articleDB,\
    file_msh_acronymDB, file_msh_articleIDToAcronymExpansions,\
    file_msh_manual_corrections, file_msh_articleDB_shuffled, max_confidence
import random
from collections import OrderedDict
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db

logger = logging.getLogger(__name__)


def _removeMarkup(text):
    textWithoutMarkup = re.sub(u"\\<e\\>", u"", text)
    textWithoutMarkup = re.sub(u"\\<\\/e\\>", u"", textWithoutMarkup)
    return textWithoutMarkup


def _createArticleIDToAcronymExpansions(acronymDB):
    articleIDToAcronymExpansions = {}
    for acronym in acronymDB:
        for expansion, articleID, def_count in acronymDB[acronym]:
            if articleID not in articleIDToAcronymExpansions:
                articleIDToAcronymExpansions[articleID] = {}
            acronymExpansion = AcronymExpansion(
                expansion, AcronymExpanderEnum.none, confidence=max_confidence)
            articleIDToAcronymExpansions[articleID][
                acronym] = [acronymExpansion]

    return articleIDToAcronymExpansions


def _create_article_and_acronym_db():
    acronymExpander = Expander_fromText_v2()
    articleDB = {}
    acronymDB = {}
    CUID_to_expansion = {}
    for fileName in os.listdir(folder_msh_arff):
        filePath = os.path.join(folder_msh_arff, fileName)
        file_reader = arff.load(open(filePath, "rt"))
        # the iterator needs to be called for the self.relation part to be
        # initialized
        lines = list(file_reader['data'])
        cuids = file_reader['relation'].strip().split("_")
        # storing all acronyms as uppercase values
        acronym = _fileNameToAcronym(fileName).upper()
        cuid_and_pmid = []
        for line in lines:
            pmid = str(line[0])
            text = line[1]
            cuid = cuids[_classToIndex(line[2])]
            textWithoutMarkup = _removeMarkup(text)
            if (cuid not in CUID_to_expansion):
                acronymExpansions = []
                acronymExpansions = acronymExpander._expandInText(
                    textWithoutMarkup, acronym)
                if (acronymExpansions is not None and len(acronymExpansions) != 0 and
                        acronymExpansions[0] != acronym):
                    CUID_to_expansion[cuid] = acronymExpansions[0]
            if (pmid not in articleDB):
                articleDB[pmid] = textWithoutMarkup
            cuid_and_pmid.append([cuid, pmid])

        if (acronym in acronymDB):
            logger.error("acronym already present in acronymDB")
        else:
            acronymDB[acronym] = []
        for cuid, pmid in cuid_and_pmid:
            if (cuid in CUID_to_expansion):
                acronymDB[acronym].append([CUID_to_expansion[cuid], pmid, 0])
            else:
                logger.warn(
                    "Expansion not found for CUID %s of %s" % (cuid, acronym))
                acronymDB[acronym].append([cuid, pmid, 0])

    return acronymDB, articleDB


def _createShuffledArticleDB(articleDB):
    items = list(articleDB.items())
    random.Random(1337).shuffle(items)
    shuffledArticleDB = OrderedDict(items)
    return shuffledArticleDB


def make_dbs():
    acronymDB, articleDB = _create_article_and_acronym_db()

    #removed acronymDB = applyManualCorrections(acronymDB)

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
        acronymDB)

    shuffledArticleDB = _createShuffledArticleDB(articleDB)

    pickle.dump(articleDB, open(file_msh_articleDB, "wb"), protocol=2)
    pickle.dump(acronymDB, open(file_msh_acronymDB, "wb"), protocol=2)
    pickle.dump(articleIDToAcronymExpansions, open(
        file_msh_articleIDToAcronymExpansions, "wb"), protocol=2)
    pickle.dump(shuffledArticleDB, open(
        file_msh_articleDB_shuffled, "wb"), protocol=2)


def applyManualCorrections(acronymDB):
    for line in DictReader(open(file_msh_manual_corrections, "rb"), delimiter=","):
        acronym = text_preparation.toUnicode(line["acronym"])
        wrong_exp = text_preparation.toUnicode(line["wrong_expansion"])
        correct_exp = text_preparation.toUnicode(line["correct_expansion"])

        for entry in acronymDB[acronym]:
            if entry[0] == wrong_exp:
                entry[0] = correct_exp

    return acronymDB


def _classToIndex(cls):
    return int(cls[1:]) - 1


def _fileNameToAcronym(fileName):
    return fileName.split("_")[0]

if __name__ == "__main__":
    make_dbs()
