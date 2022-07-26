"""
Paper Acronym Disambiguation: A Domain Independent Approach
"""
import os
import re
import json

from AcronymExpanders import AcronymExpanderEnum
import logging
import _pickle as pickle
from helper import getDatasetPath, getDatasetGeneratedFilesPath,\
    get_acronym_db_path, getArticleDBPath, getArticleDBShuffledPath, getArticleAcronymDBPath,\
    getCrossValidationFolds, getTrainTestData
from string_constants import folder_data, file_cs_wikipedia_articleDB,\
    file_cs_wikipedia_acronymDB, file_cs_wikipedia_articleIDToAcronymExpansions,\
    file_cs_wikipedia_articleDB_shuffled, max_confidence, folder_cs_wikipedia_generated, folder_cs_wikipedia_corpus, CS_WIKIPEDIA_DATASET
import random
from collections import OrderedDict
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db

logger = logging.getLogger(__name__)


def find_full_form(abbr, para):
    re_string = "\\b"
    for c in abbr:
        re_string = re_string + c + "\w+[ _-]"
    re_string = re_string[:len(re_string) - 1 - 4]
    #print(re_string)
    return re.findall(re_string, para)


def _create_article_and_acronym_db():
    articleDB = {}
    acronymDB = {}
    trainArticles = []
    testArticles = []
    filename = folder_cs_wikipedia_generated+ "cs_wikipedia.json"
    f = open(filename, 'r')
    data = json.load(f)
    
    abbvFilename = folder_cs_wikipedia_corpus + "list_of_abbr.txt"
    abbvFile = open(abbvFilename, "r")
    abbvSet = set(r[:-1].upper() for r in abbvFile.readlines())
    abbvFile.close()
    
    n=0
    pmid = 1
    for acronym, v in data.items():
        #if n > 10:
        #    return acronymDB, articleDB, trainArticles, testArticles
        
        if acronym.upper() not in abbvSet:
            logger.info("No acronym: " + acronym + " found in abbv list")
            continue
        
        n+=1
        logger.info(str(n)+"th short form processed")
        logger.info("Processing acronym: " + str(acronym))

        first_full_form = v["full_form"]
        expansionWithoutSpaces = get_expansion_without_spaces(first_full_form)

        insensitive_acronym = re.compile(re.escape(acronym), re.IGNORECASE)
        rawText = str(v["content"])
        text = insensitive_acronym.sub(expansionWithoutSpaces, rawText)
        # Gets index - later used to take into account only the surrounding chars instead of the whole article

        if rawText == text:
            logger.warn("No acronym: " + acronym + " found in: \n" + str(v["content"]))
            continue
        
        acronymDB[acronym] = []
        articleDB[pmid] = text
        first_pmid = pmid
        pmid += 1
        for poss in v['possibilities']:
            if poss.lower() == rawText.lower():
                logger.warn("Duplicate article found! Acronym: " + acronym)
                continue
                
            full_forms = find_full_form(str(acronym).lower(),poss.lower())
            if len(full_forms) > 0:
            #    print("full_forms: " + " ".join(full_forms))
                
                for full_form in set(full_forms):
                    
                    full_form = full_form.strip()
                    expansionWithoutSpaces = get_expansion_without_spaces(full_form)

                    # Gets index - later used to take into account only the surrounding chars instead of the whole article
                    insensitive_full_form = re.compile("\\b" + re.escape(full_form) + "\\b", re.IGNORECASE)
                    text = insensitive_full_form.sub(expansionWithoutSpaces, poss)
   
                    acronymDB[acronym].append([full_form, pmid])
                    articleDB[pmid] = text
                    trainArticles.append(pmid)
                    pmid += 1
                
                
        # IF we find no alternative full_form then we have to discard this
        if len(acronymDB[acronym]) > 0:
            acronymDB[acronym].append([first_full_form, first_pmid])
            testArticles.append(first_pmid)
        else:
            logger.debug("Discarded acronym: " + str(acronym))
            del acronymDB[acronym]

    return acronymDB, articleDB, trainArticles, testArticles


def _createShuffledArticleDB(articleDB):
    items = list(articleDB.items())
    random.Random(1337).shuffle(items)
    shuffledArticleDB = OrderedDict(items)
    return shuffledArticleDB


def make_dbs():
    foldsNum = 5

    acronymDB, articleDB, trainArticles, testArticles = _create_article_and_acronym_db()

    #removed acronymDB = applyManualCorrections(acronymDB)

    articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
        acronymDB)

    shuffledArticleDB = _createShuffledArticleDB(articleDB)
    pickle.dump(articleDB, open(getArticleDBPath(CS_WIKIPEDIA_DATASET), "wb"), protocol=2)
    pickle.dump(acronymDB, open(get_acronym_db_path(CS_WIKIPEDIA_DATASET), "wb"), protocol=2)
    pickle.dump(articleIDToAcronymExpansions, open(
        getArticleAcronymDBPath(CS_WIKIPEDIA_DATASET), "wb"), protocol=2)
    pickle.dump(shuffledArticleDB, open(    
        getArticleDBShuffledPath(CS_WIKIPEDIA_DATASET), "wb"), protocol=2)

    generatedFilesFolder = getDatasetGeneratedFilesPath(CS_WIKIPEDIA_DATASET)


    pickle.dump(trainArticles, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
    pickle.dump(testArticles, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)
    
    newFolds = getCrossValidationFolds(trainArticles, foldsNum)

    foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"
    pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)

    # New train, test and folds
    newTrain, newTest = getTrainTestData(articleDB.keys(), 0.70)
    pickle.dump(newTrain, open(generatedFilesFolder + 'train_articles_new.pickle', "wb"), protocol=2)
    pickle.dump(newTest, open(generatedFilesFolder + 'test_articles_new.pickle', "wb"), protocol=2)
    
    newFolds = getCrossValidationFolds(newTrain, foldsNum)

    foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation_new.pickle"
    pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)
def _classToIndex(cls):
    return int(cls[1:]) - 1


def _fileNameToAcronym(fileName):
    return fileName.split("_")[0]

if __name__ == "__main__":
    make_dbs()
