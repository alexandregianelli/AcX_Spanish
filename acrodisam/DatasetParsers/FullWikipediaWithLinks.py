'''
Created on Apr 22, 2019

use wget 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
to get wikipedia

also get redirect.sql

python3.6 WikiExtractor.py -l  --filter_disambig_page  ~/Downloads/enwiki-latest-pages-articles.xml.bz2
also include abbr tag also


mysql -u wikipedia -p wikipedia < enwiki-20200301-redirect.sql



@author: jpereira
'''

import os
import re
import csv
import _pickle as pickle
import functools
from Logger import logging

from bs4 import BeautifulSoup
from multiprocessing import Pool
from tqdm import tqdm

from sqlitedict import SqliteDict

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer

import validators
import urllib.parse

import multiprocessing_logging

#from AcronymExtractors.AcronymExtractor_v4 import AcronymExtractor_v4
#from AcronymExpanders.Expander_fromText_v2 import Expander_fromText_v2
#from AcronymExpanders.Expander_fromText_v3 import Expander_fromText_v3
#from AcroExpExtractors.AcroExpExctractor_Generic import AcroExpExtractor_Generic
#from AcroExpExtractors.AcroExpExtractor_Schwartz_Hearst import AcroExpExtractor_Schwartz_Hearst
#from AcroExpExtractors.AcroExpExtractor_Yet_Another_Improvement import AcroExpExtractor_Yet_Another_Improvement
from AcroExpExtractors.AcroExpExtractor_Yet_Another_Improvement2 import AcroExpExtractor_Yet_Another_Improvement


from helper import get_acronym_db_path, getArticleDBPath, getArticleDBShuffledPath, getArticleAcronymDBPath,\
    getCrossValidationFolds, getTrainTestData,
from string_constants import FULL_WIKIPEDIA_DATASET, FOLDER_DATA
from DataCreators.ArticleAcronymDB import create_article_acronym_db_from_acronym_db
from text_preparation import get_expansion_without_spaces

logger = logging.getLogger(__name__)

stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
tokeniser = RegexpTokenizer(r'\w+')



#wikiExtractorOutputPath = "/data/acrodisam_app/data/FullWikipedia"
#wikiExtractorOutputPath = FOLDER_DATA+ "/FullWikipedia"
wikiExtractorOutputPath = "/home/jpereira/Software/wikiextractor-master/text"

storageType = "SQLite"



def _loadRedirectLinkDict():
    redirectDict = {}
    
    pathRedirect = wikiExtractorOutputPath + "/redirect_pages.csv"
    with open(pathRedirect, newline='\n', encoding='utf-8') as csvfile:
        redirectReader = csv.reader(csvfile, delimiter=';', quotechar='"')
        first = True
        for row in redirectReader:
            if first:
                first = False
                continue
            redirectDict[row[0].lower().replace("_"," ")] = row[1].lower().replace("_"," ")
    return redirectDict
            
            
def processTags(docTitle, doc, acronymConditions, selectExpansion, redirectDict, acroExpDict):
    inLinks = []
    extLinks = []    
    for c in doc.children:
        if c.name == "a":
            candidateAcronym = c.text
            if acronymConditions(candidateAcronym):
                link = c.attrs["href"]
                if link:
                    uncodedLink = urllib.parse.unquote(link)
                    valid=validators.url(uncodedLink)
                    if valid == True:
                        extLinks.append((docTitle, candidateAcronym, uncodedLink))
                        #print("Todo process external link: " + link)
                    else:
                        #wikipedia internal link
                        # Try to get expansion from link
                        candidateExpansion = uncodedLink.replace("_", " ")
                        expansion = selectExpansion(candidateAcronym, candidateExpansion)
                        if expansion:
                            acroExpDict[candidateAcronym] = expansion
                        else:
                            #check for redirect and save link somewhere
                            redirectLink = redirectDict.get(uncodedLink, uncodedLink)
                            inLinks.append((docTitle, candidateAcronym, redirectLink))
            else:
                link = c.attrs["href"]
                uncodedLink = urllib.parse.unquote(link)
                if link:
                    valid=validators.url(uncodedLink)
                    if valid == True:
                        print("here TODO")
        elif c.name == "abbv":
            print("Here TODO")
            
    return inLinks, extLinks
        
def _addExpToDB(acronymDB, docId, acroExpDict):
    for acronym, expansion in acroExpDict.items():
        expList = acronymDB.setdefault(acronym, [])
        expList.append((expansion, docId))
        

def preprocess_text(text, acroExpDict):
    expansionsWithoutSpaces = []
    for acronym, expansion in acroExpDict.items():
        if expansion is not None:
            expansionWithoutSpaces = get_expansion_without_spaces(expansion)
            expansionsWithoutSpaces.append(expansionWithoutSpaces)
        else:
            expansionWithoutSpaces = ""
        
        # Replace acronym
        text = re.sub("\\b" + re.escape(acronym) + "\\b", expansionWithoutSpaces, text, re.IGNORECASE)
        
        if expansion is not None:
            # Replace expansion
            text = re.sub(re.escape(expansion), expansionWithoutSpaces, text, re.IGNORECASE)

    
    #tokens = tokeniser.tokenize(text)
    #punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''


    text = re.sub('\W',' ', text)
    
    tokens = word_tokenize(text) 
    stopped_tokens = [t for t in tokens if not t.lower() in stop_words]

    #number_tokens = [x for x in stopped_tokens if x.isalpha]
    number_tokens = []
    for x in stopped_tokens:
        if x in expansionsWithoutSpaces:
            number_tokens.append(x)
        elif x.isalnum() and not x.isdigit():
            number_tokens.append(p_stemmer.stem(x))
            
    #stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]

    return ' '.join(number_tokens)

def processWikiDoc(docId, text, acroExp, acronymDB):
    
    acroExpDict = acroExp(text)
    
    if acroExpDict and len(acroExpDict) > 0:
        # TODO preprocessamento watever e substituir os acronimos e expansoes 
        preProcessedtext = preprocess_text(text, acroExpDict)

        _addExpToDB(acronymDB, docId, acroExpDict)

        logger.debug("Finished processing document "+docId + " found " + str(len(acroExpDict)) + " acronyms with expansion.")
        
        return preProcessedtext
        
    logger.debug("Finished processing document: "+docId + " no acronyms with expansion were found.")
    return None




def processWikiFile(filePath, acroExp, acronymConditions, selectExpansion, redirectDict, acronymDB=None, articleDB=None):
    if acronymDB is None:
        acronymDB = {}
        
    if articleDB is None:
        articleDB = {}
        
    logger.debug("Processing file: "+filePath)
    with open(filePath) as file:
            soupOut = BeautifulSoup(markup=file, features="lxml")
            for doc in soupOut.findAll(name="doc"):
                attributes = doc.attrs
                docId = attributes["id"]
                docUrl = attributes["url"]
                docTitle = attributes["title"].lower().replace("_"," ")
                
                text = doc.text
                
                #Check expansion in text
                acroExpDict = acroExp(text)
                    
                processTags(docTitle, doc, acronymConditions, selectExpansion, redirectDict, acroExpDict)
                
                if acroExpDict and len(acroExpDict) > 0:
                    _addExpToDB(acronymDB, docTitle, acroExpDict)
                
                articleDB[docTitle] = text
                
                # Text is returned only if acronyms and expansions are found
                #if preProcessedtext is not None:
                #    articleDB[docId] = preProcessedtext
                        
                        
    logger.debug("Finished processing file: "+filePath)
    return acronymDB, articleDB
                
def _mergeDicts(dictList):
    newDict = {}
 
    for d in dictList:
        for key, value in d.items():
            newDictValue = newDict.setdefault(key, [])
            newDictValue.extend(value)
    return newDict

def _extendAcronymDB(baseAcronymDB, resultAcronymDB):
    for key, value in resultAcronymDB.items():
        newDictValue = baseAcronymDB.setdefault(key, [])
        newDictValue.extend(value)
        #when using SQLite we have to make sure the value is assigned
        baseAcronymDB[key] = newDictValue
        

def multiProcessWikiFiles(filePathsList, acroExp, acronymConditions, selectExpansion, redirectDict, processes_number):
#    results = []
    with SqliteDict(getArticleDBPath(FULL_WIKIPEDIA_DATASET),
                     flag='n',
                     autocommit=True) as articleDB, SqliteDict(get_acronym_db_path(FULL_WIKIPEDIA_DATASET),
                     flag='n',
                     autocommit=True) as acronymDB:
    
        tasksNum = len(filePathsList)
        partialFunc = functools.partial(processWikiFile, acroExp=acroExp, acronymConditions=acronymConditions, selectExpansion=selectExpansion, redirectDict=redirectDict)
        with Pool(processes=processes_number) as process_pool:
            with tqdm(total=tasksNum) as pbar:
                #for i, r in tqdm(enumerate(process_pool.imap_unordered(processWikiFile, filePathsList, chunksize=1))):
                for i, r in tqdm(enumerate(process_pool.imap_unordered(partialFunc, filePathsList, chunksize=1))):
                    resultAcronymDB = r[0]
                    _extendAcronymDB(acronymDB, resultAcronymDB)
                    
                    resultArticleDB = r[1]
                    articleDB.update(resultArticleDB)
                    pbar.update()              
                
        #results = process_pool.starmap_async(_processArticles, fileInfoDict.items(), chunksize=1).get()
        #process_pool.join()
    
    #articlesDicts = [r[0] for r in results]
    #articleDB = _mergeDicts(articlesDicts)
#    articleDB = {r[0]:r[1] for r in results if r[0] is not None}
    
#    acronymDBDicts = [r[2] for r in results if r[2] is not None]
#    acronymDB = _mergeDicts(acronymDBDicts)
    
    return acronymDB, articleDB

def processWikiFolder(startdir, acroExp, acronymConditions, selectExpansion, processes_number = 1):
    filePathsList = []
    
    redirectDict = _loadRedirectLinkDict()
    
    directories = os.listdir(startdir)
    for wikiDir in directories:
        fullPathWikiDir = os.path.join(startdir, wikiDir)
        if os.path.isdir(fullPathWikiDir):
            for file in os.listdir(fullPathWikiDir):
                filePath = os.path.join(fullPathWikiDir, file)
                filePathsList.append(filePath)
    
    
    if processes_number > 1:
        return multiProcessWikiFiles(filePathsList, acroExp, acronymConditions, selectExpansion, redirectDict,  processes_number)
    else:
        acronymDB = {}
        articleDB = {}
        for filePath in filePathsList:
            processWikiFile(filePath, acroExp, acronymConditions, selectExpansion, redirectDict, acronymDB=acronymDB, articleDB=articleDB)
           
        return acronymDB, articleDB



def make_dbs(processes = 8):
    foldsNum = 5
    #acronymExtractor = AcronymExtractor_v4()
    #expansionExtractor = Expander_fromText_v2()
    #acroExpExtractor = AcroExpExtractor_Generic(acronymExtractor, expansionExtractor)
    
    #acroExpExtractor = AcroExpExtractor_Schwartz_Hearst()
    
    acroExpExtractor = AcroExpExtractor_Yet_Another_Improvement()
    acroExp = acroExpExtractor.get_acronym_expansion_pairs
    acronymConditions = acroExpExtractor.conditions
    selectExpansion = acroExpExtractor.getLinkExpansion
    
    acronymDB, articleDB = processWikiFolder(wikiExtractorOutputPath, acroExp, acronymConditions, selectExpansion, processes_number=processes)
    logger.debug("Finished processing Wikipedia")

    newArticleDB = {}
    for articleId, text in articleDB.items():
        acroExp = articleIDToAcronymExpansions[articleId]
        testExecutionTimeObserver = ExecutionTimeObserver()
        
        testExecutionTimeObserver.start()  
        preProcessedText = preprocess_text(text, acroExp)
        testExecutionTimeObserver.stop()
        
        newArticleDB[articleId] = preProcessedText
        accExecutionTimeObserver += testExecutionTimeObserver

    #removed acronymDB = applyManualCorrections(acronymDB)
    if storageType == "SQLite":
    
        with SqliteDict(get_acronym_db_path(FULL_WIKIPEDIA_DATASET),
                         flag='r',
                         autocommit=True) as acronymDB:
            #print("Expansions found:")
            #for acronym, expansion in acronymDB.items():
                #print("\t"+acronym+"\t"+' | '.join([exp[1]+":"+exp[0] for exp in expansion]))
            #    print("\t"+acronym+"\t"+' | '.join([exp[0] for exp in expansion]))

            articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
                acronymDB)
        logger.debug("Storing into SQLite")
#         with SqliteDict(getArticleDBPath(FULL_WIKIPEDIA_DATASET),
#                          flag='n',
#                          autocommit=True) as articles:
#             for article_id,  text in articleDB.items():
#                 articles[article_id] = text
#                 
#         with SqliteDict(get_acronym_db_path(FULL_WIKIPEDIA_DATASET),
#                          flag='n',
#                          autocommit=True) as acronymsExp:
#             for acronym,  expList in acronymDB.items():
#                 acronymsExp[acronym] = expList
                
        with SqliteDict(getArticleAcronymDBPath(FULL_WIKIPEDIA_DATASET),
                         flag='n',
                         autocommit=True) as articleAcroExpan:
            for article,  acroExp in articleIDToAcronymExpansions.items():
                articleAcroExpan[article] = acroExp
    else:
    #shuffledArticleDB = createShuffledArticleDB(articleDB)
        articleIDToAcronymExpansions = create_article_acronym_db_from_acronym_db(
                acronymDB)
        logger.debug("Storing into pickle")
        pickle.dump(articleDB, open(getArticleDBPath(FULL_WIKIPEDIA_DATASET), "wb"), protocol=2)
        pickle.dump(acronymDB, open(get_acronym_db_path(FULL_WIKIPEDIA_DATASET), "wb"), protocol=2)
        pickle.dump(articleIDToAcronymExpansions, open(
            getArticleAcronymDBPath(FULL_WIKIPEDIA_DATASET), "wb"), protocol=2)
    
    #pickle.dump(shuffledArticleDB, open(    
    #    getArticleDBShuffledPath(FULL_WIKIPEDIA_DATASET), "wb"), protocol=2)

    logger.debug("Generate Folds")
    generatedFilesFolder = getDatasetGeneratedFilesPath(FULL_WIKIPEDIA_DATASET)

    if storageType == "SQLite":
    
        with SqliteDict(getArticleDBPath(FULL_WIKIPEDIA_DATASET),
                     flag='r',
                     autocommit=True) as articleDB:
            articleDBKeys = set(articleDB.keys())
            
    else:
        articleDBKeys = articleDB.keys()

    newTrain, newTest = getTrainTestData(articleDBKeys, 0.70)
    pickle.dump(newTrain, open(generatedFilesFolder + 'train_articles.pickle', "wb"), protocol=2)
    pickle.dump(newTest, open(generatedFilesFolder + 'test_articles.pickle', "wb"), protocol=2)
    
    newFolds = getCrossValidationFolds(newTrain, foldsNum)

    foldsFilePath = generatedFilesFolder + str(foldsNum) + "-cross-validation.pickle"
    pickle.dump(newFolds, open(foldsFilePath, "wb"), protocol=2)

if __name__ == "__main__":
    #import cProfile
    #cProfile.run('make_dbs()')
    make_dbs(1)
    
    