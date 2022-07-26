import os
import functools
import math
from multiprocessing import Pool
from threading import Thread

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from DataCreators import ArticleDB
import logging
from TextTools import getCleanedWords
import text_preparation
import _pickle as pickle
from helper import LDAStructure, getArticleDBPath, getDatasetGeneratedFilesPath
from string_constants import file_lda_model_all, file_lda_word_corpus,\
    file_lda_bow_corpus, MSH_SOA_DATASET,\
    file_lda_model_all, file_lda_model,\
    file_lda_articleIDToLDA, file_lda_gensim_dictionary

logger = logging.getLogger(__name__)


class USESAVED:
    none = -1
    word_corpus = 0
    dictionary = 1
    bow_corpus = 2
    lda_model = 3


def load(path=file_lda_model_all):
    """
    Returns: SavedLDAModel object
    """

    logger.info("Loading LDA model from " + path)
    return pickle.load(open(path, "rb"))

# todo: put "_" in front of all private methods


def serialGetWordCorpus(articleDB):
    word_corpus = {}
    for article_id, text in articleDB.items():
        word_corpus[article_id] = text_preparation.getCleanedWords(
            text
            , stem_words=stem_words
            , removeNumbers=removeNumbers)
        if len(word_corpus) % 1000 == 0:
            logger.debug(
                "converted " + str(len(word_corpus)) + " articles to words")
    return word_corpus


def serialGetBoWCorpus(dictionary, word_corpus_values):
    return [dictionary.doc2bow(words) for words in word_corpus_values]

def preProcessText(text):
    # We assume that the input text was already preprocessed
    # this is just the minimum for LDA input
    return text_preparation.tokenizePreProcessedArticle(text.lower())

def parallelGetCleanedWords(article):
    #return article[0], getCleanedWords(article[1]
    #                                   , stem_words=stem_words
    #                                  , removeNumbers=removeNumbers)
    return article[0], preProcessText(article[1])

def parallelGetWordCorpus(articleDB, process_pool):
    articles = articleDB.items()
    results = process_pool.map(
        parallelGetCleanedWords, articles, chunksize=chunkSize_getCleanedWords)

    logger.info("Back from multiprocessing, making dict now")
    word_corpus = dict(results)

    return word_corpus


def _doc2bow_alias(dictionary, words):
    """
    Alias for instance method that allows the method to be called in a 
    multiprocessing pool
    see link for details: http://stackoverflow.com/a/29873604/681311
    """
    return dictionary.doc2bow(words)


def parallelGetBoWCorpus(dictionary, word_corpus_values, process_pool):
    bound_instance = functools.partial(_doc2bow_alias, dictionary)

    result = process_pool.map(
        bound_instance, word_corpus_values, chunksize=chunkSize_doc2BoW)

    return result


def getWordCorpus(articleDB, process_pool, useSavedTill):
    if(useSavedTill >= USESAVED.word_corpus):
        logger.info("Loading word_corpus from out_file")
        word_corpus = pickle.load(open(file_lda_word_corpus, "rb"))
        return word_corpus, None
    else:
        logger.info("Getting word_corpus from articles")
        word_corpus = parallelGetWordCorpus(
            articleDB, process_pool) if process_pool != None else serialGetWordCorpus(articleDB)

        return word_corpus, None
        """
        logger.info(
            "Saving word_corpus asynchronously, in case the script ahead fails")
        out_file = open(file_lda_word_corpus, "wb")
        word_corpus_dumper = Thread(
            target=pickle.dump, args=(word_corpus, out_file), kwargs={"protocol": 2})
        word_corpus_dumper.start()
        return word_corpus, word_corpus_dumper
        """


def getDictionary(word_corpus, useSavedTill):
    if(useSavedTill >= USESAVED.dictionary):
        logger.info("loading dictionary from file")
        dictionary = Dictionary.load(file_lda_gensim_dictionary)
        return dictionary
    else:
        logger.info("Creating dictionary from corpus")
        dictionary = Dictionary(word_corpus.values())
        return dictionary


def getBoWCorpus(word_corpus, dictionary, process_pool, useSavedTill):
    if(useSavedTill >= USESAVED.bow_corpus):
        logger.info("loading bow_corpus from out_file")
        bow_corpus = pickle.load(open(file_lda_bow_corpus, "rb"))
        return bow_corpus, None
    else:
        logger.info("Creating BoW representations from articles")
        bow_corpus = parallelGetBoWCorpus(dictionary, word_corpus.values(
        ), process_pool) if process_pool != None else serialGetBoWCorpus(dictionary, word_corpus.values())
        return bow_corpus, None


def getLdaModel(bow_corpus, dictionary, useSavedTill, num_topics=100, numPasses=1):
    if(useSavedTill >= USESAVED.lda_model):
        logger.info("loading LDA model from file")
        return LdaModel.load(file_lda_model)
    else:
        logger.info("Training LDA model")
        if(num_topics == 'log(nub_distinct_words)+1'):
            num_topics = int(math.log(len(bow_corpus)) + 1)
        else:
            num_topics = int(num_topics)
        
        lda_model = LdaModel(
            bow_corpus, num_topics=num_topics, id2word=dictionary, passes=numPasses)
        return lda_model

def createArticleIdToLdaDict(word_corpus, dictionary, lda_model):
    logger.info("Creating article_id -> lda_vector dictionary")
    article_lda = {}
    index = 0
    for article_id in word_corpus.keys():
        bow = dictionary.doc2bow(word_corpus[article_id])
        lda_vec = lda_model[bow]
        article_lda[article_id] = lda_vec
        index += 1
        if(index % 1000 == 0):
            logger.debug("done with %d articles", index)

    return article_lda


def waitForDumper(dumper, name):
    if(dumper != None):
        if(dumper.is_alive()):
            logger.info(
                "Waiting for" + name + " dumper to finish saving to disk")
            dumper.join()
        else:
            logger.info(
                name + " dumper has already finished saving to disk, not waiting")



def create_model(process_pool, articleDB, useSavedTill=USESAVED.none, num_topics=100, numPasses=1, 
                                datasetName = None,
                                fold = "",
                                saveAndLoad = False,
                                persistentArticles = None,
                                executionTimeObserver = None):
    """
    This takes a long time to train (~1 week), 
    run on a compute node with ~250 GB RAM and fast processor
    for wikipedia corpus of 410k documents

    Above time and storage estimates are not correct yet.
    """
    
    if saveAndLoad:
        generatedFilesFolder = getDatasetGeneratedFilesPath(datasetName)
        varToName = "_".join([str(s) for s in [datasetName,fold,num_topics,numPasses]])
        ldaLocation = generatedFilesFolder + file_lda_model_all + "_" + varToName + ".pickle"
            
        if os.path.isfile(ldaLocation):
            with open(ldaLocation, "rb") as f:
                return pickle.load(f)

    
    word_corpus, word_corpus_dumper = getWordCorpus(
        articleDB, process_pool, useSavedTill)
    if executionTimeObserver:
        executionTimeObserver.start()
    dictionary = getDictionary(word_corpus, useSavedTill)

    bow_corpus, bow_corpus_dumper = getBoWCorpus(
        word_corpus, dictionary, process_pool, useSavedTill)

    if(process_pool):
        logger.info("terminating process pool")
        process_pool.close()
        process_pool.terminate()

    ldaModel = getLdaModel(bow_corpus, dictionary, useSavedTill, num_topics=num_topics, numPasses=numPasses)

    articleIDToLDADict = createArticleIdToLdaDict(
        word_corpus, dictionary, ldaModel)
    
    model_all = LDAStructure(
        ldaModel, dictionary, articleIDToLDADict)
    if executionTimeObserver:
        executionTimeObserver.stop()
    
    if saveAndLoad:
        #logger.info("Saving LDA model object with all data")
        _saveAll(model_all, path=ldaLocation)

    
    return model_all


def _saveAll(model_all, path=file_lda_model_all):
    pickle.dump(model_all, open(path, "wb"), protocol=-1)


def update_model(articledb_path):
    """returns built lda_model, lda_dictionary"""
    pass  # todo: lda has update method, use it


def logConfig():
    logger.info("Logging config of script")
    logger.info("numProcesses = %d" % numProcesses)
    logger.info("goParallel = %s" % goParallel)
    logger.info("useSavedTill = %d" % useSavedTill)
    logger.info("chunkSize_getCleanedWords = %d" %
                       chunkSize_getCleanedWords)
    logger.info("chunkSize_doc2BoW = %d" % chunkSize_doc2BoW)
    logger.info("stem_words = %s" % stem_words)
    logger.info(
        "removeNumbers = %s" % removeNumbers)

def getArticleDB(datasetName):
    articleDBPath = getArticleDBPath(datasetName)    
    articleDB = ArticleDB.load(path=articleDBPath)
    return articleDB

# global config for making LDA model
numProcesses = 3
goParallel = True
useSavedTill = USESAVED.none
chunkSize_getCleanedWords = 1000
chunkSize_doc2BoW = 1000
stem_words = False
removeNumbers = True

if __name__ == "__main__":
    datasetName = MSH_SOA_DATASET
        
    articleDB = getArticleDB(datasetName)

    
    logger.info("LDA Model script started")
    logConfig()
    if(goParallel):
        process_pool = Pool(numProcesses)
        create_model(process_pool, articleDB, useSavedTill=useSavedTill, saveAndLoad=True)
    else:
        create_model(None, articleDB, useSavedTill=useSavedTill, saveAndLoad=True)
