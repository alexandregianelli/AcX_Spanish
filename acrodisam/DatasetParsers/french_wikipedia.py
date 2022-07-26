"""Parse the French Wikipedia database

Before executing this code please follow the next steps:
use wget 'https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2'
to download the latest french wikipedia dump file
Download and setup WikiExtractor
http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
Execute Wikipedia Extractor with the downloaded dump file as argument
E.g., python3.6 WikiExtractor.py  ~/Downloads/frwiki-latest-pages-articles.xml.bz2
Move the WikiExtractor output to:
{project root folder}/acrodisam_app/data/FrenchWikipedia/
Execute this script


Created on jun 2, 2020

@author: jpereira, mprieur
"""

import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sqlitedict import SqliteDict
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer

from AcroExpExtractors.AcroExpExtractor_FR import AcroExpExtractor_FR
from DatasetParsers import FullWikipedia
from helper import getDatasetGeneratedFilesPath
from Logger import logging
from string_constants import FRENCH_WIKIPEDIA_DATASET as FR_WIKI, FOLDER_DATA
from DatasetParsers.FullWikipedia import ParserFullWikipedia
from helper import getDatasetGeneratedFilesPath,\
    get_acronym_db_path, getArticleDBPath, getArticleDBShuffledPath, getArticleAcronymDBPath,\
    getCrossValidationFolds
from text_preparation import get_expansion_without_spaces

TEST_DB_PATH = getDatasetGeneratedFilesPath(FR_WIKI)+"labelled_acronyms.pickle"
TEST_IDS = [key for key in SqliteDict(TEST_DB_PATH, flag='r').keys()]
STORAGE_TYPE = "SQLite"

LOGGER = logging.getLogger(__name__)
stop_words = set(stopwords.words('french'))
p_stemmer = PorterStemmer()


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
    text = re.sub('\W',' ', text)

    tokens = word_tokenize(text) 
    stopped_tokens = [t for t in tokens if not t.lower() in stop_words]

    number_tokens = []
    for x in stopped_tokens:
        if x in expansionsWithoutSpaces:
            number_tokens.append(x)
        elif x.isalnum() and not x.isdigit():
            number_tokens.append(p_stemmer.stem(x))
        
    return ' '.join(number_tokens)

class ParserFrenchWikipedia(ParserFullWikipedia):

    def __init__(self):
        super().__init__()
        self.wikiExtractorOutputPath = FOLDER_DATA+ "FrenchData/FrenchWikipedia"
        self.dataset = FR_WIKI
        self.default_extractor = AcroExpExtractor_FR


    def process_wiki_doc(self, doc_id, text, acro_exp, acro_db):
        """Process the Wikipedia document.
        Args :
            doc_id (str) : the document id number
            text (str) :  the document
            acro_exp (function) :  return a dict with expansions as value.
            acro_db (List) :  List of pair(acronym, extansions)
        Returns :
            str : the preprocessed wikipedia document
        """
        try:
            acro_exp_dict = acro_exp.get_acronym_expansion_pairs(text)
        except:
            LOGGER.exception("Fatal error in acroExp.get_acronym_expansion_pairs for docId: "
                             +doc_id
                             +" text: "
                             + text)
            return None
        if acro_exp_dict and len(acro_exp_dict) > 0:
            pre_processed_text = preprocess_text(text, acro_exp_dict)
            self._addExpToDB(acro_db, doc_id, acro_exp_dict)
            LOGGER.debug("Finished processing document %s found %s acronyms with expansions."
                         , doc_id
                         , str(len(acro_exp_dict)))
            return pre_processed_text

        LOGGER.debug("Finished processing document: %s no acronyms with expansion were found.",
                     doc_id)
        return None

    def processWikiFile(self, file_path, acroExp, acro_db=None, article_db=None):
        """Process the Wikipedia documents.
        Args :
            file_path (str) : the document id number
            acroExp (function) :  return a dict with expansions as value.
            acro_db (dict) :  A dictionnary of acronym as key
            article_db (dict) :   dictionnary of processed texts
        Returns :
            acro_db : A dictionnary of acronym as key
            article_db (dict) : A dictionnary of processed texts
        """
        if acro_db is None:
            acro_db = {}
        if article_db is None:
            article_db = {}
        LOGGER.debug("Processing file: %s", file_path)
        file = open(file_path, 'r').read()
        docs = file.split('</doc>')
        for doc in docs[:-1]:
            doc = doc + '</doc>'
            soup = BeautifulSoup(markup=doc, features="lxml").find("doc")
            attributes = soup.attrs
            doc_id = attributes["id"]
            text = soup.text
            if doc_id not in TEST_IDS:
                pre_processed_text = self.process_wiki_doc(doc_id, text, acroExp, acro_db)
                # Text is returned only if acronyms and expansions are found
                if pre_processed_text is not None:
                    article_db[doc_id] = pre_processed_text
        LOGGER.debug("Finished processing file: %s", file_path)
        return acro_db, article_db


if __name__ == "__main__":
    parser = ParserFrenchWikipedia()
    parser.make_dbs()
