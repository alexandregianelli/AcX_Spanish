import json
import os
import pickle

from helper import *

from process_tokens_and_bio_tags import (
    _match_capital_initials,
)

'''
        dict:
         a dictionary where each key is an acronym and each value is a list of lists with an expansion and article id
        dict:
         a dictionary where each key is an article id and each value is a dict where each key is an acronym and each value is an expansion
        dict:
         a dictionary where each key is an article id and each value is the raw text of the article
        dict:
         a dictionary where each key is an acronym
        list:
         a list with the article ids for training
        list:
         a list with the article ids for testing

'''

data_path = os.getcwd()+'/AcX_Spanish-main/AcX_Spanish-main/data/AcronymExpansion/'
print(data_path)
dev= json.load(open(data_path+'/dev.json', encoding='utf-8'))
train = json.load(open(data_path+'/train.json', encoding='utf-8'))
test = json.load(open(data_path+'/test.json', encoding='utf-8'))

raw_article_db = {}
acronym_db = {}
article_acronym_db = {}

train_articles = []
dev_articles = []

def coord_to_text(text, coords):
    result = list()
    for coord in coords:
        result.append(text[coord[0]:coord[1]].replace('(', '').replace(')', ''))
    return result

def match_acronyms(acronyms, long_forms):
    match_pairs = dict()
    for acronym in acronyms:
        for lf in long_forms:
            if _match_capital_initials(acronym, lf.split()) == True:
                match_pairs[acronym] = lf
    return match_pairs
               
def make_dbs(dataset_path, 
        acronym_db,
        article_acronym_db,
        article_db,
        train_articles,
        test_articles,
    ):
    generated_files_folder = dataset_path+'/generated_files/SpanishAcronymExpansion/'

    folds_num = 5

    pickle.dump(article_db, open(generated_files_folder+'raw_articledb.pickle', "wb"), protocol=2) # raw_article_db
    pickle.dump(acronym_db, open(generated_files_folder+'acronymdb.pickle', "wb"), protocol=2) # acronym_db
    pickle.dump(article_acronym_db, open(generated_files_folder+'article_acronym_db.pickle', "wb"), protocol=2,) #article_acronym_db

    pickle.dump(
        train_articles,
        open(generated_files_folder + "train_articles.pickle", "wb"),
        protocol=2,
    ) # train_articles
    pickle.dump(
        test_articles,
        open(generated_files_folder + "test_articles.pickle", "wb"),
        protocol=2,
    ) # test_articles

    folds_file_path = (
        generated_files_folder + str(folds_num) + "-cross-validation.pickle"
    )

    folds = getCrossValidationFolds(train_articles, folds_num)
    pickle.dump(folds, open(folds_file_path, "wb"), protocol=2) #cross_validation




for article in train:
    id = 'train_'+article["ID"]
    text = article["text"]
    acronyms = article["acronyms"]
    long_forms = article["long-forms"]

    acronyms = list(set(coord_to_text(text, acronyms)))
    long_forms = list(set(coord_to_text(text, long_forms)))
    in_pairs = match_acronyms(acronyms, long_forms)

    train_articles.append(id)
    raw_article_db[id] = text
    article_acronym_db[id] = in_pairs
    for acr in in_pairs.keys():
        acronym_db[acr] = [in_pairs[acr], id]

for article in dev:
    id = 'dev_'+article["ID"]
    text = article["text"]
    acronyms = article["acronyms"]
    long_forms = article["long-forms"]

    acronyms = list(set(coord_to_text(text, acronyms)))
    long_forms = list(set(coord_to_text(text, long_forms)))
    in_pairs = match_acronyms(acronyms, long_forms)

    dev_articles.append(id)
    raw_article_db[id] = text
    article_acronym_db[id] = in_pairs
    for acr in in_pairs.keys():
        acronym_db[acr] = [in_pairs[acr], id]

make_dbs(os.getcwd(), acronym_db, article_acronym_db, raw_article_db, train_articles, dev_articles)


