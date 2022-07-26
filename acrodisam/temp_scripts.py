"""
**************************************************************************************************
Figure out the reason for "expansion not predicted"

"""
"""
from __future__ import division
import cPickle
from string_constants import folder_logs, min_confidence, file_msh_articleDB
from os.path import sep
from DataCreators import ArticleDB
import text_preparation

correct_confidence = []
incorrect_confidence = []
min_conf_count = 0
count_flukes = 0
erroneous_acronyms = {}
report = cPickle.load(
    open(folder_logs + "benchmark" + sep + "report_benchmark_MSH_algo18.pickle", "rb"))
for articleID, expansion_details in report:
    for expansion_detail in expansion_details:
        confidence = expansion_detail[4]
        if(expansion_detail[1]):
            if(confidence != min_confidence):
                correct_confidence.append(confidence)
            else:
                count_flukes += 1
        else:
            if(confidence != min_confidence):
                incorrect_confidence.append(confidence)
            else:
                min_conf_count += 1
                if(expansion_detail[0] not in erroneous_acronyms):
                    erroneous_acronyms[expansion_detail[0]] = []
                erroneous_acronyms[expansion_detail[0]].append(articleID)

print("Correct: %d, Incorrect: %d (%f are invalid)" % (len(correct_confidence), len(
    incorrect_confidence) + min_conf_count, min_conf_count / (len(incorrect_confidence) + min_conf_count)))
print("%d flukes" % count_flukes)
print("%d on minimum confidence" % min_conf_count)
print(sorted(erroneous_acronyms, key=lambda line: line[0].lower))
for acronym in erroneous_acronyms:
    print("%s,%d" %(acronym, len(erroneous_acronyms[acronym])))

mshArticleDB = ArticleDB.load(file_msh_articleDB)

for acronym in erroneous_acronyms:
    print(acronym)
    for articleID in erroneous_acronyms[acronym]:
        print("\t%s, %s\n" %
              (text_preparation.toAscii(articleID), text_preparation.toAscii(mshArticleDB[articleID])))


def showPlot(correct_confidence, incorrect_confidence):
    from matplotlib import pyplot as plt
    plt.subplot(211)
    plt.title("correct")
    plt.hist(correct_confidence, bins=100)
    plt.subplot(212)
    plt.title("incorrect")
    plt.hist(incorrect_confidence, bins=100)
    plt.show()

#showPlot(correct_confidence, incorrect_confidence)
"""

"""
**************************************************************************************************
Simulate an ensemble of LDA and SVC (based on who is more confident (without normalization)

Ensemble not giving statistically significant gain. Need to look for improvements other places.
Gain: 40 out of 12445 predictions, not statistically significant
"""
"""
import os
import cPickle
from string_constants import folder_logs, file_lda_model
from matplotlib import pyplot as plt
import numpy

def simulateEnsemble(report_lda, report_svc, gap):
    simulatedEnsemble = []
    corrected_in_lda = 0
    lost_from_lda = 0
    corrected_in_svc = 0
    lost_from_svc = 0
    for articleID in report_lda:
        expansion_details_lda = report_lda[articleID]
        expansion_details_svc = report_svc[articleID]
        for expansion_detail_lda in expansion_details_lda:
            acronym1 = expansion_detail_lda[0]
            expansion_detail_svc = [element for element in expansion_details_svc if element[0] == acronym1][0]
            confidence_lda = expansion_detail_lda[4]
            confidence_svc = expansion_detail_svc[4]
            if (confidence_lda - confidence_svc >= gap):
                simulatedEnsemble.append([articleID, acronym1, expansion_detail_lda[1]])
                if (expansion_detail_lda[1] and not expansion_detail_svc[1]):
                    corrected_in_svc += 1
                if (expansion_detail_svc[1] and not expansion_detail_lda[1]):
                    lost_from_svc += 1
            else:
                simulatedEnsemble.append([articleID, acronym1, expansion_detail_svc[1]])
                if (expansion_detail_svc[1] and not expansion_detail_lda[1]):
                    corrected_in_lda += 1
                if (expansion_detail_lda[1] and not expansion_detail_svc[1]):
                    lost_from_lda += 1
    
    #print("SVC: corrected: %d, lost: %d" % (corrected_in_svc, lost_from_svc))
    #print("LDA: corrected: %d, lost: %d" % (corrected_in_lda, lost_from_lda))
    correct_in_ensemble = 0
    wrong_in_ensemble = 0
    for entry in simulatedEnsemble:
        if (entry[2]):
            correct_in_ensemble += 1
        else:
            wrong_in_ensemble += 1
    
    #print("Ensemble: correct: %d, wrong: %d" % (correct_in_ensemble, wrong_in_ensemble))
    return correct_in_ensemble


report_lda = cPickle.load(open(folder_logs+"benchmark"+os.sep+"report_benchmark_MSH_algo10.pickle", "rb"))
report_svc = cPickle.load(open(folder_logs+"benchmark"+os.sep+"report_benchmark_MSH_algo7.pickle", "rb"))
report_lda = dict(report_lda)
report_svc = dict(report_svc)

ensemble_successes=[]
for gap in numpy.arange(-1.0,3.0,0.01):
    ensemble_successes.append(simulateEnsemble(report_lda, report_svc, gap))
    
plt.plot(ensemble_successes, label="ensemble")
plt.plot([10069]*len(ensemble_successes), label="SVC")
ensemble_successes.append(10069)
plt.yticks(ensemble_successes)
plt.legend()
plt.show()
print max(ensemble_successes)
"""

"""
**************************************************************************************************
Evaluate confidence scores for predictions on success and failure
"""
"""
from __future__ import division
import cPickle
from matplotlib import pyplot as plt
from string_constants import folder_logs, min_confidence
from os.path import sep

correct_confidence = []
incorrect_confidence = []
min_conf_count = 0
count_flukes = 0
report = cPickle.load(
    open(folder_logs + "benchmark"+sep+"report_benchmark_MSH_algo7.pickle", "rb"))
for articleID, expansion_details in report:
    for expansion_detail in expansion_details:
        confidence = expansion_detail[4]
        if(expansion_detail[1]):
            if(confidence != min_confidence):
                correct_confidence.append(confidence)
            else:
                count_flukes+=1
        else:
            if(confidence != min_confidence):
                incorrect_confidence.append(confidence)
            else:
                min_conf_count += 1

print("Correct: %d, Incorrect: %d (%f are invalid)" % (len(correct_confidence), len(
    incorrect_confidence) + min_conf_count, min_conf_count / (len(incorrect_confidence) + min_conf_count)))
print("%d flukes" %count_flukes)

plt.subplot(211)
plt.title("correct")
plt.hist(correct_confidence, bins=100)

plt.subplot(212)
plt.title("incorrect")
plt.hist(incorrect_confidence, bins=100)

plt.show()
"""

"""
**************************************************************************************************
Shuffle articleDBs (take care to import OrderedDict into the program where this is loaded)
"""
"""
from collections import OrderedDict
import random

from DataCreators import ArticleDB
from string_constants import file_msh_articleDB,\
    file_msh_articleDB_shuffled, file_articledb, file_articledb_shuffled


articleDB = ArticleDB.load(path=file_articledb)
items = articleDB.items()
random.shuffle(items)
shuffledArticleDB = OrderedDict(items)
ArticleDB.dump(shuffledArticleDB, path = file_articledb_shuffled)
"""

"""
**************************************************************************************************
Package LDA model into one piece
"""
"""
from DataCreators.LDAModel import SavedLDAModel
from string_constants import file_lda_model, file_lda_gensim_dictionary,\
    file_lda_articleIDToLDA, file_msh_articleDB, file_lda_model_all
import cPickle
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

def loadFromPath(path):
    return cPickle.load(open(path, "rb"))
ldaModel = LdaModel.load(file_lda_model)
dictionary = Dictionary.load(file_lda_gensim_dictionary)
articleIDToLDADict = loadFromPath(file_lda_articleIDToLDA)
articleDBused = file_msh_articleDB
stem_words = False
numPasses=1
path = file_lda_model_all

SavedLDAModel.save(ldaModel, dictionary, articleIDToLDADict, articleDBused, stem_words, numPasses, path)
"""
"""
**************************************************************************************************
See effect of article length on algo performance
"""

"""
import cPickle

from string_constants import folder_logs, file_msh_articleDB
from DataCreators import ArticleDB
from matplotlib import pyplot as plt


report = cPickle.load(open(folder_logs+"report_benchmark_MSH_algo9.pickle", "rb"))
articleDB = ArticleDB.load(file_msh_articleDB)

lengths_forCorrect = []
lengths_forWrong = []

for articleID, expansion_details in report:
    for expansion_detail in expansion_details:
        if(expansion_detail[1]):
            lengths_forCorrect.append(len(articleDB[articleID]))
        else:
            lengths_forWrong.append(len(articleDB[articleID]))

plt.subplot(121)
plt.hist(lengths_forCorrect, bins=50)
plt.yticks(range(0,851,50))
plt.xticks(range(0,6001, 500))
plt.grid()
plt.title("Histogram of lengths for Correct prediction")

plt.subplot(122)
plt.hist(lengths_forWrong, bins=50)
plt.yticks(range(0,850,50))
plt.xticks(range(0,6000, 500))
plt.grid()
plt.title("Histogram of lengths for Wrong prediction")

plt.show()
"""

"""
**************************************************************************************************
Compare reports on MSH from two algos
"""
"""
import os
import cPickle
from string_constants import folder_logs, file_lda_model

report_lda = cPickle.load(open(folder_logs+"benchmark"+os.sep+"report_benchmark_MSH_algo7.pickle", "rb"))
report2 = cPickle.load(open(folder_logs+"benchmark"+os.sep+"report_benchmark_MSH_algo10.pickle", "rb"))
report_lda = dict(report_lda)
report2 = dict(report2)

#print where 1 is correct 2 is wrong
print("articleID,acronym,correct_expansion,wrong_expansion2,isOneMoreConfident")
oneCorrectTwoWrong = []
for articleID in report_lda:
    expansion_details_lda = report_lda[articleID]
    expansion_details2 = report2[articleID]
    for expansion_detail_lda in expansion_details_lda:
        acronym1 = expansion_detail_lda[0]
         
        expansion_detail2 = [element for element in expansion_details2 if element[0]==acronym1][0]
        
        if(expansion_detail_lda[1]==True and expansion_detail2[1]==False):
            isOneMoreConfident = expansion_detail_lda[4]>=expansion_detail2[4]
            oneCorrectTwoWrong.append([articleID,acronym1,expansion_detail_lda[2],expansion_detail2[3],isOneMoreConfident])      

for entry in sorted(oneCorrectTwoWrong, key=lambda entry: entry[1].lower()):
    print("%s,%s,%s,%s,%s" %(entry[0],entry[1],entry[2],entry[3],entry[4]))
"""

"""
**************************************************************************************************
Print the expansions along with article ID of the following acronyms:
"DI", "Ice", "CCl4", "DAT", "TSF", "Orf", "ADP", "DBA", "MCC", "SS", "CDR"
"""
"""
from DataCreators import AcronymDB
from string_constants import file_msh_acronymDB

acronymDB = AcronymDB.load(file_msh_acronymDB)

for acronym in ["DI", "Ice", "CCl4", "DAT", "TSF", "Orf", "ADP", "DBA", "MCC", "SS", "CDR"]:
    expansions = []
    for expansion, articleID, defCount in acronymDB[acronym]: 
        if expansion not in expansions:
            expansions.append(expansion)
            print(acronym, expansion, articleID)
"""
"""
**************************************************************************************************
Verify all MSH true expansions by visual expansion
"""
"""
from DataCreators import AcronymDB
from string_constants import file_msh_acronymDB

acronymDB = AcronymDB.load(file_msh_acronymDB)

for acronym in sorted(acronymDB.keys(), key=lambda entry:entry.lower()):
    expansions_set = []
    expansions = []
    for expansion, articleID, defcount in acronymDB[acronym]:
        if expansion not in expansions_set:
            expansions_set.append(expansion)
            expansions.append([expansion, articleID])
    print "%s: %s" %(acronym, expansions)
"""

"""
**************************************************************************************************
Check what's the role of PMID in MSH Corpus
"""
"""
from arff import load
import os

pmids = {}
folder = r"C:\Cloud\github\AcroDisam\server\storage\data_all\MSHCorpus\arff"
for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)
    for row in load(file_path):
        pmid = row.PMID
        if(pmid not in pmids):
            pmids[pmid] = []
        pmids[pmid].append(file_name)

for pmid in pmids:
    if len(pmids[pmid])>1:
        print("PMID %s is repeated" %str(pmid))
print(len(pmids.keys()))
"""
"""
PMID is a unique identifier of citation text in the MSH corpus
"""

# print(pmids)

"""
**************************************************************************************************
Create a smaller DBs for debugging/testing scripts
Steps:
- Copy the following files to a new_folder:
        scraped_article_info.csv
        scraped_articles.csv
        scraped_definitions.csv
        vectorizer
        vectorizer_01.npy
        vectorizer_02.npy
        wordsEn.txt
- Keep relevant articles from scraped_articles in scraped_articles.csv
- Change data folder in string_constants to point to new_folder
- Run the script below
"""
"""
import csv
import os

from DataCreators import ArticleDB, AcronymDB, ArticleInfoDB
from string_constants import file_scraped_articles_list,\
    file_scraped_definitions_list, file_scraped_article_info
import sys

csv.field_size_limit(sys.maxint)

## Get articleIDs which are needed in each DB
articleIds = []
for file in file_scraped_articles_list:
    for line in csv.DictReader(open(file,"rb"), delimiter=","):
        articleIds.append(line["article_id"])

## Save smaller version of scraped definitions csv
file_path = file_scraped_definitions_list[0]    
small_path = file_path+".csv"
with open(file_path, "rb") as infile, open(small_path, "wb") as outfile:
    source_csv = csv.DictReader(infile, delimiter=",")
    
    headers = ["acronym","acronym_expansion","article_id"]
    small_csv = csv.DictWriter(outfile, fieldnames=headers)
    small_csv.writeheader()
    
    for line in source_csv:
        if(line["article_id"] in articleIds):
            small_csv.writerow(line)

os.remove(file_path)
os.rename(small_path, file_path)

## Save smaller version of scraped article info csv
file_path = file_scraped_article_info    
small_path = file_path+".small"
with open(file_path, "rb") as infile, open(small_path, "wb") as outfile:
    source_csv = csv.DictReader(infile, delimiter=",")
    
    headers = ["article_id","article_title","article_source"]
    small_csv = csv.DictWriter(outfile, fieldnames=headers)
    small_csv.writeheader()
    
    for line in source_csv:
        if(line["article_id"] in articleIds):
            small_csv.writerow(line)
    
os.remove(file_path)
os.rename(small_path, file_path)

## Make new DB files
ArticleDB.createFromScrapedArticles()
AcronymDB.createFromScrapedDefinitions()
ArticleInfoDB.dump(ArticleInfoDB.fromCSV())
"""

"""
**************************************************************************************************
Handle missing article aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvU2h1amFheg== in articleDB
"""
"""
import csv
import sys

from DataCreators import ArticleDB
from string_constants import file_scraped_articles


articleDB = ArticleDB.load()
csv.field_size_limit(sys.maxint)
file = csv.DictReader(open(file_scraped_articles,"rb"), delimiter=",")

changeSuccessful = False

for line in file:
    if(line["article_id"] == "aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvU2h1amFheg=="):
        articleDB[line["article_id"]] = line["article_text"]
        changeSuccessful = True

if changeSuccessful:
    ArticleDB.dump(articleDB)
else:
    print "badluck!"
"""

"""
**************************************************************************************************
Check if duplication in scraped_articles is creating article_id: article_text pairs where the acronym is not present
"""
"""
from __future__ import division
from DataCreators import AcronymDB, ArticleDB
acronymDB = AcronymDB.load()
articleDB = ArticleDB.load()

errors = []
missing_articles=[]
numSuccesses = 0

for acronym in acronymDB.keys():
    for expansion, article_id, def_count in acronymDB[acronym]:
        if(article_id in articleDB):
            article = articleDB[article_id].lower()
            expansion = expansion.lower()
            if(acronym.lower() not in article):# or expansion not in article):
                errors.append([acronym, expansion, article_id])
            else:
                numSuccesses+=1
        else:
            missing_articles.append(article_id)

print "errors:", len(errors), "successes: ", numSuccesses, "%age error: ", len(errors)*100/(len(errors)+numSuccesses)
print errors[:10]

missing_articles = set(missing_articles)
print "missing articles: " +str(len(missing_articles))
print missing_articles
"""

"""
**************************************************************************************************
Test text_preparation.getCleanedWords() by visual inspection
"""
"""
from DataCreators import ArticleDB
import text_preparation

articles = ArticleDB.load()

article = articles.values()[0]

print article 

words = text_preparation.getCleanedWords(article)
print words
"""

"""
Re-format the old CSV files to reflect new changes
new scraped_article_infos file
    format: article_id, article_title, article_source
""""""
import cPickle

from string_constants import file_article_infodb, file_scraped_article_info
import csv


article_info = cPickle.load(open(file_article_infodb, "rb"))

headers = ["article_id", "article_title", "article_source"]
csv_writer = csv.DictWriter(
    open(file_scraped_article_info, "wb"), fieldnames=headers)
csv_writer.writeheader()

for article_id in article_info.keys():
    rowdict = {"article_id": article_id, "article_title": article_info[
        article_id][0]}
    
    if(len(article_info[article_id]) == 2):
        rowdict["article_source"] = article_info[article_id][1]

    csv_writer.writerow(rowdict)
"""

"""
**************************************************************************************************
Check scraped_articles for duplicate entries
""""""
from collections import Counter
import csv
import sys

from matplotlib import pyplot

from string_constants import file_scraped_articles


csv.field_size_limit(sys.maxint)
scraped_articles = csv.DictReader(open(file_scraped_articles, "rb"), delimiter=",")

tracker = Counter()
line_num = 1
for line in scraped_articles:
    article_id = line["article_id"]
     
    # if article_id in tracker:
    #    print "First, line " +str(line_num)
    #    print tracker[article_id]
    #    print "Second, line " +str(line_num)
    #    print line["article_text"]
    #    raise Exception
    # line_num+=1
    tracker[article_id] += 1

#unique_repetition_values = list(set(tracker.values()))    
#pyplot.plot(tracker.values())
#pyplot.yticks(unique_repetition_values)
#pyplot.hist(tracker.values(), bins=len(unique_repetition_values))
#pyplot.xticks(unique_repetition_values)
#pyplot.grid()
#pyplot.show()

duplicates = {}
for article_id in tracker.keys():
    if(tracker[article_id]!=1):
        duplicates[article_id] = tracker[article_id]
 
articles_lost = sum(duplicates.values())-len(duplicates.keys())
print "articles lost: " +str(articles_lost)
#unique_duplication_counts = list(set(duplicates.values()))    
#pyplot.subplot(211)
#pyplot.title("plotting duplicate counts")
#pyplot.plot(duplicates.values())
#pyplot.yticks(unique_duplication_counts)
#pyplot.grid()
#
#pyplot.subplot(212)
#pyplot.title("histogram of duplicate counts")
#pyplot.hist(duplicates.values(), bins=len(unique_duplication_counts))
#pyplot.grid()
#
#pyplot.show()

#print type(tracker.values())
#print len(tracker.values())
#print sum(tracker.values())
#print tracker.values()
"""

"""
**************************************************************************************************
Re-format the old CSV files to reflect new changes
scraped_articles
    old format: article_id,article_text,article_path
    new format: article_id,article_text
""""""
import csv
import sys

from string_constants import file_scraped_articles


csv.field_size_limit(sys.maxint)
old_article_file = csv.DictReader(open(file_scraped_articles, "rb"), delimiter=",")

headers = ["article_id", "article_text"]
new_article_file = csv.DictWriter(open(file_scraped_articles + ".new", "wb"), fieldnames=headers)
new_article_file.writeheader()

for line in old_article_file:
    article_id = line["article_id"]
    article_text = line["article_text"]
    
    new_article_file.writerow({"article_id": article_id, "article_text":article_text})
"""

"""
**************************************************************************************************
Re-format the old CSV files to reflect new changes
scraped_definitions
    old format: acronym,acronym_expansion,article_id,article_title
    new format: acronym,acronym_expansion,article_id
""""""
import csv
import sys
from string_constants import file_scraped_definitions

csv.field_size_limit(sys.maxint)
old_acronym_file = csv.DictReader(open(file_scraped_definitions, "rb"), delimiter=",")

headers = ["acronym","acronym_expansion","article_id"]
new_acronym_file = csv.DictWriter(open(file_scraped_definitions+".new", "wb"), fieldnames=headers)
new_acronym_file.writeheader()

for line in old_acronym_file:
    acronym = line["acronym"]
    expansion = line["acronym_expansion"]
    article_id = line["article_id"]
    
    new_acronym_file.writerow({"acronym": acronym, "acronym_expansion": expansion, "article_id": article_id})
"""

"""
**************************************************************************************************
Examine acronymDB, and extract articleID: [article_title, article_source] from it
"""
"""
from string_constants import file_scraped_definitions, file_article_infodb, file_scraped_articles
import csv
import sys
import cPickle
from Logger import logger
logger = logging.getLogger(__name__)


csv.field_size_limit(sys.maxint)
old_acronym_file = csv.DictReader(open(file_scraped_definitions, "rb"), delimiter=",")

# dictionary of the form: articleID: [article_title, article_source]
article_info = {}

# put article_title in dictionary
for line in old_acronym_file:
    article_id = line["article_id"]
    article_title = line["article_title"]
    if article_id in article_info:
        if article_title not in article_info[article_id]:
            article_info[article_id].append(article_title)
            logger.critical("article_info creation: " + article_id + ": " + str(article_info[article_id]))
        else:
            continue
    else:
        article_info[article_id] = [article_title]
        
# put article_source in dictionary
old_article_file = csv.DictReader(open(file_scraped_articles, "rb"), delimiter=",")
for line in old_article_file:
    article_id = line["article_id"]
    article_source = line["article_path"]
    if(len(article_info[article_id]) == 1):
        article_info[article_id].append(article_source)
    else:  # (len(article_info[article_id])>2):
        article_info[article_id][-1] += "," + article_source
    
    if((len(article_info[article_id]) > 2)):
        logger.critical("article_info creation: " + article_id + ": " + str(article_info[article_id]))

    

cPickle.dump(article_info, open(file_article_infodb, "wb"), protocol=2)
"""

"""
**************************************************************************************************
Remove article_title from acronymdb
"""
"""
import DataCreators.AcronymDB as AcronymDB

acronymDB = AcronymDB.load()

for acronym in acronymDB.keys():
    expansions = acronymDB[acronym]
    new_expansions = []
    for expansion in expansions:
        new_expansions.append(expansion[:-2])
    
    for expansion in new_expansions:
        if(len(expansion)!=2):
            raise Exception
    
    acronymDB[acronym] = new_expansions

AcronymDB.dump(acronymDB)
"""
