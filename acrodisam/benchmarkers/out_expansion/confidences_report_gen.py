"success"'''
Created on Apr 21, 2022

@author: jpereira
'''
import os
import re

import pandas as pd
from string_constants import SCIENCE_WISE_DATASET, FOLDER_LOGS, sep,\
    MSH_ORGIN_DATASET, SDU_AAAI_AD_DEDUPE_DATASET, CS_WIKIPEDIA_DATASET
import DataCreators.ArticleDB
from helper import get_raw_article_db_path

from Logger import logging

logger = logging.getLogger(__name__)

FOLDER_REPORTS = FOLDER_LOGS 

def load_resuls_confs_to_pd(dataset_name, expander):
    try:
        df_conf = pd.read_csv(FOLDER_REPORTS + sep + "confidences_"+dataset_name+"_confidences_"+expander+".csv")
        df_results = pd.read_csv(FOLDER_REPORTS + sep +"quality_results_"+dataset_name+"_confidences_"+expander+".csv")
        #df2 = df_conf.join(df_results, lsuffix="conf", rsuffix="results")
        df_conf_results = pd.merge(df_conf, df_results,  how='left', left_on=['fold','doc_id','acronym'], right_on = ['fold','doc id','acronym']).drop(columns=["doc id", "confidence"])
        df_conf_results["success"] = df_conf_results["success"].astype(int)
    
        if df_conf_results.empty or df_conf_results.isnull().values.any():
            print("Couldn't create df for this expander: " + expander)
            return None
    
        return df_conf_results
    except Exception as e:
        print(e)
        return None

def get_expanders(datasetname):
    expanders = []
    for file in os.listdir(FOLDER_REPORTS):
            m = re.match('^confidences_'+datasetname+'_confidences_([.:=_\-\\w]+).csv$', file)
            if m:
                expander = m.group(1)
                expanders.append(expander)
    return expanders

def create_expander_results_dataframe(dataset_name):
    expanders_list = get_expanders(dataset_name)
    df_expanders = None
    for expander in expanders_list:
        print(expander)
        df = load_resuls_confs_to_pd(dataset_name, expander)
        if not isinstance(df, pd.DataFrame):
            continue
        
        if isinstance(df_expanders, pd.DataFrame):
            df_expanders_tmp = pd.merge(df_expanders, df,  how='outer', suffixes=[None, expander], on=['fold', 'doc_id', 'acronym', 'actual_expansion']).rename(columns={'confidences_json_dict': 'confidences_json_dict_'+expander, 'predicted_expansion': "predicted_expansion_"+expander, "success": "success_"+expander}, errors="raise")
            if df_expanders_tmp.isnull().values.any():
                print("Missmatch DFs: " + expander)
                #continue
            df_expanders = df_expanders_tmp
        else:
            df_expanders = df.rename(columns={'confidences_json_dict': 'confidences_json_dict_'+expander, 'predicted_expansion': "predicted_expansion_"+expander, "success": "success_"+expander}, errors="raise")
           
    articles_db = DataCreators.ArticleDB.load(get_raw_article_db_path(dataset_name))
    df_articles = pd.DataFrame(articles_db.items(), columns=['doc_id', 'text'])
    df_final = pd.merge(df_expanders, df_articles, how='left') 
    return df_final
    
if __name__ == '__main__':
    datasets = [#SCIENCE_WISE_DATASET,
                #SDU_AAAI_AD_DEDUPE_DATASET,
                MSH_ORGIN_DATASET, 
                #"CSWikipedia_res-dup"
                ]
    for dataset_name in datasets:
        logger.critical("Creating report for dataset: " + dataset_name)
        df = create_expander_results_dataframe(dataset_name)
        df.to_csv(FOLDER_REPORTS + "confidences_report_"+dataset_name+".csv")

    
    
    