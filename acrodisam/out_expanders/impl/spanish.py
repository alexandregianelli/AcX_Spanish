from Logger import logging
from helper import getDatasetGeneratedFilesPath, ExecutionTimeObserver
from inputters import TrainOutDataManager
import numpy as np
from out_expanders._base import OutExpanderArticleInput

from text_preparation import get_expansion_without_spaces

from .._base import OutExpanderFactory, OutExpander
from run_config import RunConfig
from typing import Optional

import yake


def train_doc_to_dict(
    articles_db,
    acronym_db):

    spanish_acronym_dict = {}

    for acronym, expansions in acronym_db.items():
        for expansion in expansions:
            exp = expansion[0]
            pmid = expansion[1]
            text = articles_db[pmid]

            keywords = yake.KeywordExtractor(lan="spanish", n=5).extract_keywords(text)
            if acronym not in spanish_acronym_dict:
                spanish_acronym_dict[acronym] = []
            spanish_acronym_dict[acronym].append([exp, keywords]) 
    
    return spanish_acronym_dict

def test_doc_to_dict(
    articles_db,
    acronym_db):

    spanish_acronym_dict = {}

    for acronym, expansions in acronym_db.items():
        for expansion in expansions:
            exp = expansion[0]
            pmid = expansion[1]
            text = articles_db[pmid]

            keywords = yake.KeywordExtractor(lan="spanish", n=1).extract_keywords(text)
            if acronym not in spanish_acronym_dict:
                spanish_acronym_dict[acronym] = []
            spanish_acronym_dict[acronym].append([exp, keywords]) 
    
    return spanish_acronym_dict



class SpanishFactory(OutExpanderFactory):
    def __init__(self,
        run_config: Optional[RunConfig] = RunConfig(),
        **kwargs
        ):
        self.run_name = run_config.name


    def get_expander(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: ExecutionTimeObserver=None,
    ):
        execution_time_observer.start()
        
        
        articles_db = train_data_manager.get_raw_articles_db()
        articles_acronym_db = train_data_manager.get_article_acronym_db()
        acronym_db = train_data_manager.get_acronym_db()
        spanish_acronym_dict = train_doc_to_dict(articles_db, acronym_db)
        execution_time_observer.stop()

        return _Spanish(spanish_acronym_dict)
    

class _Spanish(OutExpander):
    def __init__(self, 
        spanish_acronym_dict
        ):
        self.dict = spanish_acronym_dict

    def process_article(self, out_expander_input: OutExpanderArticleInput):
        predicted_expansions = []

        acronyms_list = out_expander_input.acronyms_list

        article_text = out_expander_input.article.get_raw_text()

        keywords = yake.KeywordExtractor(lan="spanish", n=1).extract_keywords(article_text)

        for acronym in acronyms_list:
            confidence = 1
            max_score = 0
            for pot_expansion in self.dict[acronym]:
                score = 0
                for key in keywords:
                    score += sum(1-n for _, n in [item for item in pot_expansion[1] if item[0] == key[0]])
                if score > max_score:
                    expansion = pot_expansion[0]
                
            predicted_expansions.append((expansion, confidence)) 

        return predicted_expansions
