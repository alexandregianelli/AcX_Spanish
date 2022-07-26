
import logging
import os
import sys
import csv

from pdfminer.high_level import extract_text

from DatasetParsers import FullWikipedia
from acronym_expander import AcronymExpanderFactory, TrainOutDataManager
from inputters import InputArticle
from run_config import RunConfig
from string_constants import (
    FULL_WIKIPEDIA_DATASET,
)


LOGGER = logging.getLogger(__name__)
LOGGER.info("Starting server")

RTP_DIR_PATH = "/home/jpereira/Downloads/RTP_PDF/"



def get_acronym_expansions(file_path):
    try:
        raw_text = extract_text(file_path)
        
        article = InputArticle(raw_text=raw_text, preprocesser=TEXT_PRE_PROCESS)

        expanded_acronyms = EXPANDER.process_article(article)
        # Removes the acronyms with no expansion from the dict
        # Better look for a demo
        exp_acronyms = {
            acro: exp[0]
            for acro, exp in expanded_acronyms.items()
            if exp is not None
        }
        return exp_acronyms
    except Exception:
        LOGGER.critical("Error in processing file: " + file_path)
        LOGGER.exception("")



def main():
    with open(RTP_DIR_PATH + "expansions.csv", 'x') as csvfile:
        exp_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_writer.writerow(["file_name", "acronym", "expansion"])
        
        for filename in os.listdir(RTP_DIR_PATH):
            if filename.endswith(".pdf"):
                acro_exp_dict = get_acronym_expansions(RTP_DIR_PATH + filename)
                if acro_exp_dict:
                    for acro, exp in acro_exp_dict.items():
                        exp_writer.writerow([filename, acro, exp])


if __name__ == "__main__":
    LOGGER.info("Initializing Acronym Expander")

    DATASET_NAME = FULL_WIKIPEDIA_DATASET
    ACRO_EXP_EXTRACTOR = "schwartz_hearst_original"
    TEXT_PRE_PROCESS = FullWikipedia.text_preprocessing

    train_data_manager = TrainDataManager(DATASET_NAME)

    text_representator_name = "doc2vec"
    text_representator_args = ["50", "CBOW", "25", "8"]

    out_expander_name = "svm"
    out_expander_args = [
        "l2",
        "0.1",
        False,
        text_representator_name,
        text_representator_args,
    ]


    run_config = RunConfig(
        name=DATASET_NAME, save_and_load=True, persistent_articles="SQLITE"
    )
    expander_fac = AcronymExpanderFactory(
        text_preprocessor=TEXT_PRE_PROCESS,
        in_expander_name=ACRO_EXP_EXTRACTOR,
        out_expander_name=out_expander_name,
        out_expander_args=out_expander_args,
        follow_links=True,
        run_config=run_config,
    )
    EXPANDER, _ = expander_fac.create_expander(train_data_manager)
    main()
