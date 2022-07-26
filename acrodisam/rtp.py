
import logging
import os
import sys
import csv

import textract

from AcroExpExtractors.AcroExpExtractor_FR import AcroExpExtractor_FR
from DatasetParsers import french_wikipedia, FullWikipedia
from acronym_expander import AcronymExpanderFactory, TrainDataManager
from inputters import InputArticle
from run_config import RunConfig
from string_constants import (
    FRENCH_WIKIPEDIA_DATASET,
    FULL_WIKIPEDIA_DATASET,
)


LOGGER = logging.getLogger(__name__)
LOGGER.info("Starting server")

RTP_DIR_PATH = "/home/jpereira/Downloads/RTP_PDF/"

import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io

"""
rsrcmgr = PDFResourceManager()
retstr = io.StringIO()
codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
# Create a PDF interpreter object.
interpreter = PDFPageInterpreter(rsrcmgr, device)
# Process each page contained in the document.

def pdfparser(data):

    with open(data, 'rb') as fp:

        text = ""
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
            text += "\n" + retstr.getvalue()
    
        return text

"""

from pdfminer.high_level import extract_text

def get_acronym_expansions(file_path):
    try:
        """
        content = textract.process(file_path)
        content = [content.decode("utf-8")]
        raw_text = content[0]
        """
        #raw_text = pdfparser(file_path)
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
    except textract.exceptions.ShellError as textractexcep:
        LOGGER.warning(textractexcep)
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
    if len(sys.argv) > 1 and sys.argv[1] == "FR":
        DATASET_NAME = FRENCH_WIKIPEDIA_DATASET
        ACRO_EXP_EXTRACTOR = AcroExpExtractor_FR()
        TEXT_PRE_PROCESS = french_wikipedia.preprocess_text
    else:
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

    EXTENSIONS_SUPPORTED = textract.parsers._get_available_extensions()

    if len(sys.argv) > 1 and sys.argv[1] == "FR":
        """
        TODO
        EXPANDER = AcronymExpander_Extension(text_extractor=Extract_PdfMiner(),
                                          textPreProcess=TEXT_PRE_PROCESS,
                                          acroExpExtractor=ACRO_EXP_EXTRACTOR,
                                          expanders=[DISAM_EXPANDER],
                                          articleDB=ARTICLE_DB,
                                          acronymDB=ACRONYM_DB)
        """
        pass
    else:
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
