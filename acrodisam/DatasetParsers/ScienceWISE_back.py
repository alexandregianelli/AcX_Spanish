import csv
import os
import time
import urllib2

import logging
from string_constants import file_ScienceWise_index_train,\
    folder_scienceWise_pdfs

logger = logging.getLogger(__name__)


def downloadPdfs():
    with open(file_ScienceWise_index_train, "r") as file:
        reader = csv.DictReader(file, delimiter=",")
        for line in reader:
            pdfID = line["ARXIV_ID"]
            filename = _arxivIDToFilename(pdfID)
            try:
                if(os.path.exists(folder_scienceWise_pdfs + filename)):
                    logger.debug("present already " + pdfID)
                    continue
                _downloadPdf(pdfID)
                logger.debug("successfully downloaded " + pdfID)
                time.sleep(5 * 60)
            except:
                logger.exception("Error in file " + pdfID)


def _arxivIDToFilename(arxivID):
    filename = arxivID.replace("/", "_").replace("\\", "_")
    filename = filename + ".pdf"
    return filename


def _downloadPdf(pdfID):
    url = "http://arxiv.org/pdf/" + pdfID + ".pdf"
    response = urllib2.urlopen(url)

    filename = _arxivIDToFilename(pdfID)
    local_file = open(folder_scienceWise_pdfs + filename, "wb")
    local_file.write(response.read())

    response.close()
    local_file.close()


def visualize():
    import matplotlib.pyplot as plt

    data = {}
    with open(file_ScienceWise_index_train, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for line in reader:
            acronym = line[1]
            expansion = line[-1]
            if(not acronym in data):
                data[acronym] = []
            if(not expansion in data[acronym]):
                data[acronym].append(expansion)

    print("number of acronyms", len(data.keys()))

    numAmbs = []
    for key in data.keys():
        num = len(data[key]) - 1
        if(num > 0):
            numAmbs.append(num)

    print(len(numAmbs))
    print(max(numAmbs))

    plt.subplot(121)
    plt.title("Histogram of number of ambiguities")
    plt.grid()
    plt.yticks(range(1, 66))
    plt.hist(numAmbs)

    plt.subplot(122)
    plt.title("Plot of number of ambiguities")
    plt.plot(numAmbs)

    plt.show()

if __name__ == "__main__":
    visualize()
