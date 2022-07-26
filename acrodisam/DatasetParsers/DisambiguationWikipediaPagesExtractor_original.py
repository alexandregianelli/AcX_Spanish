import os
import wikipedia
from bs4 import BeautifulSoup
import requests
import json
import re
from string_constants import folder_cs_wikipedia_corpus, folder_cs_wikipedia_generated
DATA_FILE_PATH = folder_cs_wikipedia_corpus + "cs_wikipedia_acronyms.json"
SEARCH_URL = "https://en.wikipedia.org/w/index.php?title=Special:Search&go=Go&search="
DISAMBIGUATION_URL = "https://en.wikipedia.org/wiki/%s_(disambiguation)"

#Part of this code from Paper Acronym Disambiguation: A Domain Independent Approach


"""
def get_doc(url):
    text_all = list()
    if url == "":
        # print(url)
        return ""
    response = requests.get(url)
    soup = BeautifulSoup(markup=response.text, features="lxml")
    if soup is None:
        # print(url)
        return ""
    content = soup.find(name="div", attrs={"class": "mw-parser-output"})
    if content is None:
        # print(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        # print(url)
        return ""

    for p in list_p:
        text_all.append(str(p.text))
  
    return " ".join(text_all)
"""

def get_doc(url):
    text = list()
    if url == "":
        print(url)
        return ""
    response = requests.get(url)
    soup = BeautifulSoup(markup=response.text, features="lxml")
    if soup is None:
        print(url)
        return ""
    content = soup.find(name="div", attrs={"class": "mw-parser-output"})
    if content is None:
        print(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        print(url)
        return ""
    for p in list_p:
        text.append(str(p.text))
    return " ".join(text)


def get_docFromPage(responseText):
    text = list()

    soup = BeautifulSoup(markup=responseText, features="lxml")
    if soup is None:
       # print(url)
        return ""
    content = soup.find(name="div", attrs={"class": "mw-parser-output"})
    if content is None:
       # print(url)
        return ""
    list_p = content.findAll(name="p")
    if list_p is None:
        #print(url)
        return ""
    for p in list_p:
        text.append(str(p.text))
    return " ".join(text)

def get_pages(query):
    pages = list()
    if len(query.strip()) <= 0:
        raise ValueError

    response = requests.get(SEARCH_URL + str(query))
    soup = BeautifulSoup(markup=response.text, features="lxml")

    if soup is None:
        raise Exception

    if "search" in str(soup.title).lower():
        result_ul = soup.find(name="ul", attrs={"class": "mw-search-results"})
        results_list = result_ul.find_all("li")

        for li in results_list:
            li_div = li.find(name="div", attrs={"class": "mw-search-result-heading"})
            a = li_div.find("a")
            link = "https://en.wikipedia.org" + a["href"]
            heading = str(a.text)
            pages.append((link, heading))

        return pages
    else:
        return wikipedia.summary(query)




def get_acronyms(query, csArticle):
    possibilities = list()
    extractedLinks = [csArticle["link"].strip().lower()]
    extractedDocs = [csArticle["content"].strip().lower()]

    response = requests.get(DISAMBIGUATION_URL % str(query))
    print(DISAMBIGUATION_URL % str(query))

    query = query.lower()
    if response.status_code != 404:
        print("Disambiguation Page Exists :D")
        soup = BeautifulSoup(markup=response.text, features="lxml")
        if soup is None:
            return None
        div = soup.find("div", attrs={"class": "mw-parser-output"})
        all_uls = div.findAll("ul")

        for ul in all_uls:
            all_lis = ul.findAll("li")
            for li in all_lis:
                a = li.find("a")
                if a is None:
                    continue
                url = "https://en.wikipedia.org" + a["href"]
                
                if not url.strip().lower() in extractedLinks:
                    extractedLinks.append(url.strip().lower())
                    content = str(get_doc(url))#.lower()
                    if not content.strip().lower() in extractedDocs:
                        extractedDocs.append(content.strip().lower())
                        possibilities.append(content)
                    else:
                        print("Removed duplicate doc for: " + str(query))
                else:
                    print("Removed duplicate link for: " + str(url))


    results = wikipedia.search(query=query, results=10)
    print(results)
    if len(results) <= 0:
        return possibilities

    for each_result in results:
        try:
            url = wikipedia.page(each_result).url#.lower()
            if not url.strip().lower() in extractedLinks:
                extractedLinks.append(url.strip().lower())
                content = str(get_doc(url))#.lower()
                if not content.strip().lower() in extractedDocs:
                    extractedDocs.append(content.strip().lower())
                    possibilities.append(content)
                else:
                    print("search Removed duplicate doc for: " + str(query))
            else:
                print("search Removed duplicate link for: " + str(url))
        except Exception:
            continue

    return possibilities


# print(get_acronyms("FDS"))

new_acronyms = dict()
acronyms = json.load(open(DATA_FILE_PATH, mode="r"))
print("Existing File Loaded...")
print("Total Acronyms: %s" % len(acronyms))
i = 1
for item in acronyms: 
    print("Acronyms for %s" % item)
    
    # get_acronyms(item)
    try:
        possibilities = get_acronyms(item, acronyms[item])
    except Exception as e:

        print(e)
        print("Exception Occurred")
        continue
    if possibilities is None or len(possibilities) == 0:
        # del acronyms[item]
        print("No Possibilities :(")
        continue

    acronyms[item]["possibilities"] = possibilities
    # print(possibilities)
    print("%s More to go" % (len(acronyms) - i))
    i += 1
    new_acronyms[item] = acronyms[item]
    print("------------------------------------------------------------")
#    if i == 10:
#        break
    
json.dump(new_acronyms, open(folder_cs_wikipedia_generated+ "cs_wikipedia.json", "w"))

