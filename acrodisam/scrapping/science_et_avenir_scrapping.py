"""Save articles from "Science et Avenir" as a json file.

Created on jun 4, 2020

@author: mprieur
"""

from datetime import date, timedelta

import sys
import json
import bs4
import requests

from bs4 import BeautifulSoup
from tqdm import tqdm

from string_constants import FOLDER_DATA

SCI_ET_AV_JSON_PATH = FOLDER_DATA+"FrenchData/ScienceEtAvenir/data.json"
START_DATE = date(2019, 1, 1)

def get_json_content():
    """ Load the json containing the articles.
    Returns :
        (dict) : The dictionary of articles
    """
    with open(SCI_ET_AV_JSON_PATH+"") as json_file:
        data = json.load(json_file)
        for key in data:
            data[key] = data[key].replace(u'\xa0', u' ')
            data[key] = data[key].replace(u'\n', u' ')
    return data

def get_inside_text(content):
    """ Get the text inside html element.
    Args :
        content (bs4.element.Tag) : the html element
    Returns :
        (str) : the text
    """
    text = ""
    if isinstance(content, bs4.element.Tag):
        #print("Content" , content)
        for elem in content.contents:
            if isinstance(elem, bs4.element.Tag):
                text += get_inside_text(elem)
            elif not "Crédit" in elem and not ".com" in elem:
                text += elem
        return text
    else:
        return content

def get_text(href):
    """ Get the text from an article.
    Args :
        href (str) : the article link
    Returns :
        text (str) : the article text
    """
    text = ""
    response = requests.get(href).text
    article = BeautifulSoup(response, "html.parser")
    section = article.find('article', class_="container-inside-right")
    if not section is None:
        markup = section.find('div')
        for para in markup.contents:
            text += get_inside_text(para)
    return text

def save_as_json(dict_articles):
    """Same the article ditionary to a json file.
    Args (dict) : the articles with their link as key
    """
    with open(SCI_ET_AV_JSON_PATH, 'w') as out_file:
        json.dump(dict_articles, out_file)

def get_articles(num_articles):
    """Return the articles extracted from "Science et Avenir".
    Args :
        num_articles (Int) : the number of article to obtain
    Returns :
        dict_articles (dict): The dictionnary of plain text article with their link as key.
    """
    base_link = "https://www.sciencesetavenir.fr/index/"
    part_link = "https://www.sciencesetavenir.fr/"
    dict_articles = {}
    date_iterator = START_DATE
    delta = timedelta(days=1)
    end_date = date.today()+ delta
    pbar = tqdm(total=num_articles)
    while len(dict_articles) < num_articles and not date == end_date:
        response = requests.get(base_link+date_iterator.strftime("%Y/%m/%d")).text
        articles_index = BeautifulSoup(response, features="lxml")
        articles_section = articles_index.find('div', class_="content-main")
        for link in articles_section.findAll('a'):
            href = link.get('href')
            if isinstance(href, str) and part_link in href and len(dict_articles) < num_articles:
                text = get_text(href)
                if len(text) > 0:
                    dict_articles[href] = text
                    pbar.update(1)
        date_iterator += delta
    pbar.close()
    return dict_articles

if __name__ == "__main__":
    num_article = int(sys.argv[1])
    articles = get_articles(num_article)
    save_as_json(articles)
