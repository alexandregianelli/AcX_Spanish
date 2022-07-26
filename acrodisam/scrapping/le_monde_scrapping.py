"""Save articles from "Le Monde" as a json file.

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

LE_MONDE_JSON_PATH = FOLDER_DATA+"FrenchData/LeMonde/data.json"
START_DATE = date(2019, 1, 1)

def get_json_content():
    """ Load the json containing the articles.
    Returns :
        (dict) : The dictionary of articles
    """
    with open(LE_MONDE_JSON_PATH+"") as json_file:
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
    while isinstance(content, bs4.element.Tag):
        if len(content.contents) == 0:
            return ''
        content = content.contents[0]
    return content

def get_text(href):
    """ Get the text from an article.
    Args :
        href (str) : the article link
    Returns :
        text (str) : the article text
    """
    response = requests.get(href).text
    article = BeautifulSoup(response, "html.parser")
    text = ""
    for para in article.find_all('p', class_="article__paragraph"):
        for content in para.contents:
            text += get_inside_text(content)
    return text

def save_as_json(dict_articles):
    """Same the article ditionary to a json file.
    Args (dict) : the articles with their link as key
    """
    with open(LE_MONDE_JSON_PATH, 'w') as out_file:
        json.dump(dict_articles, out_file)

def get_articles(num_articles):
    """Return the articles extracted from "Le Monde".
    Args :
        num_articles (Int) : the number of article to obtain
    Returns :
        dict_articles (dict): The dictionnary of plain text article with their link as key.
    """
    base_link = "https://www.lemonde.fr/archives-du-monde/"
    dict_articles = {}
    date_iterator = START_DATE
    delta = timedelta(days=1)
    end_date = date.today()+ delta
    pbar = tqdm(total=num_articles)
    while len(dict_articles) < num_articles and not date_iterator >= end_date:
        response = requests.get(base_link+date_iterator.strftime("%d-%m-%Y/")).text
        date_index = BeautifulSoup(response, features="lxml")
        part_link = "article/"+date_iterator.strftime("%Y/%m/%d")
        for link in date_index.find_all('a'):
            href = link.get('href')
            if part_link in href and len(dict_articles) < num_articles:
                dict_articles[href] = get_text(href)
                pbar.update(1)
        date_iterator += delta
    pbar.close()
    return dict_articles

if __name__ == "__main__":
    num_article = int(sys.argv[1])
    articles = get_articles(num_article)
    save_as_json(articles)
