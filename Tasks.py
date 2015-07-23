#encoding: utf8

from celery import Celery
import requests
import re
from BeautifulSoup import BeautifulSoup as soup

c = Celery('Tasks',broker='redis://localhost:6379/0')

@c.task
def getContent(url,patterns={},ignore_pattern=[]):
    """
    Arugments:
    url(String): target url
    extract_pattern(dict): (name,extract_function)
    """
    ret = {}
    url_text = requests.get(url).text
    bs = soup(url_text)
    # 404 case
    if not bs.find(ignore_pattern[0],attrs=ignore_pattern[1]) == None:
        return None
    if not type(patterns) == dict:
        raise TypeError("Error type of patterns, should be (name,extract_function)!")
    for p in patterns.items():
        ret[p[0]] = bs.find(p[1][0],attrs=p[1][1]).text
    return ret
