# Id numbers for the team members
# 60264498,66721660,55488661,43376858

import json
from bs4 import BeautifulSoup
import nltk
import gensim
import os
from pathlib import Path
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore')


def getUserInput():
    d = input("Enter the directory:")
    p = Path(d)
    return p


def loopAllFiles(directory):
    """loop through all csv files in a single director which user enters."""
    extension = 'json'
    os.chdir(directory)
    result = glob.glob('*.{}'.format(extension))
    return result


def walks_dirs(file_path):
    #return the list contains directory
    lst_dir = []
    all_lst = []
    for dirpath, dirname, files in os.walk(file_path):
        lst_dir.append(dirpath)
    for i in lst_dir:
        file_list = loopAllFiles(i)        # error 
        for index in range(len(file_list)):
            file_list[index] = str(i)+"/"+str(file_list[index])
        all_lst += file_list
    return all_lst


def get_content(soup_page):  # Pass in a soup object --> this function will extract the text content in the json and put then into a giant string
    lst_word = []
    lst = []
    for data in soup_page.find_all(["b",'strong']):
        text_page_bold = data.get_text()
        lst_word += nltk.word_tokenize(text_page_bold.lower()) * 2

    for data in soup_page.find_all("h1"):
        text_page_h1 = data.get_text()
        lst_word += nltk.word_tokenize(text_page_h1.lower()) * 8

    for data in soup_page.find_all("h2"):
        text_page_h2 = data.get_text()
        lst_word += nltk.word_tokenize(text_page_h2.lower()) * 6

    for data in soup_page.find_all("h3"):
        text_page_h3 = data.get_text()
        lst_word += nltk.word_tokenize(text_page_h3.lower()) * 4

    for data in soup_page.find_all("title"):
        text_page_title = data.get_text()
        lst_word += nltk.word_tokenize(text_page_title.lower()) * 10

    for data in soup_page.find_all("p"):
        text_page = data.get_text()
        lst_word += nltk.word_tokenize(text_page.lower())
    lst.append(lst_word) #[[""]]
    dictionary = gensim.corpora.Dictionary(lst) # { "" :0}
    corpus = [dictionary.doc2bow(l) for l in lst] # unicode
    tf_idf = gensim.models.TfidfModel(corpus, smartirs='ntc')
    for doc in tf_idf[corpus]:
        return [[dictionary[id], round(freq, 2)] for id, freq in doc]


def indexing(dict_ind, dict_url, url_num, url, tfidf_lst):
    #modify the dict
    dict_url[url_num] = url
    for tple in tfidf_lst:
        dict_ind[tple[0]].append((url_num, tple[1]))


def sortResult(dict1):
    for i in dict1.values():
        i.sort(key=lambda x:x[1], reverse=True)


def dict_to_file(dic, filename):
    '''
    with open('indexer.json', 'r') as r:
        read = json.load(r)
        print(type(read))
    '''

    with open(f'{filename}.json', 'w') as j:
        json.dump(dic, j)


def run():
    # get all the files in the directory
    lst_files = walks_dirs(getUserInput())
    # loop through each file and get the content in it
    indexes = defaultdict(list)
    url_dict = dict()
    numDoc = 0
    for dir in lst_files:
        with open(dir, 'r') as js:
            html_content = json.load(js)
            urls = html_content['url']
            numDoc += 1
            soup = BeautifulSoup(html_content['content'], 'html.parser')
            tfidf_lst = get_content(soup)
            indexing(indexes, url_dict, numDoc, urls, tfidf_lst) # {word: [(url_id, score)]}
    sortResult(indexes)
    dict_to_file(indexes, "indexer")
    dict_to_file(url_dict, "urlTokens")
    """with open("report.txt", "w") as report:
        report.write("The number of indexed documents: "+str(numDoc)+"\n")
        report.write("The number of unique words: "+str(len(indexes))+"\n")"""


if __name__ == '__main__':
    run()
