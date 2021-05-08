# Id numbers for the team members
# 60264498,66721660,55488661,43376858

import json
from bs4 import BeautifulSoup
import os
from pathlib import Path
import glob
from collections import defaultdict
import tokenizer as tknz
import warnings
warnings.filterwarnings(action='ignore')

def getUserInput():
    d = input("Enter the diretory:")
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
    dict_score = defaultdict(int)
    dict_freq = defaultdict(int)
    for data in soup_page.find_all(["b",'strong']):
        text_page_bold = data.get_text()
        lst_word = tknz.tokenize(text_page_bold)
        for i in lst_word:
            if len(i) > 1:
                dict_score[i.lower()] += 5
                dict_freq[i.lower()] += 1
    for data in soup_page.find_all("h1"):
        text_page_h1 = data.get_text()
        lst_word = tknz.tokenize(text_page_h1)
        for i in lst_word:
            if len(i) > 1:
                dict_score[i.lower()] += 5
                dict_freq[i.lower()] += 1
    for data in soup_page.find_all("h2"):
        text_page_h2 = data.get_text()
        lst_word = tknz.tokenize(text_page_h2)
        for i in lst_word:
            if len(i) > 1:
                dict_score[i.lower()] += 5
                dict_freq[i.lower()] += 1
    for data in soup_page.find_all("h3"):
        text_page_h3 = data.get_text()
        lst_word = tknz.tokenize(text_page_h3)
        for i in lst_word:
            if len(i) > 1:
                dict_score[i.lower()] += 5
                dict_freq[i.lower()] += 1
    for data in soup_page.find_all("title"):
        text_page_title = data.get_text()
        lst_word = tknz.tokenize(text_page_title)
        for i in lst_word:
            if len(i) > 1:
                dict_score[i.lower()] += 5
                dict_freq[i.lower()] += 1
    for data in soup_page.find_all("p"):
        text_page = data.get_text()
        lst_word = tknz.tokenize(text_page)
        for i in lst_word:
            if len(i) > 1:
                dict_score[i.lower()] += 1
                dict_freq[i.lower()] += 1
    return dict_score, dict_freq # dict_score: key= Word, val= score, .....

def indexing(dict_ind, url, score, freq):
    #modify the dict
    for key in freq.keys():
        dict_ind[key].append((url, score[key], freq[key]))
    
def sortResult(dict1):
    for i in dict1.values():
        i.sort(key=lambda x:x[2], reverse=True)
        
def dict_to_file(index_dict):
    '''
    with open('indexer.json', 'r') as r:
        read = json.load(r)
        print(type(read))
    '''

    with open('indexer.json', 'w') as j:
        json.dump(index_dict, j)

def run():
    # get all the files in the directory
    lst_files = walks_dirs(getUserInput())
    # loop through each file and get the content in it
    indexes = defaultdict(list)
    numDoc = 0
    for dir in lst_files:
        with open(dir, 'r') as js:
            html_content = json.load(js)
            url = html_content['url']
            numDoc += 1
            soup = BeautifulSoup(html_content['content'], 'html.parser')
            dict_score, dict_freq = get_content(soup)
            indexing(indexes, url, dict_score, dict_freq)
    sortResult(indexes)
    dict_to_file(indexes)
    with open("report.txt", "w") as report:
        report.write("The number of indexed documents: "+str(numDoc)+"\n")
        report.write("The number of unique words: "+str(len(indexes))+"\n")

if __name__ == '__main__':
    run()