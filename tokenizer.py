import re

# ---------------------------------------- Tokenizer ----------------------------------------
def tokenize(string): # Take a string and split them in to token list
    list_token = []
    # The complexity class for this function is O(len(lines)) (Linear)
    # create tokens
    #print('breaking the text into tokens')
    list_token += re.split(pattern=r'\W+', string=string)
    return list_token

def computeWrodFrequencies(lst_token):
    dict_token = {}
    # The complexist class for this function is O(len(lst_token)) (Linear)
    # compute the frequencies of each token
    for tk in lst_token:
        if tk != '':
            if tk.lower() in dict_token:
                dict_token[tk.lower()] += 1
            else:
                dict_token[tk.lower()] = 1
    return dict_token

def print_info(dict_token):
    # The complexity class for this function is O(lst_token.unique) (Linear)
    uniq_tk = sorted(list(dict_token.keys()), key= lambda x: dict_token[x], reverse=True)
    for tk in uniq_tk:
        print('   ', tk, '->', dict_token[tk])

def run_tokenize(string):  # return a dict of tokens with key being the token, value being the freq
    list_token = tokenize(string)
    dict_token = computeWrodFrequencies(list_token)
    #print_info(dict_token)
    return dict_token
# ---------------------------------------- Tokenizer ----------------------------------------

if __name__ == "__main__":
    string = input()
    run_tokenize(string)