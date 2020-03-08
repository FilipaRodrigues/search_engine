# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:59:53 2019

@author: afrodrigues
"""

# Standard imports
import numpy as np
import pandas as pd
import re
import string
import warnings; warnings.simplefilter('ignore')
import pickle

# NLTK imports
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer

# SKLearn related imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import linear_kernel


# Custom transformer to implement sentence cleaning
class TextCleanerTransformer(TransformerMixin):
    def __init__(self, tokenizer, stemmer, regex_list,
                 lower=True, remove_punct=True, join_words=False):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.regex_list = regex_list
        self.lower = lower
        self.remove_punct = remove_punct
        self.join_words = join_words
        
    def transform(self, X, *_):
        X = list(map(self._clean_sentence, X))
        return X
    
    def _clean_sentence(self, sentence):
        
        # Replace given regexes
        for regex in self.regex_list:
            sentence = re.sub(regex[0], regex[1], sentence)
            
        # lowercase
        if self.lower:
            sentence = sentence.lower()

        # Split sentence into list of words
        words = self.tokenizer.tokenize(sentence)
            
        # Remove punctuation
        if self.remove_punct:
            words = list(filter(lambda x: x not in string.punctuation, words))

        # Stem words
        if self.stemmer:
            words = list(map(self.stemmer.stem, words))
            
        if self.join_words == True:
            
            # Join list elements into string
            words = " ".join(words)
                
        return words
    
    def fit(self, *_):
        return self
    

#Function to run for each article that relates words to positions
def words_positions(article):
	dict_words = {}
	for index, word in enumerate(article):
		if word in dict_words.keys():
			dict_words[word].append(index)
		else:
			dict_words[word] = [index]
	return dict_words

#Function that relates id of articles to words (ans subsequently, to positions)
def article_words(list_article):
	dict_doc = {}
	for article in list_article.keys():
		dict_doc[article] = words_positions(list_article[article])
	return dict_doc

#Function that changes the order of the dict (instead of document ->words -> position -----> word->document->position)
def changeto_words_article(article_to_words):
	index_final = {}
	for article in article_to_words.keys():
		for word in article_to_words[article].keys():
			if word in index_final.keys():
				if article in index_final[word].keys():
					index_final[word][article].extend(article_to_words[article][word][:])
				else:
					index_final[word][article] = article_to_words[article][word]
			else:
				index_final[word] = {article: article_to_words[article][word]}
	return index_final

#Function that searches for each word in the search query defined
def query_by_word(word, inverted_index):
	if word in inverted_index.keys():
		return [article for article in inverted_index[word].keys()]
	else:
		return []

#Function that searches a query with more than one word
def query_by_phrase(string, inverted_index):
    
    intersect=None
    result = []
    
    for word in string:
        result = query_by_word(word, inverted_index)
        if intersect != None:
            final_list = [value for value in result if value in intersect] 
            intersect = final_list
        else:
            intersect = result
            
        result=[]
        
    return list(set(intersect))


#This function is to create the invertedindex (because it takes too long, we created it already and pickled it)
def initialize_process():
    
    docs = cleaner.transform(data_1.content)

    index_doc = data_1.id.values
    text_doc = docs

    my=dict(zip(index_doc, text_doc))
    
    aa = article_words(my)
    
    bb = changeto_words_article(aa)
    
    return bb


#Function thar has as input the dataframe (output of function initialize_process) and a query produced by the user
#and ouputs the 20 most relevant articles, ranked and also the number of results that match the query
def number_results(dataframe_, clean, clean_join, data):
    
    query = str(input("Enter the search query: "))
    query_transformed = clean.transform([query])
       
    y = query_by_phrase(query_transformed[0], dataframe_)
    
    df=pd.DataFrame()
    
    for doc in y:
        serie = data[data.id == doc]
        df = df.append(serie)
        
    print( "We found {} results that match your query.".format(df.shape[0]))

    
    #For the results
    df = df.set_index(df.id)
    results_clean = clean_join.transform(df.content.values)
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(results_clean)
    
    word_count_matrix = vectorizer.transform(results_clean)
    
    tfidf = TfidfTransformer()
    tfidf.fit(word_count_matrix)

    word_term_frequency_matrix = tfidf.transform(word_count_matrix)

    #For the query
    query_clean = clean_join.transform([query])
    
    query_count_matrix = vectorizer.transform(query_clean)
    
    query_term_frequency_matrix = tfidf.transform(query_count_matrix)
    
    #Find similarities
    
    cosine_similarities = linear_kernel(query_term_frequency_matrix, word_term_frequency_matrix).flatten()
    
    df['similar'] = cosine_similarities
    
    df = df.sort_values(by=['similar'])[:-21:-1]
    
    index_ranked = np.arange(1, df.shape[0]+1, 1)
    
    results_ranked = df.set_index(index_ranked)
    
    #print(results_ranked[['id', 'title']])
        
    return results_ranked




if __name__ == '__main__':
    #FIRST STEP
    data_1 = pd.read_csv('articles1.csv')

    tokenizer = WordPunctTokenizer()
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    regex_list = [(r'[^\w\s]', "")]
    cleaner_join = TextCleanerTransformer(tokenizer, stemmer, regex_list, join_words=True)
    cleaner = TextCleanerTransformer(tokenizer, stemmer, regex_list, join_words=False)

    #dataframe = initialize_process()
    #PICKLE 
    #with open('dataframe_search_query.pickle', 'wb') as handle:
    #    pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #LOAD PICKLE
    with open('dataframe_search_query.pickle', 'rb') as handle:
        dataframe = pickle.load(handle)

    ranked_results = number_results(dataframe, clean=cleaner, clean_join=cleaner_join, data=data_1)
    