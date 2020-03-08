# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:35:51 2019

@author: afrodrigues
"""

import search
import unittest
import pandas as pd
import pickle

from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer


class TestSearch(unittest.TestCase):
    
    def test_words_position(self):
        
        dict_test = search.words_positions(['i', 'dont', 'like', 'chocolate', 'because', 'i', 'like', 'to', 'be', 'healthy'])
        
        self.assertEqual(dict_test['like'][0], 2)
        self.assertEqual(dict_test['like'][1], 6)
        self.assertEqual(dict_test['be'][0], 8)
        
        
    def test_article_words(self):
        
        dict_test = {1: ['i', 'dont', 'like', 'chocolate', 'because', 'i', 'like', 'to', 'be', 'healthy'],
                     2: ['france', 'is', 'country', 'where', 'i', 'like', 'eat', 'chocolate', 'is', 'good']}
        
        dict_doc = search.article_words(dict_test)
        
        self.assertEqual(len(dict_doc[1]), 8)
        self.assertEqual(dict_doc[1]['like'][1], 6)
        self.assertEqual(len(dict_doc), 2)
        
    def test_changeto_words_article(self):
        
        dict_test = {1: ['i', 'dont', 'like', 'chocolate', 'because', 'i', 'like', 'to', 'be', 'healthy'],
                     2: ['france', 'is', 'country', 'where', 'i', 'like', 'eat', 'chocolate', 'is', 'good']}
        
        dict_doc = search.article_words(dict_test)
        
        inverseindex = search.changeto_words_article(dict_doc)
        
        self.assertTrue('healthy' in inverseindex.keys())
        self.assertEqual(len(inverseindex['chocolate']), 2)
        self.assertEqual(len(inverseindex['i'][1]), 2)
        self.assertEqual(inverseindex['i'][1][1], 5)
        
    def test_query_by_word(self):
        
        dict_test = {1: ['i', 'dont', 'like', 'chocolate', 'because', 'i', 'like', 'to', 'be', 'healthy'],
                     2: ['france', 'is', 'country', 'where', 'i', 'like', 'eat', 'chocolate', 'is', 'good']}
        
        dict_doc = search.article_words(dict_test)
        
        inverseindex = search.changeto_words_article(dict_doc)
        
        dict_search_results = search.query_by_word('france', inverseindex)
        
        self.assertEqual(len(dict_search_results), 1)
        self.assertEqual(dict_search_results[0], 2)
        
        
    def test_query_by_phrase(self):
        
        dict_test = {1: ['i', 'dont', 'like', 'chocolate', 'because', 'i', 'like', 'to', 'be', 'healthy'],
                     2: ['france', 'is', 'country', 'where', 'i', 'like', 'eat', 'chocolate', 'is', 'good']}
        
        dict_doc = search.article_words(dict_test)
        
        inverseindex = search.changeto_words_article(dict_doc)
        
        query_1 = [['i', 'france']]
        query_2 = [['i', 'like']]
        query_3 = [['because']]
        
        result_1 = search.query_by_phrase(query_1[0], inverseindex)
        result_2 = search.query_by_phrase(query_2[0], inverseindex)
        result_3 = search.query_by_phrase(query_3[0], inverseindex)
        
        
        self.assertEqual(len(result_1), 1)
        self.assertEqual(result_1[0], 2)
        self.assertEqual(len(result_2), 2)
        self.assertEqual(result_2[0], 1)
        self.assertEqual(len(result_3), 1)
        self.assertEqual(result_3[0], 1)
        
        
    def test_number_results(self):
        
        
            #LOAD PICKLE
        with open('dataframe_search_query.pickle', 'rb') as handle:
            dataframe = pickle.load(handle)
        
        tokenizer = WordPunctTokenizer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        regex_list = [(r'[^\w\s]', "")]
        cleaner_join = search.TextCleanerTransformer(tokenizer, stemmer, regex_list, join_words=True)
        cleaner = search.TextCleanerTransformer(tokenizer, stemmer, regex_list, join_words=False)
        

        data_1 = pd.read_csv('articles1.csv')
        
        
        final_results = search.number_results(dataframe, clean=cleaner, clean_join=cleaner_join, data=data_1)
        
        
        self.assertEqual(final_results.shape[0], 3)
        self.assertEqual(final_results.shape[1], 11)
        self.assertEqual(final_results.id.values[0], 57630)
        self.assertEqual(final_results.id.values[2], 71034)
        
        

    
        
if __name__ == '__main__':
    unittest.main()



