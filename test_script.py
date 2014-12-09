# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 18:16:09 2014

@author: JDD46
"""

from summarytools import *

test = DataSet(setname='test')
test.extract_from('test.csv', 1)
test.extract_data()
test.calc_scores()

print '# Posts:', test.postcount
print test.posts[15].text
for word in test.posts[15].words_in:
    print word, 'tf:', test.word_dict[word].tf, 'idf:', test.word_dict[word].idf

i = 0
for post in test.posts:
    print i, post.score
    i += 1
    
hold = test.word_dict['water', 'NN'].co_occur.items()
    
test.remove_rt()

print 'retweets:', test.retweets

print '# Posts:', test.postcount
print test.retweets[0].text
for word in test.retweets[0].words_in:
    print word, 'tf:', test.word_dict[word].tf, 'idf:', test.word_dict[word].idf
i = 0
for post in test.posts:
    print i, post.score
    i += 1
    
print 'water co_occur'
print hold
print test.word_dict['water', 'NN'].co_occur