# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 18:16:09 2014

@author: JDD46
"""

from summarytools import *
from datetime import datetime
startTime = datetime.now()
test = DataSet(setname='test')
test.extract_from('test.csv', 1)
test.extract_data()
test.calc_scores()
test.calc_similarity()
#for post in test.posts[:19]:
#    print post.score, post.text    
test.remove_rt()
#for post in test.posts[:19]:
#    print post.score, post.text
test.calc_scores()
    
print 'Elapsed time:', (datetime.now()-startTime)
