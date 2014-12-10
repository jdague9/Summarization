# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 18:16:09 2014

@author: JDD46
"""

from summarytools import *
from datetime import datetime
import cPickle as pickle
startTime = datetime.now()
test = DataSet(setname='test', pos_tag=1)
test.extract_from('test.csv', 1)
test.extract_data()
pkl_fname = test.setname + '.pkl'
with open(pkl_fname, 'wb') as f:
    pickler = pickle.Pickler(f, -1)
    pickler.dump(test)
#test.calc_scores()
##for post in test.posts[:19]:
##    print post.score, post.text    
#test.remove_rt()
##for post in test.posts[:19]:
##    print post.score, post.text
    
print 'Elapsed time:', (datetime.now()-startTime)
