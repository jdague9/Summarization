# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 18:16:09 2014

@author: JDD46
"""

from summarytools import *
from datetime import datetime
startTime = datetime.now()
test = DataSet(setname='Coke', pos_tag=1)
test.extract_from('Coke Twitter Posts.csv', 1)
test.extract_data()
test.calc_scores() 
test.remove_rt()
    
print 'Elapsed time:', (datetime.now()-startTime)
