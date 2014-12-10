# -*- coding: utf-8 -*-
"""Created on Wed Nov 26 15:11:30 2014

@author: JDD46
"""
import csv
import string
import math
import numpy as np
import scipy.spatial.distance as dist
import nltk
import re
from nltk.corpus import stopwords
from operator import attrgetter as ag

class DataSet(object):
    'Base class to store data representations for a particular dataset.'
    
    def __init__(self, setname=''):
        self.setname = setname
        self.word_dict = {}
        self.word_list = []
        self.posts = []
        self.retweets = []
        self.tfidfs = []
        self.scores = []
        self.cosines = None
        self.co_occur_matrix = None
        self.max_word_tfidf = 0
        self.max_post_score = 0
        self.reader = None
        self.has_data = False # Include checks to see if data has already been read.
        self.postcount = 0
        self.wordcount = 0
        self.uqwordcount = 0
        
    def extract_from(self, in_fname, col):
        """Returns a CSV reader object that reads from the file "out_fname".
        
        First creates reader object for file "in_fname", then creates a writer
        object that writes to file "out_fname". Only the data specified in 
        columnncol is written to the new file. Lastly, deletes the reader and 
        writer that were created by the method, creates a new reader for
        "out_fname", and returns that reader.
        """
        # Open passed file name.
        try:
            rawcsvfile = open(in_fname, 'rU')
        except : 
            print 'Error: Not a valid file name. Try again.'
        else:
            out_fname = in_fname[:-4] + '_clean.csv'
            # Create CSV reader object to read from passed CSV file. 
            rawreader = csv.reader(rawcsvfile, dialect='excel')
            # Create write file and CSV writer object for it.
            with open(out_fname, 'wb') as f:
                writer = csv.writer(f, dialect='excel')
                # Write all entries contained in specified column to the file.
                for row in rawreader:
                    if len(row) < 2 or len(row[col]) > 250:
                        continue
                    # Remove non-ASCII characters from tweet before writing.
                    tweet = [filter(lambda x: x in string.printable, row[col])]
                    writer.writerow(tweet)
            rawcsvfile.close()
            del rawreader, writer
            f = open(out_fname, 'rU')
            self.reader = csv.reader(f, dialect='excel')
            
    def extract_data(self):
        """Extracts data from passed csv_reader object. Creates Post and Word 
        objects as they are observed.
        
        """
        # Read through the CSV, tokenize and keep only words, cast to lowercase.
        for row in self.reader:
            self.posts.append(Post(text=row[0]))
#            tokens = nltk.word_tokenize(row[0].lower().strip())
#            tagged = nltk.pos_tag(tokens)
#            clean = filter(lambda x: x[0][0] not in (
#                    string.punctuation + string.digits), tagged)
            currentpost = self.posts[self.postcount]
            for word in clean:
                if word not in self.word_dict:
                    self.word_dict[word] = Word(name=word[0], pos = word[1])
                currentpost.words_in[word] = currentpost.words_in.get(word, 0) + 1
            currentpost.add(self.word_dict)
            self.wordcount += currentpost.wordcount
            self.postcount += 1
            
        # Create word_list, which is simply a list filled with the Word objects
        #   in word_dict, and then sort it alphabetically.
        self.word_list = self.word_dict.values()
        self.word_list.sort(key=lambda x: x.name)
        # Now that word_list is sorted, store each word's index in each Word
        #   object. These indeces are used for constructing post vectors.
        i = 0
        for word in self.word_list:
            word.index = i
            i += 1
            
    def calc_scores(self):
        """Calculates tf-idf weights for words, scores posts
        """
        # Go through each Word object in word_dict, calculating the tf-idf
        #   score for each from the tf and idf values stored in each object,
        #   and storing the tf-idf scores in each object. Keep track of the
        #   highest tf-idf score, so that we can normalize later.
        self.tfidfs = []
        self.scores = []
        for word in self.word_list:
            word.calctfidf(self.postcount, self.wordcount)
            self.tfidfs.append(word.tfidf)
            if word.tfidf > self.max_word_tfidf:
                self.max_word_tfidf = word.tfidf
        # Calculate the tf-idf score of each sentence
        for post in self.posts:
            post.calcscore(self.word_dict)
            self.scores.append(post.score)
            post.word_rank = post.words_in.keys()
            i = 0
            for word in post.word_rank:
                post.word_rank[i] = self.word_dict[word]
                i += 1
            post.word_rank.sort(key=lambda x: x.tfidf, reverse=True)
            if post.score > self.max_post_score:
                self.max_post_score = post.score
            
        self.posts.sort(key=lambda x: x.score, reverse=True)
        
    def calc_similarity(self):
        """Calculates the pairwise cosine similarity of each sentence, using
        the sentence vector representation.
        """
        self.posts.sort(key=lambda x: x.score, reverse=True)
        svec_matrix = np.zeros((len(self.posts), len(self.word_dict)), 
                                    dtype=float)
        i = 0        
        for post in self.posts:
            for word in post.words_in:
                svec_matrix[i, self.word_dict[word].index] = post.words_in[word]
            i += 1    
        self.cosines = np.triu(dist.squareform(1 - dist.pdist(svec_matrix, 'cosine')))        
    
    def remove_rt(self):
        """Removes retweets from DataSet.
        
        Also adjusts Word and Post objects accordingly, so that the cleaned up
        set of tweets can be rescored.
        """
        coords = zip(np.asarray(np.where(self.cosines >= 0.9)[0], dtype=int).tolist(), 
                     np.asarray(np.where(self.cosines >= 0.9)[1], dtype=int).tolist())
        to_rt = {}           
        for coord in coords:
            if coord[1] not in to_rt:
                to_rt[coord[1]] = None
                self.posts[coord[0]].retweets += 1
        to_rt = to_rt.keys()
        to_rt.sort(reverse=True)
        for i in to_rt:
            self.posts[i].remove(self.word_dict)
            self.postcount -= 1
            self.wordcount -= self.posts[i].wordcount
            self.retweets.append(self.posts.pop(i))               
        self.calc_scores()
        
class Word(object):
    'Base class to store word data relevant to summarization.'
    
    def __init__(self, name='', pos='X'):
        self.name = name
        self.pos = pos
        self.is_stopword = False
        self.index = -1
        self.co_occur = {}
        self.tf = 0
        self.idf = 0
        self.tfidf = 0
        
        if self.name in stopwords.words('english'):
            self.is_stopword = True
            self.tf, self.idf, self.tfidf = 0, 0, 0
            
    def calctfidf(self, postcount, wordcount):
        if self.idf != 0:    
            self.tfidf = (float(self.tf) / float(wordcount)) * math.log(
            (float(postcount) / float(self.idf)), 2)
        
class Post(object):
    'Base class to store Twitter post data relevant to summarization.'
    
    def __init__(self, text=''):
        self.text = text
        self.index = -1
        self.score = 0
        self.wordcount = 0
        self.retweets = 0
        self.words_in = {}
        self.word_rank = []
        self.is_junk = True
        
    def add(self, word_dict):
        if self.is_junk:
            self.is_junk = False
            for word in self.words_in:
                if not word_dict[word].is_stopword:
                    word_dict[word].tf += self.words_in[word]
                    self.wordcount += self.words_in[word]
                    word_dict[word].idf += 1
                    for compare in self.words_in:
                        if not word_dict[compare].is_stopword:
                            word_dict[word].co_occur[compare] = word_dict[word
                            ].co_occur.get(compare, 0) + 1
    
    def remove(self, word_dict):
        if not self.is_junk:
            self.is_junk = True
            self.wordcount = 0
            for word in self.words_in:
                if not word_dict[word].is_stopword:
                    word_dict[word].tf -= self.words_in[word]
                    word_dict[word].idf -= 1
                    for compare in self.words_in:
                        if not word_dict[compare].is_stopword:
                            word_dict[word].co_occur[compare] -= 1
            self.index = -1
            self.score = 0
                
    def calcscore(self, word_dict):
        weightsum = 0
        for item in self.words_in:
            weightsum += word_dict[item].tfidf * self.words_in[item]
        self.score = weightsum / max(7, self.wordcount)