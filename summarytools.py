# -*- coding: utf-8 -*-
"""Created on Wed Nov 26 15:11:30 2014

@author: JDD46
"""
import csv
import string
import math
import numpy as np
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
        self.svec_matrix = None
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
        # posts and words will hold the Post and Word objects created while
        # extracting data.
        self.posts = []
        self.word_dict = {}
        self.postcount, self.wordcount = 0, 0
        # Read through the CSV, tokenize and keep only words, cast to lowercase.
        for row in self.reader:
            self.posts.append(Post(text=row[0]))
            tokens = nltk.word_tokenize(row[0].lower().strip())
            tagged = nltk.pos_tag(tokens)
            clean = filter(lambda x: x[0][0] not in (
                    string.punctuation + string.digits), tagged)
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
        
        length = len(self.word_dict)
        for post in self.posts:
            post.create_svec(self.word_dict, length)
            
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
            if post.score > self.max_post_score:
                self.max_post_score = post.score
        self.posts.sort(key=lambda x: x.score, reverse=True)
        
    def create_svec_matrix(self):
        """Creates a matrix from the collection of sentence vectors.
        """
        self.svec_matrix = np.zeros((len(self.word_dict), len(self.posts)), 
                                    dtype=float)
        i = 0
        for post in self.posts:
            for word in post.words_in:
                self.svec_matrix[self.word_dict[word].index, i] = post.words_in[word]
            i += 1
    
    def remove_rt(self):
        """Removes retweets from DataSet.
        
        Also adjusts Word and Post objects accordingly, so that the cleaned up
        set of tweets can be rescored.
        """
        i = 0
        for post in self.posts:
            if ('rt', 'NN') in post.words_in:
                self.retweets.append(self.posts.pop(i))
                post.remove(self.word_dict)
                self.postcount -= 1
                self.wordcount -= post.wordcount
            else:
                i += 1
                
        
                
        self.calc_scores()
                
        
        
        
            
#        for post in self.posts:
        
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
        
#    def addtf(self, num=1):
#        if not self.is_stopword:
#            self.tf += num
#            
#    def subtf(self, num=1):
#        if not self.is_stopword:
#            self.tf -= num   
#        
#    def addidf(self):
#        if not self.is_stopword:
#            self.idf += 1
#        
#    def subidf(self):
#        if not self.is_stopword:
#            self.idf -= 1
#        
#    def addco_occur(self, w):
#        if not self.is_stopword:
#            self.co_occur[w] = self.co_occur.get(w, 0) + 1
#        
#    def post_added(self, words_in_dict):
#        self.addtf(words_in_dict[self.name])
#        self.addidf()
#        for word in words_in_dict:
#            self.addco_occur
#    
#    def post_removed(self, words_in_dict):
#        self.tf -= words_in_dict[self.name]
#        self.idf -= 1
#        for word in words_in_dict:
#            self.co_occur[word] -= 1
        
class Post(object):
    'Base class to store Twitter post data relevant to summarization.'
    
    def __init__(self, text=''):
        self.text = text
        self.index = -1
        self.score = 0
        self.wordcount = 0
        self.words_in = {}
        self.s_vec = []
        self.sim = []
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
            
    def create_svec(self, word_dict, length):
        self.s_vec = [0] * length
        for word in self.words_in:
            self.s_vec[word_dict[word].index] = self.words_in[word]
                
    def calcscore(self, word_dict):
        weightsum = 0
        for item in self.words_in:
            weightsum += word_dict[item].tfidf * self.words_in[item]
        self.score = weightsum / max(7, self.wordcount)
        
        
        
        
#def csv_prep_tweet(in_fname, out_fname, col):
#    """Returns a CSV reader object that reads from the file "out_fname".
#    
#    First creates reader object for file "in_fname", then creates a writer
#    object that writes to file "out_fname". Only the data specified in column
#    col is written to the new file. Lastly, deletes the reader and writer that
#    were created by the method, creates a new reader for "out_fname", and
#    returns that reader.
#    """
#    # Open passed file name.
#    try:
#        rawcsvfile = open(in_fname, 'rU')
#    except : 
#        print 'Error: Not a valid file name. Try again.'
#    else:
#        # Create CSV reader object to read from passed CSV file. 
#        rawreader = csv.reader(rawcsvfile, dialect='excel')
#        # Create write file and CSV writer object for it.
#        with open(out_fname, 'wb') as f:
#            writer = csv.writer(f, dialect='excel')
#            # Write all entries contained in the specified column to the file.
#            for row in rawreader:
#                if len(row) < 2 or len(row[col]) > 250:
#                    continue
#                # Remove non-ASCII characters from tweet before writing.
#                tweet = [filter(lambda x: x in string.printable, row[col])]
#                writer.writerow(tweet)
#        rawcsvfile.close()
#        del rawreader, writer
#        f = open(out_fname, 'rU')
#        cleanreader = csv.reader(f, dialect='excel')
#        return cleanreader
                
#def extract_data(csv_reader):
#    """Extracts data from passed csv_reader object. Creates Post and Word 
#    objects as they are observed.
#    
#    """
#    # posts and words will hold the Post and Word objects created while
#    # extracting data.
#    posts = []
#    words = {}
#    Post.count, Word.totalcount, Word.uqcount = 0, 0, 0
#    # Read through the CSV, tokenize and keep only words, cast to lowercase.
#    for row in csv_reader:
#        posts.append(Post(text=row[0], index=(Post.count)))      
#        clean = (filter(lambda x: x not in (string.punctuation + 
#                string.digits), row[0])).lower().strip().split()
#        currentpost = posts[Post.count - 1]
#        # nodup keeps track of the words that are encountered while reading
#        #   through each tweet and ignores the word if it is encoutered again.
#        #   This is to calculate idf.
#        nodup = {}
#        for word in clean:
#            # Call Post method to add each word in the tweet to the words_in
#            #   dict for each tweet.
#            currentpost.add2words_in(word)
#            # Increment the tf counter for each word as it is encountered.
#            # Do not increment idf if we've already seen the word in the
#            #   current post.
#            if word in nodup:
#                words[word].addtf()  
#            else:
#                # If we haven't yet seen the word in the tweet, increment tf
#                #   and idf, and add the word to nodup.
#                if word not in words:
#                    words[word] = Word(name=word, tf=0)
#                words[word].addtf()
#                words[word].addidf()
#                nodup[word] = 1
#        # Run back through the words_in dict after its been completed and add
#        #   word co-occurrences to the co_occur dict in each Word object.
#        for key in posts[Post.count - 1].words_in:
#            for compare in posts[Post.count - 1].words_in:
#                words[key].add2co_occur(compare)
#    return posts, words
#        
#def calc_scores(posts, words):
#    """Calculates tf-idf weights for words, scores posts
#    """
#    maxtfidf = 0
#    for item in words:
#        words[item].calctfidf()
#        if words[item].tfidf > maxtfidf:
#            maxtfidf = words[item].tfidf
#               
#    for item in posts:
#        item.calcscore(words)
#        
#    return posts, words