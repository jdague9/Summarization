# -*- coding: utf-8 -*-
"""Created on Wed Nov 26 15:11:30 2014

@author: JDD46
"""
import csv
import string
import math
import numpy as np
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
        self.s_vec_matrix = None
        self.co_occur_matrix = None
        self.max_tfidf = 0
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
        posts = []
        words = {}
        self.postcount, self.wordcount = 0, 0
        # Read through the CSV, tokenize and keep only words, cast to lowercase.
        for row in self.reader:
            posts.append(Post(text=row[0], index=(self.postcount)))      
            clean = (filter(lambda x: x not in (string.punctuation + 
                    string.digits), row[0])).lower().strip().split()
            currentpost = posts[self.postcount]
            # nodup keeps track of the words that are encountered while reading
            #   through each tweet and ignores the word if it is encoutered again.
            #   This is to calculate idf.
            nodup = {}
            for word in clean:
                # Call Post method to add each word in the tweet to the words_in
                #   dict for each tweet.
                currentpost.add2words_in(word)
                self.wordcount += 1
                # Increment the tf counter for each word as it is encountered.
                # Do not increment idf if we've already seen the word in the
                #   current post.
                if word in nodup:
                    words[word].addtf()  
                else:
                    # If we haven't yet seen the word in the tweet, increment tf
                    #   and idf, and add the word to nodup.
                    if word not in words:
                        words[word] = Word(name=word)
                        self.uqwordcount += 1
                    words[word].addtf()
                    words[word].addidf()
                    nodup[word] = 1
            # Run back through the words_in dict after its been completed and add
            #   word co-occurrences to the co_occur dict in each Word object.
            for key in posts[self.postcount].words_in:
                for compare in posts[self.postcount].words_in:
                    words[key].add2co_occur(compare)
            self.postcount += 1
        self.word_dict, self.posts = words, posts
            
    def calc_scores(self):
        """Calculates tf-idf weights for words, scores posts
        """
        # Go through each Word object in word_dict, calculating the tf-idf
        #   score for each from the tf and idf values stored in each object,
        #   and storing the tf-idf scores in each object. Keep track of the
        #   highest tf-idf score, so that we can normalize later.
        for item in self.word_dict:
            self.word_dict[item].calctfidf(self.postcount, self.wordcount)
            if self.word_dict[item].tfidf > self.max_tfidf:
                self.max_tfidf = self.word_dict[item].tfidf
        # Create word_list, which is simply a list filled with the Word objects
        #   in word_dict, and then sort it alphabetically.
        self.word_list = self.word_dict.values()
        self.word_list.sort(key=lambda x: x.name)
        # Now that word_list is sorted, store each word's index in each Word
        #   object.
        i = 0
        for word in self.word_list:
            word.index = i
            i += 1
        # Calculate the tf-idf score of each sentence
        for item in self.posts:
            item.calcscore(self.word_dict)
    
    def remove_rt(self):
        """Removes retweets from DataSet.
        
        Also adjusts Word and Post objects accordingly, so that the cleaned up
        set of tweets can be rescored.
        """
        i = 0
        for post in self.posts:
            if 'rt' in post.words_in:
                self.retweets.append(self.posts.pop(i))
                self.postcount -= 1
                for word in post.words_in:
                    self.word_dict[word].post_removed(post)
            else:
                i += 1
        
class Word(object):
    'Base class to store word data relevant to summarization.'
    uqcount = 0
    totalcount = 0
    
    def __init__(self, name=''):
        self.name = name
        self.index = -1
        self.co_occur = {}
        self.tf = 0
        self.idf = 0
        self.tfidf = 0
        Word.uqcount += 1
        Word.totalcount += 1
        
    def addtf(self):
        self.tf += 1
        Word.totalcount += 1
        
    def addidf(self):
        self.idf += 1
        
    def add2co_occur(self, w):
        self.co_occur[w] = self.co_occur.get(w, 0) + 1
        
    def post_removed(self, post):
        self.tf -= post.words_in[self.name]
        self.idf -= 1
        for word in post.words_in:
            self.co_occur[word] -= 1
        
    def calctfidf(self, postcount, wordcount):
        self.tfidf = (float(self.tf) / float(wordcount)) * math.log(
        (float(postcount) / float(self.idf)), 2)
        
        
class Post(object):
    'Base class to store Twitter post data relevant to summarization.'
    count = 0
    
    def __init__(self, text='', index=-1):
        self.text = text
        self.index = index
        self.score = 0
        self.words_in = {}
        self.s_vec = []
        self.sim = []
        self.is_rt = False
        Post.count += 1
        
    def add2words_in(self, w):
        self.words_in[w] = self.words_in.get(w, 0) + 1
        
    def calcscore(self, word_dict):
        weightsum = 0
        length = 0
        self.s_vec = [0] * len(word_dict)
        for item in self.words_in:
            weightsum += word_dict[item].tfidf * self.words_in[item]
            length += self.words_in[item]
            self.s_vec[word_dict[item].index] = self.words_in[item]
        self.score = weightsum / max(10, length)
        
        
        
        
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