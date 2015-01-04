
####################
###    Part A    ###
####################

########################
###   Question A1    ###
########################

import nltk
import pandas as pd 
import collections 
import random

# Reading Data
JD_Data = pd.read_csv('C:\Users\Neerav Basant\Documents\GitHub\Text-Analysis\Homework 1\Train_rev1.csv')

# Creating new dataset by randomly extracting 5000 rows from the JD_Data dataset 
random.seed(1000)
rows = random.sample(JD_Data.index, 5000)
JD_Data_5000 = JD_Data.ix[rows]

#	Tokenizing the Job Description for new dataset 
f1 = lambda x: nltk.word_tokenize(x) 
tokens = JD_Data_5000['FullDescription'].apply(f1) 

#	Finding Parts of Speech for the tokens 
f2 = lambda x: nltk.pos_tag(x)
tagged = tokens.apply(f2)

#	Creating dataframes from the series of parts of speech 
tagged_df = pd.DataFrame(tagged, columns=['FullDescription']) 

#	Appending parts of speech of all the job descriptions in one list 
tag_unique = [] 
for x in tagged_df['FullDescription']: 
    tag_unique.extend(x) 
tag_unique 

#	Getting count of 5 most common parts of speech 
pos_count = collections.Counter([j for i,j in tag_unique]).most_common(5)
pos_count


########################
###   Question A2    ###
########################

import nltk
import pandas as pd 
import collections 
import random
from nltk import FreqDist 
import matplotlib
import matplotlib.pyplot as plt

# Reading Data
JD_Data = pd.read_csv('C:\Users\Neerav Basant\Documents\GitHub\Text-Analysis\Homework 1\Train_rev1.csv')

#	Creating new dataset by randomly extracting 5000 rows from the JD_Data dataset 
random.seed(1000) 
rows = random.sample(JD_Data.index, 5000) 
JD_Data_5000 = JD_Data.ix[rows] 

#	Tokenizing the Job Description for new dataset 
f1 = lambda x: nltk.word_tokenize(x)
tokens = JD_Data_5000['FullDescription'].apply(f1)

#	Creating dataframes from the series of parts of speech 
tagged_df = pd.DataFrame(tokens, columns=['FullDescription']) 

#	Appending parts of speech of all the job descriptions in one list 
tag_unique = [] 
for x in tagged_df['FullDescription']: 
    tag_unique.extend(j.lower() for j in x) 

cnt = collections.Counter() 
for tag in tag_unique: 
    cnt[tag] += 1

cnt_100 = cnt.most_common(100)

ranks = [] 
freqs = []

#	Generate a (rank, frequency) point for each counted token and 
#	and append to the respective lists, Note that the iteration 
#	over fd is automatically sorted. 

for rank, word in enumerate(cnt_100): 
    ranks.append(rank+1) 
    freqs.append(word[1])

plt.loglog(ranks, freqs)
plt.ylabel('frequency(f)', fontsize=14, fontweight='bold') 
plt.xlabel('rank(r)', fontsize=14, fontweight='bold') 
plt.grid(True)
plt.show()
 

########################
###   Question A3    ###
########################

import nltk
from nltk.corpus import stopwords as s1 
import pandas as pd
import collections 
import random 
import string

# Reading Data
JD_Data = pd.read_csv('C:\Users\Neerav Basant\Documents\GitHub\Text-Analysis\Homework 1\Train_rev1.csv')

#	Creating new dataset by randomly extracting 5000 rows from the JD_Data dataset 
random.seed(100) 
rows = random.sample(JD_Data.index, 5000) 
JD_Data_5000 = JD_Data.ix[rows] 

#	Tokenizing the Job Description for new dataset 
f1 = lambda x: nltk.word_tokenize(x)
tokens = JD_Data_5000['FullDescription'].apply(f1)
 
#	Creating dataframes from the series of parts of speech 
tagged_df = pd.DataFrame(tokens, columns=['FullDescription'])
 
#	Appending parts of speech of all the job descriptions in one list 
tag_unique1 = [] 
english_stops = set(s1.words('english')) 
for x in tagged_df['FullDescription']: 
    for y in x: 
        if '*' not in y and y not in string.punctuation and y.lower() not in english_stops:
            tag_unique1.append(y) 


from nltk.stem import WordNetLemmatizer 
wnl = WordNetLemmatizer()

tag_unique2 = []
for z in tag_unique1: 
    w=wnl.lemmatize(z) 
    tag_unique2.append(w) 
    cnt = collections.Counter() 
    for tag in tag_unique2: 
        cnt[tag] += 1

cnt_10 = cnt.most_common(10) 
cnt_10



####################
###    Part B    ###
####################

########################
###   Question B1    ###
########################

import nltk
import pandas as pd 
import numpy as np 
import random


# Reading Data
JD_Data = pd.read_csv('C:\Users\Neerav Basant\Documents\GitHub\Text-Analysis\Homework 1\Train_rev1.csv') 

# Creating new dataset by randomly extracting 5000 rows from the JD_Data dataset 
random.seed(1000)
rows = random.sample(JD_Data.index, 5000)
JD_Data_5000 = JD_Data.ix[rows]
 
# Creating the flag for high and salary
p = np.percentile(JD_Data_5000['SalaryNormalized'], 75)
JD_Data_5000['SalaryFlag'] = None
s = JD_Data_5000['SalaryFlag']

for x in JD_Data_5000['SalaryNormalized']: 
    if x >= p: 
        s[JD_Data_5000['SalaryNormalized'] == x] = 1 
    else:
        s[JD_Data_5000['SalaryNormalized'] == x] = 0

# Keeping only the required variables and Creating training and test dataset 
JD_Data_5000 = JD_Data_5000[['FullDescription','SalaryFlag']]

from sklearn import cross_validation

train, test = cross_validation.train_test_split(JD_Data_5000, train_size=0.6, test_size=0.4)
train_df = pd.DataFrame(train, columns=['FullDescription', 'SalaryFlag']) 
test_df = pd.DataFrame(test, columns=['FullDescription', 'SalaryFlag'])

def getAllTokens(data, lemmatize, removeStopWords): 
    t = []
    stpWords = nltk.corpus.stopwords.words('english') 
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    
    for index, row in data.iterrows():
        jobDesc = row['FullDescription'] 
        tkns = nltk.word_tokenize(jobDesc) 
        if removeStopWords:
            trimTkns = [w for w in tkns if w.lower() not in stpWords] 
            tkns = trimTkns[:]
        new_dict = dict() 
        for tk in tkns:
            if lemmatize:
                tr = wnl.lemmatize(tk.lower()) 
                tk = tr[:]
            new_dict[tk.lower()] = new_dict.get(tk.lower(), 0) + 1 
        t.append((new_dict, data['SalaryFlag'][index]))
    return t

#all_words = set(word.lower() for passage in train for word in nltk.word_tokenize(passage[0])) 
#t = [({word: (word in nltk.word_tokenize(x[0])) for word in all_words}, x[1]) for x in train] 

train_feature = getAllTokens(train_df, False, False)
test_feature = getAllTokens(test_df, False, False)

classifier = nltk.NaiveBayesClassifier.train(train_feature)
print(nltk.classify.accuracy(classifier, test_feature))

# Creating confusion matrix 
ref_b1 = [x[1] for x in test]
test_b1 = [classifier.classify(x[0]) for x in test_feature]
cm = nltk.ConfusionMatrix(ref_b1, test_b1) 
print cm


########################
###   Question B2    ###
########################

#	Creating Feature Set for training and test dataset using getAllTokens function from B1 
train_feature_lem = getAllTokens(train_df, True, False) 
test_feature_lem = getAllTokens(test_df, True, False) 
classifier = nltk.NaiveBayesClassifier.train(train_feature_lem) 
print(nltk.classify.accuracy(classifier, test_feature_lem))

#	Creating confusion matrix 
ref_b2 = [x[1] for x in test]
test_b2 = [classifier.classify(x[0]) for x in test_feature_lem]
cm_b2 = nltk.ConfusionMatrix(ref_b2, test_b2) 
print cm_b2


########################
###   Question B3    ###
########################

#	Creating Feature Set for training and test dataset using getAllTokens function from B1 
train_feature_lem_stop = getAllTokens(train_df, True, True) 
test_feature_lem_stop = getAllTokens(test_df, True, True) 
classifier = nltk.NaiveBayesClassifier.train(train_feature_lem_stop) 
print(nltk.classify.accuracy(classifier, test_feature_lem_stop)) 

#	Creating confusion matrix 
ref_b3 = [x[1] for x in test]
test_b3 = [classifier.classify(x[0]) for x in test_feature_lem_stop]
cm_b3 = nltk.ConfusionMatrix(ref_b3, test_b3) 
print cm_b3

def most_informative_features(classifier, n, label1, label2):
    
# Determine the most relevant features, and display them.
    cpdist = classifier._feature_probdist 
    print('Most Informative Features') 
    count = 0
    for (fname, fval) in classifier.most_informative_features(n): 
        def labelprob(l):
            return cpdist[l,fname].prob(fval)

    labels = sorted([l for l in classifier._labels if fval in cpdist[l,fname].samples()], key=labelprob)

    if len(labels) == 1: 
    continue

    l0 = labels[0]
    l1 = labels[-1]

    if str(l0) == label1 and str(l1) == label2: 
        continue

    count += 1
    if count > 10: 
        continue

    if cpdist[l0,fname].prob(fval) == 0: 
        ratio = 'INF'
    else:
        ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) / cpdist[l0,fname].prob(fval))

    print(('%24s = %-14r %6s : %-6s = %s : 1.0' % (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))


########################
###   Question B4    ###
########################

import itertools
from nltk.collocations import BigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures

# Function for getting bigrams
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200): 
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

# Creating training feature set for tokens including bigram using already created training dataset in B1
train_feature_bi = []
for index, row in train_df.iterrows(): 
    jobDesc = row['FullDescription']
    tkns = nltk.word_tokenize(jobDesc) 
    for tk in tkns:
        tk = tk.lower()
        tkns_final = bigram_word_feats(tkns) 
        
        new_dict = dict()
        for tk in tkns_final:
            new_dict[tk] = new_dict.get(tk, 0) + 1 
        train_feature_bi.append((new_dict, train_df['SalaryFlag'][index]))

# Creating test feature set for tokens including bigram using already created test dataset in B1 
test_feature_bi = []
for index, row in test_df.iterrows(): 
    jobDesc = row['FullDescription'] 
    tkns = nltk.word_tokenize(jobDesc) 
    for tk in tkns:
        tk = tk.lower()
        tkns_final = bigram_word_feats(tkns) 
    new_dict = dict()
    for tk in tkns_final:
        new_dict[tk] = new_dict.get(tk, 0) + 1 
        test_feature_bi.append((new_dict, test_df['SalaryFlag'][index])) 
        classifier_bi = nltk.NaiveBayesClassifier.train(train_feature_bi)

print(nltk.classify.accuracy(classifier_bi, test_feature_bi))

# Creating confusion matrix 
ref_bi = [x[1] for x in test]
test_bi = [classifier.classify(x[0]) for x in test_feature_bi]
cm_bi = nltk.ConfusionMatrix(ref_bi, test_bi) 
print cm_bi
