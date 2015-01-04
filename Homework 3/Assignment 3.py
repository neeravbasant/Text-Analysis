
######################
###   Question 3   ###
######################

"""
    Unweighted PageRank
"""

import networkx as nx
import csv
#create graph
G = nx.Graph()
#create list of all car models
with open(‘C:\car.csv’,’rU’)  as f:
	fr = csv.reader(f)
	for i in fr:
		G.add_edge(i[0],i[i])
pr = nx.pagerank(G,alpha=0.85)
print pr

"""
    Weighted PageRank
"""

import networkx as nx
import csv

import os
os.getcwd()
'/Users/sumi'
os.chdir('/Users/sumi/documents/Study')
os.chdir('/Users/sumi/documents/Study/Fall/Text_analysis/Assignment/3')

G = nx.Graph()
G.add_node(1,Brand="ES")
G.add_node(2,Brand="LS")
G.add_node(3,Brand="RX")
G.add_node(4,Brand="A8")
G.add_node(5,Brand="A6")
G.add_node(6,Brand="3series")
G.add_node(7,Brand="5series")
G.add_node(8,Brand="7series")
G.add_node(9,Brand="XJ")
G.add_node(10,Brand="Sclass")
c=csv.reader(open("input.csv","rU"))
for i in c:
G.add_edge(int(i[0]), int(i[1]), weight=float(i[2]))
 
pr= nx.pagerank(G,alpha=0.85)
pr



######################
###   Question 3   ###
######################

import nltk
import os
import csv
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import math

"""
    First of all, we will read Edmund Posts xlsx file (all tabs) and collate them.
    Then we will do few treatments like removing NaNs and duplicates.
"""

# Reading Data
path = os.getcwd()
files = os.listdir(path)
files

files_xlsx = [f for f in files if f[-4:] == 'xlsx']
files_xlsx

df = pd.DataFrame()

data1 = pd.read_excel(files_xlsx[0], 'Posts 1st set')
data2 = pd.read_excel(files_xlsx[0], 'Posts 2nd set')
data3 = pd.read_excel(files_xlsx[0], 'Posts 3rd set')
data4 = pd.read_excel(files_xlsx[0], 'Posts 4th set')
data5 = pd.read_excel(files_xlsx[0], 'Posts 5th set')
data6 = pd.read_excel(files_xlsx[0], 'Posts 6th set')

pieces = [df, data1, data2, data3, data4, data5, data6]
df_collated = pd.concat(pieces)

# Removing NaN's and duplicates from the collated file
df_new = df_collated[pd.notnull(df_collated['Posts'])]
df_final = df_new.drop_duplicates(cols='Posts')

"""
    Once the data is properly read, next step is to treat and remove punctuations, tokenize the sentence and remove stopwords.
    After that we will lemmatize the remaining tokens and consider only those words which have a length of greater than 1.
    Beacuse any word of length = 1 would mostly be gibberish. For instance, ":P" would have become "P" by now.
"""

# Removing punctuations, tokenizing and removing stopwords

reviewlines = pd.Series.tolist(df_final['Posts'])

reviewlinesfinal = []
for rl in reviewlines:
    rl = rl.encode('ascii','ignore')
    rl = rl.replace("'", " '")
    rl = rl.replace(".", " .")
    rl = rl.replace("-", " -")
    rl = rl.replace(",", " ,")
    rl = rl.replace("!", " !")
    rl = rl.replace("?", " ?")
    rl = rl.strip().lower().translate(None, "!#$%&()*+,-./:;<=>?@[\]^_'{|}~")
    rl = nltk.word_tokenize(rl)
    temprl = []
    stop = stopwords.words('english')
    for word in rl:
        if word not in stop:
            temprl.append(word)
    reviewlinesfinal.append(temprl)
    
# Lemmatizing    
wnl = WordNetLemmatizer()
reviewlinesfinallemmatized = []
for rlf in reviewlinesfinal:
    templmt = []
    for word in rlf:
        if len(word) > 1:
            templmt.append(wnl.lemmatize(word))
        else:
            continue
    reviewlinesfinallemmatized.append(templmt)

"""
    Next step is to get the index for each occurence of all the 10 models in a particular review.
    Once we have those indexes, we will use the to create chunks by traversing right and left of that index.
    There are few considerations that we need to take care of. For example, chunk of a particular model should 
    not consider other models in it because it might include sentiment for either of the 2 cars.
"""    

# Getting indexes for each occurence of car model
    
model = ['lexuses','lexusls','rx','a8','a6','3series','5series','7series','xj','sclass']

indexlist = []

for index, rlfl in enumerate(reviewlinesfinallemmatized):
    ind = {}
    for i, tk in enumerate(rlfl):
        if tk in model:
            if tk not in ind.keys():
                ind[tk] = ind.get(tk, [i])
            else:
                ind[tk].append(i)
    indexlist.append(ind)

# Creating chunks
chunklist = []
nwords = 10

for index, il in enumerate(indexlist):
    chunkdictionary = {}
    for key, value in il.iteritems():
        clist = []
        for v in value:
            chunk = []
            flag1 = 0
            flag2 = 0
            for i in range(1, nwords + 1):
                try:
                    if reviewlinesfinallemmatized[index][(v-i)] != key and reviewlinesfinallemmatized[index][(v-i)] in model:
                        flag1 = 1
                    if  flag1 == 0:
                        chunk.append(reviewlinesfinallemmatized[index][(v-i)])
                except:
                    pass
                try:
                    if reviewlinesfinallemmatized[index][(v+i)] != key and reviewlinesfinallemmatized[index][(v+i)] in model:
                        flag2 = 1
                    if  flag2 == 0:
                        chunk.append(reviewlinesfinallemmatized[index][(v+i)])
                except:
                    pass
            clist.extend(chunk)
        chunkdictionary[key] = clist
    chunklist.append(chunkdictionary)
    
"""
    Then we will use sentistrength dictionary to get the sentiment scores for each car model.
    We will normalize the result as the length of the chunk might vary for each model in a particular review.
    We will store the final output in the form of list of dictionaries.
    Finally, we will convert this list of dictionaries to a csv file.
"""

# Getting sentiments

dictionary = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Text Mining and Decision Analysis/GH 3/Sentistrength_Dictionary.csv")
y = dictionary.set_index('Word').to_dict()

sentimentscore = []        
        
for index, il in enumerate(chunklist):
    postscore = {}
    count = {}
    postscorefinal = {}
    for key, value in il.iteritems():        
            for ch in value:
                count[key] = count.get(key,0) + 1
                for k, v in y['Score'].iteritems():
                    if ch == k:
                        postscore[key] = postscore.get(key,0) + v
            try:
                postscorefinal[key] = float((postscore[key])/math.sqrt(count[key]))
            except:
                pass
    sentimentscore.append(postscorefinal)

# Get the final output in csv

f = open('Sentiment_Score_Final.csv', 'wb')
dict_writer = csv.DictWriter(f, model)
dict_writer.writer.writerow(model)
dict_writer.writerows(sentimentscore)

