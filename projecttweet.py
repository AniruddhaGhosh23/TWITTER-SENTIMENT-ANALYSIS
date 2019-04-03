#Tweets are generally saved in json format. Opening a json file looks like a python 'dictionary'. 'Tweepy' is a package that helps us to collect data directly in a '.csv' file and download it or use it. So, first we're going to install tweepy.
! pip install tweepy

#To perform sentiment analysis in twitter basically we need to have a twitter account. But for authentication we need to create a twitter app first
consumer_key = "9SvIkZ5zdoOsWbkhG1rXX6uPt"
consumer_secret = "KOaJikXJ65hOYwscZplvHWQdKsi83RYJdfL28VLg5om4IdeQeQ"
access_key = "3304485310-yUfQZM0xF9yMrkMuxLH4vIqDxrrcoFucbVgSUjg"
access_secret = "wSP1lK4HAQ2ghIyCOvnC9uY9oNAYzCUE6mCoPokx4Hn5e"

import tweepy
import csv
from google.colab import files
import collections
from sklearn.svm import LinearSVC
import collections
import nltk.classify.util, nltk.metrics
from nltk.metrics import precision as precision
from nltk.metrics import recall as recall
from nltk.metrics import f_measure as f_measure
from nltk.classify import SklearnClassifier
from nltk.classify import NaiveBayesClassifier 
from nltk.corpus import stopwords

#downloading tweets
def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    alltweets = []  

    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1

    c=1

    while c > 0:
        print("getting tweets before %s" % (oldest))
        new_tweets = api.user_timeline(screen_name = screen_name,count=400,max_id=oldest)

        alltweets.extend(new_tweets)

        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))
        
        c = 0

    outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]

    #write the csv  
    with open('%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        #writer.writerow(["text"])
        writer.writerows(outtweets)
        #files.download('%s_tweets.csv' % screen_name)

screen_list = ["narendramodi","RahulGandhi","SitaramYechury","smritiirani","laluprasadrjd","MamataOfficial"]

for i in range(len(screen_list)):
    get_all_tweets(screen_list[i])

#Now we will be taking some predefined positive and negative words those will be used in supervised learning as training samples. positive values are stored in a list named posdata. Negative values are stored in the list negdata
posdata = []
negdata = []
with open('positive-data.csv', 'r') as myfile:
    reader = csv.reader(myfile, delimiter = ',')
    for col in reader:
        posdata.append(col[0])
        
with open('negative-data.csv', 'r') as myfile:
    reader = csv.reader(myfile, delimiter = ',')
    for col in reader:
        negdata.append(col[0])

#used natural language processing for cleaning the data of posyivity and negativity
import nltk
nltk.download('stopwords')
def word_split(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new

def word_feats(words):
    return dict([(word, True) for word in words])

stopset = set(stopwords.words('english'))

def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])

negfeats = [(stopword_filtered_word_feats(f), 'neg') for f in word_split(negdata)]
posfeats = [(stopword_filtered_word_feats(f), 'pos') for f in word_split(posdata)]
"""
negcutoff = int(len(negfeats)*10/11)
poscutoff = int(len(posfeats)*10/11)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
"""
#dividing train and test set
import numpy as np
from sklearn.cross_validation import train_test_split

allfeats = posfeats + negfeats
allfeats = np.array(allfeats)
X = allfeats[:,0]
y = allfeats[:,1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state = 42)

trainfeats = [(i,j) for i,j in zip(X_train, y_train)]
testfeats = [(i,j) for i,j in zip(X_test, y_test)]   

#trained with linear svm and naivebayes
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

choice = int(input("Enter 1 for SVC, Enter 2 for NaiveBayes : "))
if choice == 1:
    
    str = 'SINGLE FOLD RESULT ' + '(' + 'linear-svc' + ')'
    
    #training with LinearSVC
    classifier = SklearnClassifier(LinearSVC())
    classifier.train(trainfeats)
    
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    accuracy = nltk.classify.util.accuracy(classifier, testfeats)*100
    pos_precision = nltk.precision(refsets['pos'], testsets['pos'])

elif choice == 2:
    
    str = 'SINGLE FOLD RESULT ' + '(' + 'naivebayes' + ')'
    
    #training with NaiveBayes
    classifier = nltk.NaiveBayesClassifier.train(trainfeats)
    
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    accuracy = nltk.classify.util.accuracy(classifier, testfeats)*100
    pos_precision = nltk.precision(refsets['pos'], testsets['pos'])

pos_recall = recall(refsets['pos'], testsets['pos'])
pos_fmeasure = f_measure(refsets['pos'], testsets['pos'])
neg_precision = precision(refsets['neg'], testsets['neg'])
neg_recall = recall(refsets['neg'], testsets['neg'])
neg_fmeasure = f_measure(refsets['neg'], testsets['neg'])

print ('')
print ('---------------------------------------')
print (str)
print ('---------------------------------------')
print ('accuracy: ', accuracy,'%')
print ('precision', (pos_precision + neg_precision) / 2)
print ('recall', (pos_recall + neg_recall) / 2)
print ('f-measure', (pos_fmeasure + neg_fmeasure) / 2)  

#filename = 'realDonaldTrump_tweets.csv' was for testing
filename = []
for i in screen_list:
    filename.append(i+'_tweets.csv')

print("NAME OF THE FILES READY FOR CLEANING")
print("------------------------------------")
for i in filename:
    print(i)
def clean_data(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    # split into words by white space
    words = text.split("\n")
    words = [word.lower() for word in words]
    print(words[:100])

    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    print(stripped[:100])

    # create cleaned file
    file = open("cleaned_"+filename,"w")
    for i in stripped:
        file.write(i+"\n")
    file.close()

    #files.download('cleaned_'+filename)

print('\n')
print("        DATA CLEANING STATUS        ")
print("------------------------------------")
for i in filename:
    clean_data(i)
    print(i, 'is cleaned\t|', 'cleaned_'+i, 'is created and downloaded')
    print('\n')

#NAME OF THE FILES READY FOR CLEANING
#------------------------------------
#narendramodi_tweets.csv
#RahulGandhi_tweets.csv
#SitaramYechury_tweets.csv
#smritiirani_tweets.csv
#laluprasadrjd_tweets.csv
#MamataOfficial_tweets.csv

def create_cleaned_tweet(filename):
    tweets_local = []
    with open('cleaned_'+filename, 'r') as myfile:
        reader = csv.reader(myfile)
        for col in reader:
            try:
                tweets_local.append(col[0])
            except:
                print("empty list encountered")        
    return tweets_local

p = []
n = []
tweets = []

for i in filename:
    tweets.append(create_cleaned_tweet(i))
# print(type(tweets)   created for testing

for i in range(len(tweets)):
    myfeats = [(stopword_filtered_word_feats(f), 'true') for f in word_split(tweets[i])]
    testsets_tweets = collections.defaultdict(set)

    for j, (feats, label) in enumerate(myfeats):
        observed = classifier.classify(feats)
        testsets_tweets[observed].add(j)
    p.append(len(testsets_tweets['pos']))
    n.append(len(testsets_tweets['neg']))
print(tweets)
# print(p)   created for testing
# print(n)   created for testing
print(p)
print(n)

#visualising result
import matplotlib.pyplot as plt
#plt.bar(['total','pos','neg'],[p+n,p,n],color = ['k','g','r'])

plt.figure(figsize=(12,8))
for i in range(len(filename)):
    plt.subplot(2,3,i+1)
    plt.bar(['pos','neg'],[p[i],n[i]],color = ['g','r'])
    plt.title('tweets for screen name : '+screen_list[i])
    plt.ylabel('number of tweets')
plt.show()