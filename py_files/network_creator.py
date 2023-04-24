# data processing
import pandas as pd
import numpy as np
from numpy import genfromtxt

# graphs / figures
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# sklearn / models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# nltk
import nltk
from nltk.corpus import stopwords

# others
import os
import sys
sys.path.append('../')
from embedding_functions_hugo.embedding_functions import *
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from happytransformer import HappyTextClassification
import praw

def shorten_and_clean_dataset (comment_csv, comment_column : str, desired_comment_length : int):
    dataframe = pd.read_csv(comment_csv)
    dataframe['cleaned_text'] = prep_pipeline(dataframe, comment_column)
    dataframe['short'] = shorten_sens(dataframe['cleaned_text'], desired_comment_length)
    return dataframe


def do_post_titles(df):
    texts, authors, post_ids = [], [], []
    title_author_set = set(zip(df['post_title'], df['post_author'], df['post_id']))

    for text, author, post_id in  title_author_set:
        texts.append(text)
        authors.append(author)
        post_ids.append(post_id)
    
    title_embeddings = embed_comments(texts)
    return pd.DataFrame(data=[post_ids, authors, texts, title_embeddings]).T.rename(columns={0 : 'post_id', 1: 'post_author', 2: 'post_title', 3: 'embeddings'})



def remove_nan(df):
    '''Takes dataframe as input, removed rows where username is nan'''

    for idx, row in df.iterrows():
        usr1 = row[2]
        usr2 = row[4]

        if type(usr1) != str or type(usr2) != str:
            df.drop(idx, axis=0, inplace=True)
    
    return df


def shorten_comment_text(df):
    '''Takes dataframe as input, shortens comment text, creates short_text column and removes comment_text column'''
    
    # getting all commenters that have also made a post
    shortened_df = df.query('comment_author in post_author').copy()

    # cleaning their comments and saving to new column
    clean_text = prep_pipeline(shortened_df, 'comment_text', loud=False)
    shortened_df['short_text'] = shorten_sens(clean_text, 50)
    shortened_df.drop('comment_text', axis=1, inplace=True)

    return shortened_df

def get_embed_pairs(df):
   '''Takes dataframe as input, combines all user title texts into one
      Returns embed_pairs'''
    
   # for each poster, appending all of their post text into one long string
   embed_pairs = {}
   for text, author in zip(df['post_title'], df['post_author']):
      if author not in embed_pairs.keys():
         embed_pairs[author] = text
      else:
         embed_pairs[author] += ' '+text

   return embed_pairs

def get_title_embeds(embed_pairs):
   '''Takes embed_pairs as input. Embeds the post titles and returns an array of them.'''

   # embedding all post titles and reducing them to 1 dimension
   embeds = dict(zip(embed_pairs.keys(), embed_comments(list(embed_pairs.values()))))
   embeds = reduce_to_one_dimension_kmeans(embeds)[-1]

   return embeds

def reduce_to_poster_commentors(df):
    '''given a df, reduces rows recursively until all commentors and posters are both commentors and posters'''

    # might be faster with doing query multiple times until no changes

    done = False
    initialized = False
    while done != True:

        comment_authors = list(df['comment_author'])
        post_authors = list(df['post_author'])

        if initialized == False:
            before = 0
            initialized = True
        else:
            before = len(users_to_include)

        users_to_include = set()

        after = 0

        for idx, row in df.iterrows():

            commentor = row[2]

            if commentor in post_authors:
                if commentor not in users_to_include:
                    # print(commentor)
                    users_to_include.add(commentor)
                    after += 1
            
            else:
                df.drop(idx, inplace=True)

        diff = after - before
        
        if diff == 0:
            print(f'before = {before} and after = {after}, so diff = {diff}, done!')
            done = True
        else:
            print(f'before = {before} and after = {after}, so diff = {diff}, relooping...')

    return df

def create_network(df):
    '''Takes dataframe as input, creates a directed networkx network and returns it'''

    # Initializing text classification model
    model = HappyTextClassification(model_type='DISTILBERT', model_name='distilbert-base-uncased-finetuned-sst-2-english', num_labels=2)

    # initializing networkx directed graph
    G = nx.DiGraph()

    # # ensuring only looking at users that have both commented and posted
    # df = df.query('comment_author in post_author')

    print('doing embed_pairs')

    # combining users title texts into one
    embed_pairs = get_embed_pairs(df)

    print(len(embed_pairs))

    print('doing embeds')

    # embedding those combined title texts
    embeds = get_title_embeds(embed_pairs)

    print(len(embeds))

    # adding each post author as a node to network, with their 1 dimensional title embeddings as an attribute (named polarity)
    print('\n===== adding nodes ======\n')
    for i, j in zip(embed_pairs.keys(), embeds):
        print(f'added {i} with polarity {int(j)}')
        G.add_node(i, polarity = int(j))

    # getting list of user pairs for comments left on posts
    pairings = list(zip(df['comment_author'], df['post_author']))
    print(len(pairings))

    print('doing polarities')

    # classifying sentiments of comment texts and saving them in list "polarities"
    polarities = []
    signs = []
    for comment in df['short_text']:
        classification = model.classify_text(comment)
        polarities.append(classification.score)

        if classification.label == 'POSITIVE':
            signs.append(1)
        elif classification.label == 'NEGATIVE':
            signs.append(-1)
    
    print(len(polarities))

    print('\n===== adding edges =====\n')
    # adding edges to graph, where each edge is a comment left by user a to user b, and its attribute is the sentiment of the comment
    for pair, polarity, sign in zip(pairings, polarities, signs):
        if pair[0] != pair[1]: # avoiding self edge connections
            # need positive or negative for sign
            print(f'adding edge from {pair[0]} to {pair[1]} with sentiment {polarity} and label {sign}')
            G.add_edge(u_of_edge=pair[0], v_of_edge=pair[1], sentiment=polarity, sign=sign)

    # drawing network
    nx.draw(G)

    return G

def full_pipeline(df):
    '''does everything and returns network'''
    # doing pre-network stuff
    print('removing nan...')
    df = remove_nan(df)
    print('reducing to poster commentors...')
    df = reduce_to_poster_commentors(df)
    print('shortening comment text...')
    df = shorten_comment_text(df)

    print('creating network...')
    # creating and drawing network
    G = create_network(df)

    return G


filename = str(sys.argv[-1])
dest = filename.split('/')[-1].split('.')[0]

print(filename)

# grabbing scraped df
df = pd.read_csv(filename)

G = full_pipeline(df)

# check if the directory exists
if not os.path.exists('temp'):
    # create the directory if it doesn't exist
    os.makedirs('temp')

# saving network
nx.write_gexf(G, f'temp/{dest}.gexf')