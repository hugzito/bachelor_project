#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import sys
sys.path.append('../')
from embedding_functions_hugo.embedding_functions import *
from sklearn.preprocessing import StandardScaler
import numpy as np


# In[24]:


get_ipython().system('pipreqs .')


# In[7]:


df_gaming = pd.read_csv('../data/gaming.csv')
df_satis = pd.read_csv('../data/SatisfactoryGame.csv')
df_marauders = pd.read_csv('../data/MaraudersGame.csv')
df_tarkov = pd.read_csv('../data/EscapefromTarkov.csv')
df_politics = pd.read_csv('../data/politics.csv')
df_politics['cleaned_text'] = prep_pipeline(df_politics, 'comment_text')


# In[8]:


authors = df_politics.values[:,-2]


# In[16]:


for n in [df_gaming, df_marauders, df_tarkov]:
    n['cleaned_text'] = prep_pipeline(n, 'comment_text')


# In[19]:


df_tarkov['cleaned_text']


# In[20]:


# embed_comments(df1['comment_text'])
#two_dims(df1['comment_text'])
#tarkov = embed_comments(df_tarkov['comment_text'])
#marauders = embed_comments(df_marauders['comment_text'])
#satis = embed_comments(df_satis['comment_text'])
pol_embs= embed_comments(df_marauders['cleaned_text'])


# In[ ]:


# print(len(pol_embs), len(df_politics))
usr_id = df_politics['comment_author']
usr_dict = {}
for i, j in zip(authors, pol_embs):
    if i not in usr_dict.keys():
        usr_dict[i] = []
        usr_dict[i].append(j)
    else:
        usr_dict[i].append(j)
len(usr_dict)
len(pol_embs)


# In[ ]:


for i in usr_dict:
    g = len(usr_dict[i])
    usr_dict[i] = sum(usr_dict[i])/g


# In[ ]:


new_pol_embs = list(usr_dict.values())


# In[ ]:


two_dims(new_pol_embs, True)


# In[ ]:


def clustering_comments(comments, cluster_count, pre_emb = False, scaler = False): 
    '''Takes a collection of comments, maps it to a 2d/1d space and clusters them in a user-determined amount of groups.'''
    # imports 
    from sklearn.cluster import KMeans
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from numpy import unique
    from numpy import where
    #load_data
    if not pre_emb:
        sentences = embed_comments(comments)
    else:
        sentences = comments
    # Initialize model
    model = KMeans(n_clusters=cluster_count)

    if scaler:
        sentences = scaler.fit_transform(sentences)
    # Fit model
    model.fit(sentences)

    #Make predictions
    yhat = model.predict(sentences)

    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(sentences[row_ix, 0], sentences[row_ix, 1])
        # show the plot
    plt.show()


# In[ ]:


clustering_comments(new_pol_embs, 4, True, False)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))
print()


# In[ ]:


# Using PCA from sklearn PCA
pca = decomposition.PCA(n_components=2)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)

# Plotting the results of PCA
plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0);

