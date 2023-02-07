def embed_comments (comment_list):
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    sentences = comment_list
    sentence_embeddings = sbert_model.encode(sentences)
    return sentence_embeddings

def two_dims(sentences):
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    embedded_sens = embed_comments(sentences)
    pca = PCA(n_components=2)
    new_2d = pca.fit_transform(embedded_sens)
    x = []
    y = []
    for idx,i, in enumerate(new_2d):
        #print(i[0],i[1], sentences[idx])
        x.append(i[0])
        y.append(i[1])
    plot = plt.scatter(x = x, y=y)
    return [x,y,sentences], plot

def one_dim(sentences):
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    embedded_sens = embed_comments(sentences)
    pca2 = PCA(n_components=1)
    new_1d = pca2.fit_transform(sentence_embeddings)
    xy = {}
    for idx,i, in enumerate(new_1d):
        xy[sentences[idx]] = i
    plot = plt.scatter(xy.values(),[0 for i in range(len(xy))])
    return xy, plot