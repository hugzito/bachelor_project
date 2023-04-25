import seaborn as sns
import networkx as nx

# getter functions
def get_title_embeddings(subreddit, path):
    '''Given a subreddit name and path to where the graph is stored,
       gets title embeddings, puts them to a list and returns that list.'''
    
    full_path = path + subreddit + '.gexf'

    G = nx.read_gexf(full_path)

    polarities = []
    for node in G.nodes(data=True):
        polarities.append(node[1]['polarity'])
   
    return polarities

def get_comment_sentiments(subreddit, path):
    '''Given a subreddit name and path to where its graph is stored,
       creates and returns a list of all comment sentiment scores'''
    
    full_path = path + subreddit + '.gexf'

    G = nx.read_gexf(full_path)

    sign_list = []
    sentiment_list = []
    for edge in list(G.edges(data=True)):
      sentiment = edge[2]['sentiment']
      sign = edge[2]['sign']

      if sign == -1:
         sentiment = 1-sentiment

      sentiment_list.append(sentiment)

    return sentiment_list

# main function
def do_both_save(subreddit, path):
    '''Given a subreddit name and path to its stored graph,
       does both plots and saves to data/plots/'''
    
    # documentation for displot here: https://seaborn.pydata.org/tutorial/distributions.html
    
    full_path = path + subreddit + 'gexf'

    # title embed plots
    title_embeds = get_title_embeddings(subreddit, path)

    title_plot = sns.displot(title_embeds,
                             kind='kde', # kernel density estimation for smoothed line instead of bar chart
                             bw_adjust=.3) # bw_adjust to reduce the line smoothing to retain more information at the cost of prettiness
    # bw_adjust as .3 so far seems the best for having some smoothing but still showing information
    
    title_plot.set(xlabel='Title Embeddings', title=subreddit)
    
    # comment sentiment plots
    sentiments = get_comment_sentiments(subreddit, path)
    sentiment_plot = sns.displot(sentiments, bins=10)
    sentiment_plot.set(xlabel='Comment Sentiment Score', title=subreddit)

    title_plot.savefig(('../data/plots/titles_' + subreddit))
    sentiment_plot.savefig(('../data/plots/sentiments_' + subreddit))
    return 