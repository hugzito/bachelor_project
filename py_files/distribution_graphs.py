import seaborn as sns
import networkx as nx

# getter functions
def get_title_embeddings(path):
    '''Given the path to a graph, gets title embeddings,
       puts them to a list and returns that list'''

    G = nx.read_gexf(path)

    polarities = []
    for node in G.nodes(data=True):
        polarities.append(node[1]['polarity'])
   
    return polarities

def get_comment_sentiments(path):
    '''Given the path to a graph, creates and returns a list of all comment sentiment scores'''

    G = nx.read_gexf(path)

    sentiment_list = []
    for edge in list(G.edges(data=True)):
      sentiment = edge[2]['sentiment']
      sentiment_list.append(sentiment)

    return sentiment_list

# main function
def do_both_save(title, path):
    '''Given a subreddit name and path to its graph,
       does both plots and saves to data/plots/
       "title" is used for the title of the graphs'''
    
    # documentation for displot here: https://seaborn.pydata.org/tutorial/distributions.html

    sns.set_style('darkgrid')
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

    # title embed plots
    title_embeds = get_title_embeddings(path)

    title_plot = sns.displot(title_embeds,
                             kind='kde', # kernel density estimation for smoothed line instead of bar chart
                             bw_adjust=.3) # bw_adjust to reduce the line smoothing to retain more information at the cost of prettiness
    # bw_adjust as .3 so far seems the best for having some smoothing but still showing information
    title_plot.set(xlabel='Title Embeddings', title=title,
                   xlim=(-11.5, 16.5), ylim=(0,.18)) # xlim=(-10, 12) seems good
    
    # comment sentiment plots
    sentiments = get_comment_sentiments(path)
    sentiment_plot = sns.displot(sentiments,
                                 #kind='hist',
                                 stat='density',
                                 bins=11) # bins should be odd to allow for middle bin
    sentiment_plot.set(xlabel='Comment Sentiment Score', title=title,
                       xlim=(-1.02, 1.02), ylim=(0,4))

    title_plot.savefig(('../data/plots/titles_' + title))
    sentiment_plot.savefig(('../data/plots/sentiments_' + title))
    return