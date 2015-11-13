import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def wash_words(words):
    net = nx.Graph()
    for fst, snd in zip(words, words[1:]):
        net.add_edge(fst, snd)
    return net

def cos_dist(fst, snd):
    pass

def cos_hist(net):
    neighborhoods = {}
    fill up the neighborhoods
    cosine_vals = []
    for node in net:
        for other node in net:
            cosine_vals.append(cos_dist(node, other node))
    return cosine_vals

if __name__ == "__main__":
    with open("corpus.txt") as corpus_file:
        words = corpus_file.read().split()[:5000]
    word_net = wash_words(words)
    cos_vals = create_cos_mat(word_net)
    plt.plot(cos_vals)
    plt.show()
