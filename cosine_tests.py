import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def wash_words(words):
    net = nx.Graph()
    for fst, snd in zip(words, words[1:]):
        net.add_edge(fst, snd)
    return net

def cos_dist(fst, snd):
    intersection = len(set(fst).intersection(set(snd)))
    return intersection / np.sqrt(len(fst) * len(snd))

def cos_hist(net):
    neighborhoods = {}
    for node in net.nodes():
        neighborhoods[node] = net.neighbors(node)
    cosine_vals = []
    for node1 in net.nodes():
        for node2 in net.nodes():
            cosine_vals.append(cos_dist(neighborhoods[node1], neighborhoods[node2]))
    return cosine_vals

def cos_mat(net):
    neighborhoods = {}
    for node in net.nodes():
        neighborhoods[node] = net.neighbors(node)
    mat = np.zeros((len(net.nodes()), len(net.nodes())))
    for idx1, node1 in enumerate(net.nodes()):
        for idx2, node2 in enumerate(net.nodes()):
            mat[idx1, idx2] = cos_dist(neighborhoods[node1], neighborhoods[node2])
    return mat


if __name__ == "__main__":
    with open("data/corpus.txt") as corpus_file:
        words = corpus_file.read().split()[:500]
    word_net = wash_words(words)
    mat = cos_mat(word_net)
    plt.imshow(mat)
    plt.colorbar()
    plt.show()
