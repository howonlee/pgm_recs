import numpy as np
import numpy.random as npr
import networkx as nx
import matplotlib.pyplot as plt

def wash_words(words):
    net = nx.Graph()
    for fst, snd in zip(words, words[1:]):
        net.add_edge(fst, snd)
    return net

def cos_dist(fst, snd):
    """
    Try to get a fast intersection? by sorting or other method?
    Cache it? Precompute them as sets, because that's actually nontrivial work, anyhow. "node_sets"
    """
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

def generate_rtg_words(length):
    probs = np.array([0.05, 0.05, 0.1, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.3])
    choices = map(int, list(npr.choice(10, length, p=probs)))
    members = "abcdefghi "
    return "".join(members[choice] for choice in choices)

if __name__ == "__main__":
    #with open("data/corpus.txt") as corpus_file:
    #    words = corpus_file.read().split()[:4000]
    words = generate_rtg_words(10000).split()
    word_net = wash_words(words)
    mat = cos_mat(word_net)
    plt.imshow(mat)
    plt.colorbar()
    plt.show()
