import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def wash_words(words):
    net = nx.Graph()
    for fst, snd in zip(words, words[1:]):
        net.add_edge(fst, snd)
    return net

def cos_hist(net):
    pass

if __name__ == "__main__":
    with open("corpus.txt") as corpus_file:
        words = corpus_file.read().split()[:5000]
    word_net = wash_words(words)
    cos_vals = create_cos_mat(word_net)
    plt.plot(cos_vals)
    plt.show()
