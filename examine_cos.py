import numpy as np
import numpy.random as npr
import collections
import itertools
import networkx as nx
import matplotlib.pyplot as plt

def read_small_data(sample=4000, offset=0):
    """
    Got to arrange this for the user (or item) similarities, have a threshhold for similarities
    """
    user_maps = collections.defaultdict(set)
    max_id = 0
    with open("./data/u.data") as small_file:
        for line in small_file:
            offset -= 1
            if offset > 0:
                continue
            sample -= 1
            user_id, item_id, rating, timestamp = map(int, line.split())
            if user_id > max_id:
                max_id = user_id
            user_maps[item_id].add(user_id)
            if sample <= 0:
                break
    net = nx.Graph()
    for x in xrange(max_id):
        net.add_node(x)
    for _, user_set in user_maps.iteritems():
        for pair in itertools.combinations(user_set, 2):
            net.add_edge(*pair)
    return net

def generate_rtg_words(length):
    probs = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
    choices = map(int, list(npr.choice(5, length, p=probs)))
    members = "abcd "
    return "".join(members[choice] for choice in choices)

def wash_words(letters):
    words = letters.split()
    net = nx.Graph()
    for word1, word2 in zip(words, words[1:]):
        net.add_edge(word1, word2)
    return net

def create_net()
    rtg_words = generate_rtg_words(length)
    return wash_words(rtg_words)

def get_cos_mat(net):
    node_mat = nx.to_numpy_matrix()
    cos_mat = np.zeros_like(node_mat)
    for x in xrange(node_mat.shape[0]):
        for y in xrange(node_mat.shape[1]):
            cos_mat[x,y] = cos_dist of something

if __name__ == "__main__":
    net = create_net()
    net_mat = get_cost_mat(net)
    plt.imshow(net_mat)
    plt.colorbar()
    plt.show()
