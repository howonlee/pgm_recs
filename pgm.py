import networkx as nx
import numpy as np
import numpy.random as npr
import sys
import math
import random
import collections
import matplotlib.pyplot as plt
import itertools
import scipy.stats as sci_st
import operator
import cProfile
import dtw

def select_net(net, p=0.9):
    new_net = nx.Graph()
    edge_list = net.edges()
    new_net_edges = []
    for edge in edge_list:
        if random.random() < p:
            new_net_edges.append(edge)
    new_net.add_edges_from(new_net_edges)
    return new_net

def path_to_seeds(path):
    lset = set() #left and right paths, we think of them as
    rset = set()
    lpath, rpath = map(list, path)
    seeds = []
    for idx, x in enumerate(lpath):
        y = rpath[idx]
        if x in lset:
            continue
        if y in rset:
            continue
        seeds.append((x, y))
        lset.add(x)
        rset.add(y)
    return seeds

def l2_norm(x, y):
    #because I am too lazy to actually figure out how to do it with np norm
    return np.sqrt(x ** 2 + y ** 2)

def get_seeds(src_net, tgt_net, num_seeds):
    #we're going to need to skip some, friend
    src_degs = sorted(nx.degree(src_net).items(), key=operator.itemgetter(1), reverse=True)
    tgt_degs = sorted(nx.degree(tgt_net).items(), key=operator.itemgetter(1), reverse=True)
    src_dists = list(itertools.islice(map(operator.itemgetter(1), src_degs), num_seeds))
    tgt_dists = list(itertools.islice(map(operator.itemgetter(1), tgt_degs), num_seeds))
    dist, cost, path = dtw.dtw(src_dists, tgt_dists, dist=l2_norm)
    seeds = path_to_seeds(path)
    return seeds

def normal_pgm(net1, net2, seeds, r): #seeds is a list of tups
    """
    Gets about 1200 matches on 2048 SKG, selected 90%
    """
    marks = collections.defaultdict(int)
    imp_t, imp_h = set(), set()
    unused, used = seeds[:], []
    random.shuffle(unused) # mutation!
    while unused:
        t2 = 0
        curr_pair = unused.pop()
        for neighbor in itertools.product(net1.neighbors(curr_pair[0]), net2.neighbors(curr_pair[1])):
            if neighbor[0] in imp_t or neighbor[1] in imp_h:
                continue
            marks[neighbor] += 1
            t2 += 1
            if marks[neighbor] > r:
                unused.append(neighbor)
                imp_t.add(neighbor[0])
                imp_h.add(neighbor[1])
        used.append(curr_pair)
    return used

def expando_pgm(net1, net2, seeds, r): #seeds is a list of tups
    marks = collections.defaultdict(int)
    imp_t, imp_h = set(), set()
    unused, matched, candidates = seeds[:], [], []
    random.shuffle(unused) # mutates!
    while unused:
        t2 = 0
        curr_pair = unused.pop()
        for neighbor in itertools.product(net1.neighbors(curr_pair[0]), net2.neighbors(curr_pair[1])):
            if neighbor[0] in imp_t or neighbor[1] in imp_h:
                continue
            marks[neighbor] += 1
            t2 += 1
            if marks[neighbor] > r:
                unused.append(neighbor)
                imp_t.add(neighbor[0])
                imp_h.add(neighbor[1])
        matched.append(curr_pair)
    return matched

def generate_skg_arr(order=11):
    gen = np.array([[0.99, 0.7], [0.7, 0.1]])
    skg = gen.copy()
    for x in xrange(order-1):
        skg = np.kron(skg, gen)
    skg_sample = np.zeros_like(skg)
    for x in xrange(skg.shape[0]):
        for y in xrange(skg.shape[1]):
            if random.random() < skg[x,y]:
                skg_sample[x,y] = 1
    return skg_sample

def generate_skg_net(order=11):
    arr = generate_skg_arr(order)
    return nx.from_numpy_matrix(arr)

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
    # similarities. Now.
    return net

def perform_swap(arr, swap):
    first, second = swap
    arr[:, [first, second]] = arr[:, [second, first]]
    arr[[first, second], :] = arr[[second, first], :]
    return arr

def shuffle_arr(arr, num_swaps=10000):
    """
    Take it in matrix form
    """
    swaps = [(random.randint(0, arr.shape[0]-1), random.randint(0, arr.shape[0]-1)) for x in xrange(num_swaps)]
    new_arr = arr.copy()
    for swap in swaps:
        new_arr = perform_swap(new_arr, swap)
    return arr, swaps

def show_net(net):
    arr = nx.to_numpy_matrix(net)
    plt.imshow(arr)
    plt.show()

def similarities(arr, fst, snd):
    # vectorize later
    fst_arr = np.ravel(arr[:, fst])
    snd_arr = np.ravel(arr[:, snd])
    return sci_st.pearsonr(fst_arr, snd_arr)

def euclidean_similarities(arr, fst, snd):
    fst_arr = np.ravel(arr[:, fst])
    snd_arr = np.ravel(arr[:, snd])
    return np.sqrt(np.sum((fst_arr - snd_arr) ** 2))

def filter_net(net):
    """
    Get it to O(n log n) please
    """
    print "start filtering..."
    arr = nx.to_numpy_matrix(net)
    filtered_arr = np.zeros_like(arr)
    thresh = 2
    print arr.shape, thresh
    for x in xrange(arr.shape[0]):
        if x % 10 == 0:
            print "on node: ", x
        for y in xrange(arr.shape[1]):
            #if similarities(arr, x, y)[1] < 0.0001:
            if euclidean_similarities(arr, x, y) > thresh:
                """
                that is, if we pass pearson test
                """
                filtered_arr[x, y] = 1
    print "end filtering..."
    filtered_net = nx.from_numpy_matrix(filtered_arr)
    return filtered_net

def similarity_mat(arr):
    print arr.shape
    sim_mat = np.zeros_like(arr)
    for x in xrange(arr.shape[0]):
        if x % 10 == 0:
            print "on node: ", x
        for y in xrange(arr.shape[1]):
            sim_mat[x,y] = euclidean_similarities(arr, x, y)
    return sim_mat

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

# cap the length and try the extra thought
def generate_rtg(length=10000):
    rtg_words = generate_rtg_words(length)
    return wash_words(rtg_words)

if __name__ == "__main__":
    random.seed(123456) #different seed :)
    rtg_1 = generate_rtg()
    rtg_2 = generate_rtg()
    print normal_pgm(rtg_1, rtg_2, seeds, 3)
