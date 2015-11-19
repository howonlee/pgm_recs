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
import operator as op
import cProfile
import dtw

def select_net(net, p=0.9):
    """
    Take a network and sample a portion of it,
    where p = the amount that you're sampling
    """
    new_net = nx.Graph()
    edge_list = net.edges()
    new_net_edges = []
    for edge in edge_list:
        if random.random() < p:
            new_net_edges.append(edge)
    new_net.add_edges_from(new_net_edges)
    return new_net

def l2_norm(x, y):
    #because I am too lazy to actually figure out how to do it with np norm
    return np.sqrt(x ** 2 + y ** 2)

def generate_biggest_matching(src_net, tgt_net, num_seeds):
    src_degs = map(op.itemgetter(0), sorted(nx.degree(src_net).items(), key=op.itemgetter(1), reverse=True)[:num_seeds])
    tgt_degs = map(op.itemgetter(0), sorted(nx.degree(tgt_net).items(), key=op.itemgetter(1), reverse=True)[:num_seeds])
    return zip(src_degs, tgt_degs)

def generate_possible_swap(matching):
    to_swap1, to_swap2 =\
        random.randint(0, len(matching)-1),\
        random.randint(0, len(matching)-1)
    while to_swap1 == to_swap2:
        to_swap1, to_swap2 =\
            random.randint(0, len(matching)-1),\
            random.randint(0, len(matching)-1)
    return to_swap1, to_swap2,\
        matching[to_swap1], matching[to_swap2]

def apply_swap(curr_matching, swap):
    i, j, i_match, j_match = swap
    new_matching = curr_matching[:]
    new_matching[i] = (i_match[0], j_match[1])
    new_matching[j] = (j_match[0], i_match[1])
    return new_matching

def generate_matching_neighbors(matching, num_neighbors=20):
    # unzip, switch some orders, rezip
    neighbors = []
    def rand_idx(it):
        return random.randint(0, len(it) - 1)
    for x in xrange(num_neighbors):
        curr_matching = matching[:]
        src_words = map(op.itemgetter(0), matching)
        tgt_words = map(op.itemgetter(1), matching)
        for y in xrange(5):
            first, second = rand_idx(tgt_words), rand_idx(tgt_words)
            while first == second:
                first, second = rand_idx(tgt_words), rand_idx(tgt_words)
            tgt_words[first], tgt_words[second] = tgt_words[second], tgt_words[first]
        neighbors.append(zip(src_words, tgt_words))
    return neighbors

def cos_dist(fst, snd):
    intersection = len(set(fst).intersection(set(snd)))
    return intersection / np.sqrt(len(fst) * len(snd))

def cosine_mat(net):
    neighborhoods = {}
    for node in net.nodes():
        neighborhoods[node] = net.neighbors(node)
    mat = np.zeros((len(net.nodes()), len(net.nodes())))
    for idx1, node1 in enumerate(net.nodes()):
        for idx2, node2 in enumerate(net.nodes()):
            mat[idx1, idx2] = cos_dist(neighborhoods[node1], neighborhoods[node2])
    return mat

def energy_diff_wrapper(src_net, tgt_net):
    src_sigmas = cosine_mat(src_net)
    tgt_sigmas = cosine_mat(tgt_net)
    src_sigma_means = src_sigmas.mean(axis=1)
    tgt_sigma_means = tgt_sigmas.mean(axis=1)
    alpha = 0.5
    beta = 0.5
    def pair_dist(x, y):
        r = float(x) / y if x > y else float(y) / x
        return (r - 1) ** alpha
    def energy(i, j):
        scaling_factor = (src_sigma_means[i] * tgt_sigma_means[i]) ** (beta / 2)
        weight = pair_dist(src_sigmas[i,j] / src_sigma_means[i], tgt_sigmas[i,j] / tgt_sigma_means[i])
        return scaling_factor * weight
    def total_energy():
        """
        There is a less idiotic way to do it
        I don't care
        """
        total = 0
        for x in xrange(src_sigmas.shape[0]):
            for y in xrange(src_sigmas.shape[1]):
                total += energy(x,y)
        return total
    return energy, total_energy

def search_annealing(src_net, tgt_net, biggest_matching, num_tries=5, num_iters=10000):
    """
    Currently just gradient descent
    """
    all_matchings = []
    len_src = len(src_net.nodes())
    len_tgt = len(tgt_net.nodes())
    for x in xrange(num_tries):
        calc_energy, calc_total_energy = energy_diff_wrapper(src_net, tgt_net)
        curr_matching = biggest_matching[:]
        for y in xrange(num_iters):
            if y % 1000 == 0:
                print "try: ", x
                print "annealing initial matching: ", y, " / ", str(num_iters)
            swap = generate_possible_swap(curr_matching)
            i, j, i_tup, j_tup = swap
            if i_tup[1] > len_tgt or j_tup[1] > len_tgt:
                continue
            energy = calc_energy(*i_tup) + calc_energy(*j_tup)
            energy_2 = calc_energy(i_tup[0], j_tup[1]) + calc_energy(j_tup[0], i_tup[1])
            if energy_2 - energy > 0: # apply annealing here when we do it
                curr_matching = apply_swap(curr_matching, swap)
        all_matchings.append((calc_total_energy(), curr_matching))
    best_matching = min(all_matchings, key=operator.itemgetter(0))[1]
    return curr_matching

def generate_seeds(src_net, tgt_net, num_seeds=50):
    matching = generate_biggest_matching(src_net, tgt_net, num_seeds)
    return search_annealing(src_net, tgt_net, matching)

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

def expand_once_pgm(net1, net2, seeds, r):
    ################
    ################
    ################
    ################
    pass

def expando_pgm(net1, net2, seeds, r): #seeds is a list of tups
####################3
####################3
####################3
####################3
####################3
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
    """
    Generates the words for the RTG
    """
    probs = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
    choices = map(int, list(npr.choice(5, length, p=probs)))
    members = "abcd "
    return "".join(members[choice] for choice in choices)

def wash_words(letters):
    """
    Takes a sequence that can be split and turns it into a bigram network
    """
    words = letters.split()
    word_map = dict([(tup[1], tup[0]) for tup in enumerate(set(words))])
    net = nx.Graph()
    for word1, word2 in zip(words, words[1:]):
        net.add_edge(word_map[word1], word_map[word2])
    return net

# cap the length and try the extra thought
def generate_rtg(length=10000):
    """
    Generate a Random Typer Graph a la Akoglu et al
    Really rudimentary, I forget which of the various RTG versions this is
    """
    rtg_words = generate_rtg_words(length)
    return wash_words(rtg_words)

if __name__ == "__main__":
    random.seed(123456) #different seed :)
    rtg_1 = generate_rtg()
    rtg_2 = generate_rtg()
    print generate_seeds(rtg_1, rtg_2)
    print "now go do expando_pgm properly"
