import networkx as nx
import numpy as np
import numpy.random as npr
import sys
import math
import random
import nltk
import collections
import matplotlib.pyplot as plt
import itertools
import scipy.stats as sci_st
import operator as op
import cProfile
import dtw
from blist import sortedlist

def select_net(net, p=0.9):
    """
    Take a network and sample a portion of it,
    where p = the amount that you're sampling
    """
    new_net = nx.Graph()
    # every node is added, but not every edge
    for node in net.nodes():
        new_net.add_node(node)
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

def cos_dist(fst, snd):
    intersection = len(set(fst).intersection(set(snd)))
    return float(intersection) / np.sqrt(len(fst) * len(snd))

def generate_biggest_matching(src_net, tgt_net, num_seeds):
    src_degs = map(op.itemgetter(0), sorted(nx.degree(src_net).items(), key=op.itemgetter(1), reverse=True)[:num_seeds])
    tgt_degs = map(op.itemgetter(0), sorted(nx.degree(tgt_net).items(), key=op.itemgetter(1), reverse=True)[:num_seeds])
    return zip(src_degs, tgt_degs)

def generate_smallest_matching(src_net, tgt_net, num_seeds):
    src_degs = map(op.itemgetter(0), sorted(nx.degree(src_net).items(), key=op.itemgetter(1))[:num_seeds])
    tgt_degs = map(op.itemgetter(0), sorted(nx.degree(tgt_net).items(), key=op.itemgetter(1))[:num_seeds])
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
    src_sigmas = np.nan_to_num(cosine_mat(src_net))
    tgt_sigmas = np.nan_to_num(cosine_mat(tgt_net))
    # naive smoothing
    src_sigma_means = src_sigmas.mean(axis=1)
    tgt_sigma_means = tgt_sigmas.mean(axis=1)
    alpha = 0.5
    beta = 0.5
    def pair_dist(x, y):
        # this does not work right
        r = float(x) / y if x > y else float(y) / x
        if r != r:
            return 0
        if r > 10000: # doesn't matter
            return 0
        return np.sqrt(r - 1.0)
    def energy(i, j):
        scaling_factor = (src_sigma_means[i] * tgt_sigma_means[i]) ** (beta / 2)
        weight = pair_dist(src_sigmas[i,j] / src_sigma_means[i], tgt_sigmas[i,j] / tgt_sigma_means[i])
        return scaling_factor * weight
    def total_energy(matching):
        """
        There is a less idiotic way to do it
        I don't care
        """
        total = 0
        for x, y in matching:
            total += energy(x, y)
        return total
    return energy, total_energy

def search_annealing(src_net, tgt_net, biggest_matching, num_tries=20, num_iters=10000):
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
            energy = calc_energy(*i_tup) + calc_energy(*j_tup)
            energy_2 = calc_energy(i_tup[0], j_tup[1]) + calc_energy(j_tup[0], i_tup[1])
            if energy_2 - energy > 0: # apply annealing here when we do it
                curr_matching = apply_swap(curr_matching, swap)
        all_matchings.append((calc_total_energy(curr_matching), curr_matching))
    best_matching = min(all_matchings, key=op.itemgetter(0))[1]
    return best_matching

def generate_seeds(src_net, tgt_net, num_seeds):
    matching = generate_biggest_matching(src_net, tgt_net, num_seeds)
    print matching
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
        prod_len = len(net1.neighbors(curr_pair[0])) * len(net2.neighbors(curr_pair[0]))
        for neighbor in itertools.product(net1.neighbors(curr_pair[0]), net2.neighbors(curr_pair[1])):
            if neighbor[0] in imp_t or neighbor[1] in imp_h:
                continue
            marks[neighbor] += 1
            t2 += 1
            if t2 % 100000 == 0:
                memsize = sys.getsizeof(marks)
                print "t2 , mem size, prod_len : ", t2, memsize, prod_len
                if memsize > 1200000000:
                    new_marks = collections.defaultdict(int)
                    for key, val in marks.iteritems():
                        if val > 1:
                            new_marks[key] = val
                    del marks
                    marks = new_marks
                    print "new marks made"
            if marks[neighbor] > r:
                unused.append(neighbor)
                imp_t.add(neighbor[0])
                imp_h.add(neighbor[1])
                continue
        used.append(curr_pair)
    return used

def noisy_seeds(net1, net2, seeds, r):
    marks = collections.defaultdict(int)
    imp_t, imp_h = set(), set()
    unused, used, matches = set(seeds[:]), set(), set()
    def add_neighbor_marks(pair):
        print "pair: ", pair
        ct = 0
        for neighbor in itertools.product(net1.neighbors(pair[0]), net2.neighbors(pair[1])):
            if neighbor[0] in imp_t or neighbor[1] in imp_h:
                continue
            marks[neighbor] += 1
            if marks[neighbor] >= r:
                matches.add(neighbor)
                imp_t.add(neighbor[0])
                imp_h.add(neighbor[1])
    print "begin stage 0"
    while unused:
        t2 = 0
        curr_pair = unused.pop()
        add_neighbor_marks(curr_pair)
        used.add(curr_pair)
    match_diff = matches - used
    print "begin stage 1"
    while match_diff:
        curr_pair = match_diff.pop()
        used.add(curr_pair)
        add_neighbor_marks(curr_pair)
        match_diff = matches - used
        print len(match_diff)
    return list(matches)

def net_degree_dist(net1, net2, pair):
    return abs(net1.degree(pair[0]) - net2.degree(pair[1]))

def expand_when_stuck(net1, net2, seeds):
    marks_dict = collections.defaultdict(int)
    marks_sorted = sortedlist(key=lambda x: -x[1])
    imp_t, imp_h = set(), set()
    unused, used, matches = set(seeds[:]), set(), set()
    net_curried = lambda x: net_degree_dist(net1, net2, x)
    def add_neighbor_marks(pair):
        print "pair: ", pair
        for neighbor in itertools.product(net1.neighbors(pair[0]), net2.neighbors(pair[1])):
            if neighbor[0] in imp_t or neighbor[1] in imp_h:
                continue
            if marks_dict[neighbor] > 0:
                marks_sorted.remove((neighbor, marks_dict[neighbor]))
            marks_dict[neighbor] += 1
            marks_sorted.add((neighbor, marks_dict[neighbor]))
    print "begin stage 0"
    while unused:
        print unused
        for curr_pair in unused:
            add_neighbor_marks(curr_pair)
            used.add(curr_pair)
        print "begin stage 1"
        while len(marks_sorted) and marks_sorted[0][1] >= 2:
            cand_pairs = [scored_mark[0] for scored_mark in marks_sorted[0:100] if scored_mark[0][0] not in imp_t and scored_mark[0][1] not in imp_h]
            if not cand_pairs:
                break
            cand_pairs = sorted(cand_pairs, key=net_curried)
            curr_pair = cand_pairs[0]
            imp_t.add(curr_pair[0])
            imp_h.add(curr_pair[1])
            del marks_dict[curr_pair]
            matches.add(curr_pair)
            add_neighbor_marks(curr_pair)
            used.add(curr_pair)
        unused = [x for x in set(marks_dict.keys()) - used if x[0] not in imp_t and x[1] not in imp_h]
        print len(unused)
    return list(matches)

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

def read_recdata_sample(sample=4000, offset=0):
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

def generate_rtg_words(length):
    """
    Generates the words for the RTG
    """
    probs = np.array([0.2, 0.1, 0.1, 0.3, 0.3])
    choices = map(int, list(npr.choice(5, length, p=probs)))
    members = "abcd "
    return "".join(members[choice] for choice in choices)

def wash_words(words):
    """
    Takes a sequence that can be split and turns it into a bigram network
    """
    word_map = dict([(tup[1], tup[0]) for tup in enumerate(set(words))])
    net = nx.Graph()
    for word1, word2 in zip(words, words[1:]):
        net.add_edge(word_map[word1], word_map[word2])
    return net, word_map

# cap the length and try the extra thought
def generate_rtg(length=10000):
    """
    Generate a Random Typer Graph a la Akoglu et al
    Really rudimentary, I forget which of the various RTG versions this is
    """
    rtg_words = generate_rtg_words(length)
    return wash_words(rtg_words.split())

def generate_wordnet(filename="data/corpus.txt", num_words=2000):
    with open(filename) as corpus_file:
        corpus = corpus_file.read()
    return wash_words(corpus.split()[:num_words])

def add_dummies(net1, net2):
    dummified_net1, dummified_net2 = net1.copy(), net2.copy()
    net1_len, net2_len = len(net1.nodes()), len(net2.nodes())
    if net1_len > net2_len:
        for x in xrange(net2_len + 1, net1_len + 1):
            dummified_net2.add_node(x)
    elif net2_len > net1_len:
        for x in xrange(net1_len + 1, net2_len + 1):
            dummified_net1.add_node(x)
    return dummified_net1, dummified_net2

if __name__ == "__main__":
    random.seed(123456) #different seed :)
    wordnet_1, mapping = generate_wordnet()
    inv_mapping = {v:k for k, v in mapping.iteritems()}
    wordnet_2 = select_net(wordnet_1)
    # expando is supposed to be durable to bad seeds
    # so let's lazily have some bad seeds
    seeds = generate_biggest_matching(wordnet_1, wordnet_2, 10)
    cProfile.run("res = expand_when_stuck(wordnet_1, wordnet_2, seeds)")
    #print res
    #eq_mappings = [x for x in res if x[0] == x[1]]
    #print map(lambda x: (inv_mapping[x[0]], inv_mapping[x[1]]), eq_mappings)
    #print len(res)
    #print len(eq_mappings)
    #print len(wordnet_1.nodes())
