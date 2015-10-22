import networkx as nx
import numpy as np
import sys
import math
import random
import collections
import matplotlib.pyplot as plt
import itertools
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
    unused, used = seeds[:], []
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
        used.append(curr_pair)
    return used

def generate_skg(order=11):
    gen = np.array([[0.99, 0.7], [0.7, 0.1]])
    skg = gen.copy()
    for x in xrange(order-1):
        skg = np.kron(skg, gen)
    skg_sample = np.zeros_like(skg)
    for x in xrange(skg.shape[0]):
        for y in xrange(skg.shape[1]):
            if random.random() < skg[x,y]:
                skg_sample[x,y] = 1
    #plt.imshow(skg_sample)
    #plt.show()
    net = nx.from_numpy_matrix(skg_sample)
    return net

if __name__ == "__main__":
    random.seed(123456) #different seed :)
    net = generate_skg()
    src_net, tgt_net = select_net(net), select_net(net)
    seeds = get_seeds(src_net, tgt_net, 40)
    print len(src_net.edges()), len(tgt_net.edges())
    res = normal_pgm(src_net, tgt_net, seeds, 5)
    print len(res)
    print len(set(res))
