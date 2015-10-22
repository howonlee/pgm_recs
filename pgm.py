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

def wash_words(word_ls):
    word_map = {}
    curr_idx = 0
    for word in word_ls:
        if word not in word_map:
            word_map[word] = curr_idx
            curr_idx += 1
    return [word_map[word] for word in word_ls], word_map

def word_net(corpus_ls):
    """
    Takes a list of numbers
    Do the number map if you want the words back
    In this case, usually don't want the words back
    """
    bigrams = zip(corpus_ls, corpus_ls[1:])
    net = nx.Graph()
    net.add_edges_from(bigrams)
    return net

def select_net(net, p=0.9):
    new_net = nx.Graph()
    edge_list = net.edges()
    new_net_edges = []
    for edge in edge_list:
        if random.random() < p:
            new_net_edges.append(edge)
    new_net.add_edges_from(new_net_edges)
    return new_net

def print_seed_dist(src_degs, tgt_degs, num_seeds):
    for x in xrange(num_seeds):
        print src_degs[x][1] - tgt_degs[x][1]

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

def pgm(net1, net2, seeds, r): #seeds is a list of tups
    marks = collections.defaultdict(int)
    #heap?
    imp_1 = {} #impossible tails
    imp_2 = {} #impossible heads
    unused = seeds[:]
    used = []
    #to make it online algo: unused should be of high priority, but used should also be taken fron
    while unused:
        t2 = 0
        curr_pair = unused.pop(random.randint(0,len(unused)-1))
        for neighbor in itertools.product(net1.neighbors(curr_pair[0]), net2.neighbors(curr_pair[1])):
            #take the filter out of the loops
            if imp_1.has_key(neighbor[0]) or imp_2.has_key(neighbor[1]):
                continue
            marks[neighbor] += 1
            t2 += 1
            if t2 % 1000 == 0:
                #print t2
                pass
            if t2 % 250000 == 0:
                #this is an awful hack
                break
            #take it out, I guess?
            if marks[neighbor] > r:
                unused.append(neighbor)
                imp_1[neighbor[0]] = True
                imp_2[neighbor[1]] = True
        #maximum of the marks here, but later
        used.append(curr_pair)
    return used

def generate_skg(order=11):
    gen = np.array([[0.99, 0.8], [0.8, 0.3]])
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

def score_easy(pgm_res):
    #only works for the choicing model
    #not on real dataset, in other words
    corrects = len([x for x in pgm_res if x[0] == x[1]])
    print "correct / total: ", corrects, " / ", len(pgm_res)

if __name__ == "__main__":
    random.seed(123456) #different seed :)
    net = generate_skg()
    src_net, tgt_net = select_net(net, p=0.9), select_net(net, p=0.9)
    seeds = get_seeds(src_net, tgt_net, 200)
    print len(src_net.edges()), len(tgt_net.edges())
    print len(pgm(src_net, tgt_net, seeds, 3))
