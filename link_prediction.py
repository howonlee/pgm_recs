import numpy as np
import networkx as nx
import pgm

def make_prediction(matching, sampled_net, ansatz_net):
    print matching
    return []
    #predictions = []
    #neighbors is something
    #matching_set = set([x for _, x in matching])
    #for sample_elem, ansatz_elem in matching:
    #    if ansatz elems is in matching_set:
    #        add the corresponding elements to predictions
    #return predictions

def predict_with_ansatz(sampled_net):
    ansatz_net = pgm.skg_like(sampled_net)
    seed = pgm.get_biggest_matchings(sampled_net, ansatz_net)
    matching = pgm.expand_with_something(sampled_net, ansatz_net, seed)
    return make_predictions(matching, sampled_net, ansatz_net)

def print_acc(predictions, sampled_net, orig_net):
    """
    This is a bullshit accuracy, oh well
    """
    raise NotImplementedError()

if __name__ == "__main__":
    orig_net = nx.read_edge_list("fb_net.txt")
    sampled_net = pgm.sample_net(orig_net)
    net_predictions = predict_with_ansatz(sampled_net)
    print_acc(net_predictions, sampled_net, orig_net)
    #if not, then see why, be durable, look real hard at the original 2009 paper
