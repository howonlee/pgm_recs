import numpy as np
import networkx as nx
import pgm

def make_prediction(sample_elem, ansatz_elem, sampled_net, ansatz_net):
    pass

def predict_with_ansatz(sampled_net):
    ansatz_net = pgm.skg_like(sampled_net)
    seed = pgm.get_biggest_matchings(sampled_net, ansatz_net)
    matching = pgm.expand_with_something(sampled_net, ansatz_net, seed)
    predictions = []
    for sample_elem, ansatz_elem in matching:
        prediction = make_prediction(sample_elem, ansatz_elem, sampled_net, ansatz_net)
        predictions.append(prediction)
    return prediction

def print_acc(predictions):
    raise NotImplementedError()

if __name__ == "__main__":
    net = nx.read_edge_list("fb_net.txt")
    sampled_net = pgm.sample_net(net)
    net_predictions = predict_with_ansatz(sampled_net)
    print_acc(net_predictions)
    #if not, then see why, be durable, look real hard at the original 2009 paper
