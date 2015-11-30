import numpy as np
import networkx as nx
import pgm

def make_prediction(matching, sampled_net, ansatz_net):
    print "begin predictions..."
    predictions = {}
    matching_dict = {ansatz_elem:sample_elem for sample_elem, ansatz_elem in matching}
    for sample_elem, ansatz_elem in matching:
        ansatz_neighbors = ansatz_net.neighbors(ansatz_elem)
        sample_neighbors = [matching_dict.get(ansatz_neighbor, None) for ansatz_neighbor in ansatz_neighbors]
        sample_neighbors = filter(lambda x: x not in sampled_net.neighbors(sample_elem), sample_neighbors)
        sample_neighbors = filter(lambda x: x, sample_neighbors)
        predictions[sample_elem] = sample_neighbors
    return predictions

def predict_with_ansatz(sampled_net):
    print "making skg..."
    ansatz_net = pgm.skg_like(sampled_net)
    print "making seed...."
    seed = pgm.generate_biggest_matching(sampled_net, ansatz_net, 20)
    print "making matching...."
    matching = pgm.noisy_seeds(sampled_net, ansatz_net, seed, 3)
    return make_prediction(matching, sampled_net, ansatz_net)

def print_acc(predictions, sampled_net, orig_net):
    print predictions
    """
    This is a bullshit accuracy, oh well
    """
    raise NotImplementedError()

if __name__ == "__main__":
    orig_net = nx.Graph()
    with open("data/net.txt") as net_file:
        for line in net_file:
            fst, snd = tuple(line.split())
            orig_net.add_edge(int(fst), int(snd))
    sampled_net = pgm.select_net(orig_net)
    net_predictions = predict_with_ansatz(sampled_net)
    print_acc(net_predictions, sampled_net, orig_net)
    #if not, then see why, be durable, look real hard at the original 2009 paper
