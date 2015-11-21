import pytest
import pgm
import networkx as nx

@pytest.fixture
def ex_net():
    # idiot line graph
    net = nx.Graph()
    net.add_edges_from(zip(range(100), range(1,101)))
    return net

def test_select_net(ex_net):
    selected = pgm.select_net(ex_net)
    assert len(selected.nodes()) == 101 # no nodes taken
    assert len(selected.edges()) < 100

def test_l2_norm():
    scalar_res = pgm.l2_norm(2, 2)
    assert abs(scalar_res - 2.8284271247) < 0.00001

def test_cos_dist():
    first_cat = [1,2,3,4,5,5,5,5]
    second_cat = [-1,0,1,2,2,2,2]
    assert abs(pgm.cos_dist(first_cat, second_cat) - 0.267261) < 0.00001

def test_generate_biggest_matching():
##############33
##############33
##############33
    pass

def test_generate_possible_swap():
##############33
##############33
##############33
    pass

def test_apply_swap():
##############33
##############33
##############33
    pass
