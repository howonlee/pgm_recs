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
    assert len(selected.nodes()) == 100 # no nodes taken
    assert len(selected.edges()) < 100

def test_l2_norm():
    scalar_res = pgm.l2_norm(2, 2)
    assert abs(scalar_res - 2.8284271247) < 0.00001
