import pytest
import pgm

class TestPGM():
    def test_select_net(self, net, p=0.9):
        pass

    def test_l2_norm(self, x, y):
        pass

    def test_generate_biggest_matching(self, src_net, tgt_net, num_seeds):
        pass

    def test_generate_possible_swap(self, matching):
        pass

    def test_apply_swap(self, curr_matching, swap):
        pass

    def test_generate_matching_neighbors(self, matching, num_neighbors=20):
        pass

    def test_cos_dist(self, fst, snd):
        pass

    def test_cosine_mat(self, net):
        pass

    def test_energy_diff_wrapper(self, src_net, tgt_net):
        # test pair_dist
        # test energy
        # test total_energy

    def test_search_annealing(self, src_net, tgt_net, biggest_matching, num_tries=5, num_iters=10000):
        pass

    def test_generate_seeds(self, src_net, tgt_net, num_seeds=50):
        pass

    def test_normal_pgm(self, net1, net2, seeds, r): #seeds is a list of tups
        pass

    def test_expand_once_pgm(self, net1, net2, seeds, r):
        pass

    def test_expando_pgm(self, net1, net2, seeds, r): #seeds is a list of tups
        pass

    def test_generate_skg_arr(self, order=11):
        pass

    def test_generate_skg_net(self, order=11):
        pass

    def test_read_small_data(self, sample=4000, offset=0):
        pass

    def test_generate_rtg_words(self, length):
        pass

    def test_wash_words(self, letters):
        pass

# cap the length and try the extra thought
    def test_generate_rtg(self, length=10000):
        pass
