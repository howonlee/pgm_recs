import collections
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    rating_cards = collections.defaultdict(set)
    user_idx = 0
    user_ids = dict()
    item_idx = 0
    item_ids = dict()
    with open("data/ratings.dat") as data_file:
        for line in data_file:
            user_id, item_id, rating, timestamp = map(float, line.split("::"))
            rating_cards[item_id].add(user_id)
            if user_id not in user_ids:
                user_ids[user_id] = user_idx
                user_idx += 1
            if item_id not in item_ids:
                item_ids[item_id] = item_idx
                item_idx += 1
    print "finished writing file"
    rating_mat = np.zeros((item_idx, user_idx))
    for item, users in rating_cards.iteritems():
        print len(users)
        item_id = item_ids[item]
        for user in users:
            user_id = user_ids[user]
            rating_mat[item_id, user_id] += 1
    plt.matshow(rating_mat)
    plt.show()
