import collections
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rating_cards = collections.Counter()
    with open("data/u.data") as data_file:
        for line in data_file:
            user_id, item_id, rating, timestamp = map(int, line.split())
            rating_cards[user_id] += 1
    ratings = list(reversed(sorted(rating_cards.values())))
    plt.loglog(ratings)
    plt.show()
