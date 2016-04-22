from nltk.corpus import brown
import pickle

THRESHOLD = 5

if __name__ == '__main__':
    token_frequency = {}
    for word in brown.words():
        token_frequency[word] = 1 + token_frequency.get(word, 0)

    too_infrequent_words = set([word for word, count in token_frequency.items()
                                if count < THRESHOLD])

    with open('data/infrequent_brown_words.pickle', 'wb') as f:
        pickle.dump(too_infrequent_words, f)