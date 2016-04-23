from nltk.corpus import brown
from Signature import Signature
import numpy as np
import pickle

THRESHOLD = 5


def get_corpus_and_vocabulary():
    with open('data/corpus.pickle', 'rb') as f:
        return pickle.load(f)


def read_corpus():
    sig = Signature()
    token_frequency = {}

    # Build the word to int mapping and count frequencies
    for word in brown.words():
        word_id = sig.add_word(word)
        token_frequency[word_id] = 1 + token_frequency.get(word_id, 0)

    too_infrequent_words = set([word for word, count in token_frequency.items()
                                if count < THRESHOLD])

    # Create the actual corpus
    corpus = []
    for word in brown.words():
        word_id = sig.get_for_word(word)
        if word_id not in too_infrequent_words:
            corpus.append(word_id)

    return (np.asarray(corpus, dtype='int32'),
            np.asarray(sig.int_to_word),
            sig)


if __name__ == '__main__':
    signature_file = open('data/signature.pickle', 'wb')
    corpus_file = open('data/corpus.pickle', 'wb')

    corpus, vocabulary, signature = read_corpus()
    print(corpus[20])
    print(len(corpus))

    pickle.dump((corpus, vocabulary), corpus_file)
    pickle.dump(signature, signature_file)

    signature_file.close()
    corpus_file.close()
