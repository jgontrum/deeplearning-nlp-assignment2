from nltk.corpus import brown
from Signature import Signature
import numpy as np
import pickle

THRESHOLD = 5
CONTEXT = 2


def get_corpus_and_vocabulary():
    return np.load('data/corpus_x.npy'), np.load('data/corpus_y.npy')


def read_corpus():
    sig = Signature()
    token_frequency = {}

    # Build the word to int mapping and count frequencies
    for word in brown.words(categories='news'):
        word_id = sig.add_word(word)
        token_frequency[word_id] = 1 + token_frequency.get(word_id, 0)

    too_infrequent_words = set([word for word, count in token_frequency.items()
                                if count < THRESHOLD])

    # Create the actual corpus
    corpus = []
    for word in brown.words(categories='news'):
        word_id = sig.get_for_word(word)
        if word_id not in too_infrequent_words:
            corpus.append(word_id)

    x = []
    y = []

    for i in range(len(corpus)):
        c = i + CONTEXT
        if c + CONTEXT < len(corpus):
            before = corpus[c - CONTEXT: c]
            after = corpus[c + 1: c + CONTEXT + 1]

            x.append(np.array(before + after))
            y.append(corpus[c])

    return (np.array(x, dtype='int32'),
            np.array(y, dtype='int32'),
            np.asarray(sig.int_to_word),
            sig)


if __name__ == '__main__':
    signature_file = open('data/signature.pickle', 'wb')

    x, y, vocabulary, signature = read_corpus()
    print(x[20])
    print(len(x))

    np.save('data/corpus_x', x)
    np.save('data/corpus_y', y)

    pickle.dump(signature, signature_file)

    signature_file.close()
