from nltk.corpus import brown
from Signature import Signature
import sys
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)

THRESHOLD = 5


def main():
    if len(sys.argv) > 1:
        CONTEXT = int(sys.argv[1])
        path = sys.argv[2]
    else:
        logger.error("Missing argument: Context, Path incl. '/'")
        logger.error("Example: frameworkpython prepare_brown.py 2 data/")

    x, y, vocabulary, signature = read_corpus(CONTEXT)

    logger.info("Saving the corpus to '%s%s' and '%s%s'" % (path,
                                                            'corpus_x',
                                                            path,
                                                            'corpus_y'))
    np.save('%scorpus_x' % path, x)
    np.save('%scorpus_y' % path, y)

    logger.info("Saving the signature to '%s%s'" % (path, 'signature.pickle'))
    pickle.dump(signature, open('%ssignature.pickle' % path, 'wb'))


def get_corpus_and_vocabulary(path):
    logger.info("Loading '%scorpus_x.npy'"  % path)
    return np.load('%scorpus_x.npy' % path), np.load('%scorpus_y.npy' % path)


def read_corpus(CONTEXT):
    sig = Signature()
    token_frequency = {}

    # Build the word to int mapping and count frequencies
    logger.info("First iteration: Counting words and create signature.")
    for word in brown.words():
        word_id = sig.add_word(word)
        token_frequency[word_id] = 1 + token_frequency.get(word_id, 0)

    logger.info("Identifying infrequent words that will be ignored.")
    too_infrequent_words = set([word for word, count in token_frequency.items()
                                if count < THRESHOLD])

    # Create the actual corpus
    logger.info("Second iteration: Creating cleaned numeric corpus.")
    corpus = []
    for word in brown.words():
        word_id = sig.get_for_word(word)
        if word_id not in too_infrequent_words:
            corpus.append(word_id)

    x = []
    y = []

    # Create the windows and create data and labels
    logger.info("Third iteration: Creating context windows.")
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
    main()
