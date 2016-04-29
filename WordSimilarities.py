import numpy as np
import pickle
from scipy.spatial import distance
from pprint import pprint
import sys


class WordSimilarities(object):

    def __init__(self, signature_path, weight_path, eval_path):
        self.signature = self.__read_signature(signature_path)
        self.weights = self.__read_weights(weight_path)
        self.eval = self.__read_evaluation_words(eval_path)
        self.sim_cache = {}

    def evaluate_n_best(self):
        results = []
        for w1, w2, eval1, eval2 in self.eval:
            try:
                v1 = self.get_vector_for_word(w1)
                v2 = self.get_vector_for_word(w2)
                ev1 = self.get_vector_for_word(eval1)

                distance = v2 - v1
                expected_vector = ev1 + distance
                nearest_words = self.get_n_most_similar_words_for_vector(
                    expected_vector, -1)

                # pprint(nearest_words)

                results.append(nearest_words.index(eval2))
            except ValueError:
                results.append(-1)
            print("{0}-{1} = {2}-{3}: {4}".format(w1, w2, eval1, eval2,
                                                  results[-1]))
        return results

    def get_vector_for_word(self, word):
        wid = self.signature.get_for_word(word)
        if wid is None:
            raise ValueError("Word '%s' not found." % word)
        return self.get_vector_for_id(wid)

    def get_vector_for_id(self, wid):
        return self.weights[wid].astype('float64')

    def get_similarity(self, word1, word2):
        word1_v = self.get_vector_for_word(word1)
        word2_v = self.get_vector_for_word(word2)
        return distance.cosine(word1_v, word2_v)

    def get_distance(self, u, v):
        return distance.cosine(u, v)

    def get_n_most_similar_words_for_vector(self, vector, n):
        sim_words = []
        for other_word in self.signature.word_to_int.keys():
            other_v = self.get_vector_for_word(other_word)
            sim = self.get_distance(vector, other_v)
            sim_words.append((sim, other_word))
        sims, words = zip(*sorted(sim_words,
                                  key=lambda x: x[0],
                                  reverse=False)[:n])
        return words

    def get_n_most_similar_words_for_word(self, word, n):
        word_v = self.get_vector_for_word(word)
        return self.get_n_most_similar_words_for_vector(word_v, n)

    def __read_evaluation_words(self, path):
        eval_words = []
        for line in open(path, 'r'):
            if not line.startswith("//") and line[0] != ":":
                assert len(line.split()) == 4
                eval_words.append(line.strip().split())
        return eval_words

    def __read_signature(self, signature_path):
        return pickle.load(open(signature_path, 'rb'))

    def __read_weights(self, weight_path):
        return np.load(weight_path)

if __name__ == '__main__':
    sig = sys.argv[1]
    weights = sys.argv[2]
    res = sys.argv[3]

    sim = WordSimilarities(sig,
                           weights,
                           'eval/word-test.v1')

    f = open(res, "w")
    for i, idx in enumerate(sim.evaluate_n_best()):
        f.write("%s,%s" % (i, idx))
    f.close()
