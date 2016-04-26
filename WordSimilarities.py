import numpy as np
import pickle
from scipy.spatial import distance
from scipy import sparse
from sklearn.metrics import pairwise


class WordSimilarities(object):

    def __init__(self, signature_path, weight_path, sim_path):
        self.signature = self.__read_signature(signature_path)
        self.weights = self.__read_weights(weight_path)
        try:
            self.similarities = np.load(sim_path)
        except Exception as e:
            self.similarities = pairwise.cosine_similarity(self.weights)
            np.save(sim_path, self.similarities)

    def get_vector_for_word(self, word):
        return self.get_vector_for_id(self.signature.get_for_word(word))

    def get_vector_for_id(self, wid):
        return sparse.csr_matrix(self.weights[wid])

    def calculate_similarity(self, word1, word2):
        word1_v = self.get_vector_for_word(word1)
        word2_v = self.get_vector_for_word(word2)
        print(word1_v.shape)
        print(word2_v.shape)

        return self.__cos(word1_v, word2_v)

    def get_similarity(self, word1, word2):
        if not self.similarities:
            return self.calculate_similarity(word1, word2)
        return self.similarities[self.signature.get_for_word(word1)]
                                [self.signature.get_for_word(word2]

    def get_n_most_similar_words(self, word, n):
        return False

    def __read_evaluation_words(self):
        return False

    def __read_signature(self, signature_path):
        return pickle.load(open(signature_path, 'rb'))

    def __read_weights(self, weight_path):
        return np.load(weight_path)

    def __cos(self, u, v):
        return distance.cosine(u, v)

    def __calculate_similarity_matrix(self):
        return [[self.__cos(v1, v2) for v2 in self.weights]
                    for v1 in self.weights]

if __name__ == '__main__':
    sim = WordSimilarities('data/context15/signature.pickle',
                           'first_15.npy',
                           'sims.npy')
    print(sim.get_vector_for_word('man'))
    print(sim.get_vector_for_word('king'))
    print(sim.get_similarity('rabbit', 'mouse'))
