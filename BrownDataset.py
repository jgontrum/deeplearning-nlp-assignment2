from fuel.datasets.base import Dataset
from prepare_brown import get_corpus_and_vocabulary
import numpy as np

class BrownDataset(Dataset):

    def __init__(self, context=2, **kwargs):
        self.provides_sources = ('context', 'output')

        self.corpus, self.vocabulary = get_corpus_and_vocabulary()
        self.vocabulary_size = len(self.vocabulary)
        self.corpus_size = len(self.corpus)

        self.iteration_index = 0
        self.context = context
        super(BrownDataset, self).__init__(**kwargs)

    def __active(self):
        return self.iteration_index + self.context < self.corpus_size

    def __get_next_window(self, index):
        if not self.__active():
            return None
        index += self.context
        assert index - self.context >= 0

        before = self.corpus[index - self.context:
                             index]

        after = self.corpus[index + 1:
                            index + self.context + 1]

        return np.concatenate([before, after]), self.corpus[index]

    def num_instances(self):
        return self.corpus_size - self.context * 2

    def get_data(self, state=None, request=None):
        if not request:
            request = [self.iteration_index]
            self.iteration_index += 1
        x, y = zip(*[self.__get_next_window(i) for i in request])
        return np.array(x), np.array(y, dtype='int32')

    def get_vocabulary_size(self):
        return self.vocabulary_size

if __name__ == '__main__':
    d = BrownDataset()
    print(d.num_instances())
    print(list(d.get_data(request=range(10))))
