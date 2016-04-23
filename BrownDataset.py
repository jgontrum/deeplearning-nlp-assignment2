from fuel.datasets.base import Dataset
from prepare_brown import get_corpus_and_vocabulary
import numpy as np

class BrownDataset(Dataset):

    def __init__(self, context=2, **kwargs):
        self.provides_sources = ('context', 'output')

        self.corpus, self.vocabulary = get_corpus_and_vocabulary()
        self.vocabulary_size = len(self.vocabulary)
        self.corpus_size = len(self.corpus)

        self.iteration_index = context
        self.context = context
        super(BrownDataset, self).__init__(**kwargs)

    def __active(self):
        return self.iteration_index + self.context < self.corpus_size

    def __get_next_window(self):
        if not self.__active():
            return None
        assert self.iteration_index - self.context >= 0

        before = self.corpus[self.iteration_index - self.context:
                             self.iteration_index + self.context]

        after = self.corpus[self.iteration_index + 1:
                            self.iteration_index + self.context + 1]

        context = np.concatenate((before, after))

        print("context:", np.asarray(context))
        print("all:", self.corpus[self.iteration_index - self.context:
                          self.iteration_index + self.context + 1])
        print("idx:", np.asarray(context)[self.iteration_index])

        self.iteration_index += 1
        return np.asarray(context), context[0]

    def num_instances(self):
        return self.corpus_size - self.context * 2

    def get_data(self, state=None, request=None):
        return [self.__get_next_window() for i in request]

if __name__ == '__main__':
    d = BrownDataset()
    print(d.num_instances())
    print(list(d.get_data(request=range(10))))
