from fuel.datasets.base import Dataset
from prepare_brown import get_corpus_and_vocabulary
import logging


class BrownDataset(Dataset):

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.provides_sources = ('context', 'output')
        self.axis_labels = None

        self.corpus, self.vocabulary = get_corpus_and_vocabulary()
        self.vocabulary_size = len(self.vocabulary)
        self.corpus_size = len(self.corpus)

        self.iteration_index = 0
        super(BrownDataset, self).__init__(**kwargs)
        self.logger.info('Corpus is prepared.')

    def num_instances(self):
        return self.corpus_size

    def get_data(self, state=None, request=None):
        # self.logger.info('Requesting data for %s to %s' % (request[0],
        #                                                    request[-1] + 1))
        if not request:
            request = [self.iteration_index]
            self.iteration_index += 1

        context = self.corpus[request[0]:request[-1] + 1]
        output = self.vocabulary[request[0]:request[-1] + 1]

        # self.logger.info('Returning data...')
        return context, output

    def get_vocabulary_size(self):
        return self.vocabulary_size

if __name__ == '__main__':
    d = BrownDataset()
    print(d.num_instances())
    print(list(d.get_data(request=range(10))))
