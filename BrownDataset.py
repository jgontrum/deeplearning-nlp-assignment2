from fuel.datasets.base import Dataset
from nltk.corpus import brown
from prepare_brown import get_infrequent_words

class BrownDataset(Dataset):
    INFREQUENT = set()

    def __init__(self, **kwargs):
        self.provides_sources = ('data',)
        self.INFREQUENT = get_infrequent_words()

        super(BrownDataset, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        return None

if __name__ == '__main__':
    d = BrownDataset()
