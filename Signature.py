class Signature(object):

    def __init__(self):
        self.word_to_int = {}
        self.int_to_word = []

    def add_word(self, word):
        word_id = self.word_to_int.get(word, None)

        if not word_id:
            word_id = len(self.int_to_word)
            self.int_to_word.append(word)
            self.word_to_int[word] = word_id

        return word_id

    def get_for_id(self, id):
        return self.int_to_word[id] if len(self.int_to_word) > id else None

    def get_for_word(self, word):
        return self.word_to_int.get(word, None)

    def get_vocabulary_size(self):
        return len(self.int_to_word)
