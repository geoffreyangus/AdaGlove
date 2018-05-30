import re


class TextReader(object):

        def __init__(self, text_path, regex_rules):
            self.text_path = text_path
        self.file = open(text_path, "r")
        self.regex_rules = regex_rules

    # Source: https://stackoverflow.com/questions/16922214/reading-a-text-file-and-splitting-it-into-single-words-in-python
    def read_word():
        while True:
            buf = self.file.read(10240)
            if not buf:
                break

            # make sure we end on a space (word boundary)
            while not str.isspace(buf[-1]):
                ch = self.file.read(1)
                if not ch:
                    break
                buf += ch

            words = buf.split()
            for word in words:
                yield word
        yield '' #handle the scene that the file is empty


    def get_next_sentence():
        sentence = []
        for word in self.read_word():
            sentence.append(word)
            if word == '.':
                return self.clean_sentence(sentence)
        return None


    def clean_sentence(sentence_arr):
        sentence = ' '.join(sentence_arr)
        new_text = re.sub(self.regex_rules, '', new_text)
        return new_text.split(' ')

    def __del__(self):
        self.file.close()
