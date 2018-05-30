import re


class TextReader(object):

    def __init__(self, text_path, regex_rules):
        self.text_path = text_path
        self.file = open(text_path, "r")
        self.regex_rules = regex_rules

    # Expand this as necessary.
    def preprocess_text(self, outfile, rules):
        with open(outfile, 'w') as f:
            if 'remove' in rules:
                for word in self.read_word():
                    if word in rules['remove']:
                        continue
                    f.write(word + ' ')
        self.file.seek(0)

    def get_next_sentence(self):
        while True:
            buf = self.file.read(1)
            if not buf:
                break
            while buf[-2:] != '. ':
                ch = self.file.read(1)
                if not ch:
                    break
                buf += ch

            sentence = buf.split()
            yield self.clean_sentence(sentence)
        yield None

    def clean_sentence(self, sentence_arr):
        sentence = ' '.join(sentence_arr)
        new_text = re.sub(self.regex_rules, '', sentence)
        return new_text.split(' ')

    def __del__(self):
        self.file.close()
