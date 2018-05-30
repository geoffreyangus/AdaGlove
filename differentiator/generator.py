from predictor import Predictor
from collections import defaultdict
from os.path import join, dirname, realpath
import data
import torchtext
from reader import TextReader
import numpy as np
import subprocess

class GloVeGenerator(object):
    def __init__(
        self, 
        data_path='../data/wikitext-2', 
        regex_rules=r"(= +.*= +)", 
        threshold=200, 
        glove_dim=100
    ):
        self.data_path = data_path
        self.regex_rules = regex_rules
        self.threshold = threshold
        self.glove_dim = glove_dim

        self.corpus = data.Corpus(data_path)
        self.glove_dim = glove_dim
        self.glove = self.init_glove('train.txt', 'vectors')
        self.centroid_dict = defaultdict(list)
        self.predictor = Predictor(self.corpus)

    def init_glove(self, text_path, vector_path):
        target_path = dirname(realpath(__file__))
        # Find parent directory agnostic of current location
        offset = target_path.find('AdaGlove') + len('AdaGlove')
        target_path = target_path[:offset] + '/GloVe/'
        ext_loc = self.data_path.rfind('.')
        glove_corpus_name = text_path[ext_loc:] + '_no_unk' + text_path[:ext_loc]
        glove_corpus_path = join(self.data_path, glove_corpus_name)
        reader = TextReader(join(self.data_path, text_path), regex_rules=None)
        reader.preprocess_text(glove_corpus_path, rules={'remove': ['<unk>']})
        # Cannot do this call from within a different folder than demo because of path dependencies
        subprocess.call(['../GloVe/demo.sh', 'python', glove_corpus_path, target_path, vector_path, self.glove_dim])

    def update_centroid_dict(self, target, context):
        candidates = self.predictor.predict_candidates(context, 10)

        for candidate in candidates:
            rank, word, score = candidate
            print('Candidate #{}: \"{}\" with score {}.'.format(rank, word, score))

        new_centroid = np.zeros(self.glove_dim)
        for candidate in candidates:
            new_centroid += self.glove[word]

        new_centroid /= len(candidates)
        old_centroids = self.centroid_dict[target]
        is_homonym = True
        for i, old_centroid in enumerate(old_centroids):
            # If another centroid exists close in meaning, assign this word to the same meaning.
            if np.linalg.norm(old_centroid - new_centroid) < self.threshold:
                is_homonym = False
                target += str(i)

        # TODO: Update logic for determining homonymy
        if is_homonym:
            target += str(len(old_centroids))
            old_centroids.append(new_centroid)

        return target

    def train(self, outfile):
        reader = TextReader(join(self.data_path, 'train.txt'), regex_rules=r"(= +.*= +)")
        sentence = reader.get_next_sentence()

        with open(join(self.data_path, outfile)) as f:
            while sentence:
                # returns sentence as list
                for i, word in enumerate(sentence):
                    if word == '.':
                        break
                    target = word
                    context = sentence[:i]
                    # target word modified to reflect centroid assignment
                    outword = self.update_centroid_dict(target, context)
                    f.write(outword + ' ')

                sentence = reader.get_next_sentence()

        self.init_glove(outfile, 'new_vectors')

if __name__ == '__main__':
    generator = GloVeGenerator()
    generator.train()
