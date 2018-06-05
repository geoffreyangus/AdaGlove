from predictor import Predictor
from collections import defaultdict
from os.path import join, dirname, realpath
import data
import torchtext
from reader import TextReader
import numpy as np
import subprocess
import multiprocessing 
import argparse

parser = argparse.ArgumentParser(description='CS224U Final Project.')
parser.add_argument('--in_dir', required=True, help='the dataset from which to pull text')

class GloVeGenerator(object):
    def __init__(
        self, 
        data_path='../data/wikitext-2', 
        regex_rules=r"(= +.*= +)", 
        threshold=200, 
        glove_dim=50
    ):
        print('Initializing GloVeGenerator...')
        self.data_path = data_path
        self.regex_rules = regex_rules
        self.threshold = threshold
        self.glove_dim = glove_dim
        print('Initializing dataset corpus...')
        self.corpus = data.Corpus(data_path)
        self.glove_dim = glove_dim
        self.centroid_dict = defaultdict(list)
        print('Initializing language model...')
        self.predictor = Predictor(self.corpus)

        target_path = dirname(realpath(__file__))
        # Find parent directory agnostic of current location
        offset = target_path.find('AdaGlove') + len('AdaGlove')
        self.glove_path = target_path[:offset] + '/GloVe/'

    def read_glove(self, vector_file):
        glove_dict = {}
        with open(join(self.glove_path, vector_file), 'r') as f:
            line = f.readline()
            while line:
                line_list = line.split(' ')
                word = line_list[0]
                glove_arr = np.array([float(el) for el in line_list[1:]])
                glove_dict[word] = glove_arr
                line = f.readline()
        return glove_dict

    def init_glove(self, text_file, vector_file):
        ext_loc = text_file.rfind('.')
        glove_corpus_name = text_file[:ext_loc] + '_no_unk' + text_file[ext_loc:]
        glove_corpus_path = join(self.data_path, glove_corpus_name)
        reader = TextReader(join(self.data_path, text_file), regex_rules=None)
        reader.preprocess_text(glove_corpus_path, rules={'remove': ['<unk>']})
        # Cannot do this call from within a different folder than demo because of path dependencies
        subprocess.call([join(self.glove_path, 'demo.sh'), 'python', glove_corpus_path, self.glove_path, vector_file, str(self.glove_dim)])
        return self.read_glove(vector_file)

    def update_centroid_dict(self, example):
        target, context = example
        if target == '<unk>':
            return target

        candidates = self.predictor.predict_candidates(context, 10)

        new_centroid = np.zeros(self.glove_dim)
        for candidate in candidates:
            rank, word, score = candidate
            if word not in self.glove_dict.keys():
                continue
                
            # print('Candidate #{}: \"{}\" with score {}.'.format(rank, word, score))
            new_centroid += self.glove_dict[word]            

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

    def fit(self, in_file, out_file):
        print('Fitting to {}...'.format(join(self.data_path, in_file)))
        print('Initializing GloVe vectors...')
        self.glove_dict = self.init_glove(in_file, 'vectors.txt')
        reader = TextReader(join(self.data_path, in_file), regex_rules=r"(= +.*= +)")

        num_iters = 0
        with open(join(self.data_path, out_file), 'w') as f:
            for sentence in reader.get_next_sentence():
                contexts = [sentence[:i] for i in range(len(sentence))]
                examples = list(zip(sentence, contexts))
                # target word modified to reflect centroid assignment
                outwords = list(map(self.update_centroid_dict, examples))
                for outword in outwords:
                    f.write(outword + ' ')
                
                num_iters += 1
                if num_iters % 100 == 0:
                    print('Processed {} sentences...'.format(num_iters))

        self.init_glove(out_file, 'new_vectors')

    def generate_regex_pattern(self, word_list):
        if len(word_list):
            return r"{}[0-9]+".format(word_list[0])

        pattern = r'('
        for word in word_list[:-1]:
            pattern += word + '|'

        pattern += word_list[-1] + ')' + '[0-9]+'
        return pattern

    def set_glove_file(self, vector_file):
        self.vector_file = vector_file

    def find_nearest_semantic_neighbor(self, context, word):
        homonym_pattern = self.generate_regex_pattern(word)
        glove_dict = self.read_glove(self.vector_file)

        candidates = self.predictor.predict_candidates(left_context, 10)
        candidate_pattern = self.generate_regex_pattern([candidate[1] for candidate in candidates])
        # TODO: retrain language model on the new dataset. For now we are averaging all centroids.
        candidates = np.array([vec for key, vec in glove_dict.items() if re.search(candidate_pattern, key)])
        candidate_centroid = np.average(candidates, axis=1)

        homonyms = np.array([vec for key, vec in glove_dict.items() if re.search(homonym_pattern, key)])
        # Broadcast candidate_centroid to get a difference matrix, then calculate length
        result_idx = np.argmin(np.linalg.norm(homonyms - candidate_centroid))
        result = homonyms[result_idx]
        return result

    def predict(self, X):
        results = []
        for x in X:
            word1, context1, word2, context2 = x
            result1 = self.find_nearest_semantic_neighbor(word1, context1)
            result2 = self.find_nearest_semantic_neighbor(word2, context2)
            results.append((result1, result2))

        return results

if __name__ == '__main__':
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = 'out_' + in_dir
    generator = GloVeGenerator()
    generator.fit(in_dir, out_dir)
