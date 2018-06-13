from predictor import Predictor
from collections import defaultdict
from os.path import join, dirname, realpath, splitext
import data
import torchtext
from reader import TextReader
import numpy as np
import subprocess
import multiprocessing
import argparse
import re
from scipy.sparse import csc_matrix
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from sklearn.cluster import KMeans

STOP_WORDS = [',', '.', '*', '\\', '(', ')', '|', '<unk>', '[', ']', '?', '+', '-']

parser = argparse.ArgumentParser(description='CS224U Final Project.')
parser.add_argument('--in_file', required=True, help='the dataset from which to pull text')

class GloVeGenerator(object):
    def __init__(
        self,
        data_path='../data/wikitext-2',
        regex_rules=r"(= +.*= +)",
        threshold=200,
        glove_dim=50
    ):
        #print('Initializing GloVeGenerator...')
        self.data_path = data_path
        self.regex_rules = regex_rules
        self.threshold = threshold
        self.glove_dim = glove_dim
        #print('Initializing dataset corpus...')
        self.corpus = data.Corpus(data_path)
        self.glove_dim = glove_dim
        self.centroid_dict = defaultdict(list)
<<<<<<< Updated upstream
        print('Initializing language model...')
        self.predictor = Predictor(self.corpus, checkpoint='../checkpoints/model_87.pt')
=======
        #print('Initializing language model...')
        self.predictor = Predictor(self.corpus)
>>>>>>> Stashed changes

        N = len(self.corpus.dictionary.word2idx)
        self.cooccurence_counts = csc_matrix((N,N))

        self.homonym_count = self.get_homonyms_count()

        target_path = dirname(realpath(__file__))
        # Find parent directory agnostic of current location
        offset = target_path.find('AdaGlove') + len('AdaGlove')
        self.glove_path = target_path[:offset] + '/GloVe/'

    def get_homonyms_count(self):
        homonym_count = defaultdict(int)

        for word, index in tqdm(self.corpus.dictionary.word2idx.items()):
            homonym_count[word] = len(set([b for a in wn.synsets(word) for b in a.hypernyms()]))

        return homonym_count

    def read_glove(self, vector_file):
        glove_dict = {}
        with open(join(self.glove_path, vector_file + '.txt'), 'r') as f:
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
        subprocess.call([join(self.glove_path, 'demo.sh'), 'python', glove_corpus_path, self.glove_path, splitext(vector_file)[0], str(self.glove_dim)])
        return self.read_glove(vector_file)

    def parse_word(self, example):
        target, context = example
        if target == '<unk>':
            return
        if target not in self.corpus.dictionary.word2idx:
            return

        # word -> id
        target_id = self.corpus.dictionary.word2idx[target]

        candidates = self.predictor.predict_candidates(context, 10)

        for candidate in candidates:
            rank, word, score = candidate
            if word not in self.glove_dict.keys():
                continue

            # word -> id
            word_id = self.corpus.dictionary.word2idx[word]

            self.cooccurence_counts[target_id, word_id] += 1

    def update_centroid_dict(self, example):
        target, context = example
        if target == '<unk>':
            return target
        if target not in self.kmeans_clusters:
            #print("not in clusters")
            return target

        candidates = self.predictor.predict_candidates(context, 10)
        #print('candidates', candidates)

        candidate_embeddings = [
            self.glove_dict[word]
            for id, word, score in candidates
            if word in self.glove_dict
        ]

        if len(candidate_embeddings) == 0:
            return target

        predictions = self.kmeans_clusters[target].predict(candidate_embeddings)

        counts = np.bincount(predictions)
        meaning_id = np.argmax(counts)

        target += str(meaning_id)

        # new_centroid = np.zeros(self.glove_dim)
        # for candidate in candidates:
        #     rank, word, score = candidate
        #     if word not in self.glove_dict.keys():
        #         continue
        #
        #     # #print('Candidate #{}: \"{}\" with score {}.'.format(rank, word, score))
        #     new_centroid += self.glove_dict[word]
        #
        # new_centroid /= len(candidates)
        # old_centroids = self.centroid_dict[target]
        # is_homonym = True
        # for i, old_centroid in enumerate(old_centroids):
        #     # If another centroid exists close in meaning, assign this word to the same meaning.
        #     if np.linalg.norm(old_centroid - new_centroid) < self.threshold:
        #         is_homonym = False
        #         target += str(i)
        #
        # # TODO: Update logic for determining homonymy
        # if is_homonym:
        #     target += str(len(old_centroids))
        #     old_centroids.append(new_centroid)

        return target

    def cluster_words(self):
        # word -> k_means trained on dataset
        cluster_words = {}

        # words that have actual values
        found_words = set(np.nonzero(self.cooccurence_counts)[0])
        #print("Found", found_words)

        for word_id in found_words:
            word = self.corpus.dictionary.idx2word[word_id]

            clusters = max(self.homonym_count[word], 1)

            # Find neighbors
            neighbor_ids = np.nonzero(self.cooccurence_counts[word_id])[1]
            neighbor_words = [self.corpus.dictionary.idx2word[id] for id in neighbor_ids]
            #print("WORD", neighbor_words)
            neighbor_vecs = [self.glove_dict[word] for word in neighbor_words if word in self.glove_dict]
            #print("Vecs", neighbor_vecs)

            if len(neighbor_vecs) == 0:
                continue

            clusters = min(clusters, len(neighbor_vecs))

            #print("Will fit", clusters)

            # Get dictionary entries for this word
            kmeans = KMeans(n_clusters=clusters)
            kmeans.fit(neighbor_vecs)
            cluster_words[word] = kmeans

        #print("Cluster words", cluster_words)
        return cluster_words

    def fit(self, in_file, out_file):
<<<<<<< Updated upstream
        print('Fitting to {}...'.format(join(self.data_path, in_file)))
        print('Initializing GloVe vectors...')
        self.glove_dict = self.init_glove(in_file, 'vectors')
=======
        #print('Fitting to {}...'.format(join(self.data_path, in_file)))
        #print('Initializing GloVe vectors...')
        self.glove_dict = self.init_glove(in_file, 'vectors.txt')
>>>>>>> Stashed changes
        reader = TextReader(join(self.data_path, in_file), regex_rules=r"(= +.*= +)")

        # Read the first time to get a sense of the neighbors of words
        for sentence in tqdm(reader.get_next_sentence(), total=reader.sentences_count()):
            contexts = [sentence[:i] for i in range(len(sentence))]
            examples = list(zip(sentence, contexts))

            for example in examples:
                self.parse_word(example)

        # Perform K-means clustering to get the proposed new embeddings of each word
        # This lets us assign a meaning ID to each word
        self.kmeans_clusters = self.cluster_words()

        # Read the second time to get the true word usage in the specific
        # contexts
        reader.reset()
        with open(join(self.data_path, out_file), 'w') as f:
            for sentence in tqdm(reader.get_next_sentence(), total=reader.sentences_count()):
                contexts = [sentence[:i] for i in range(len(sentence))]
                examples = list(zip(sentence, contexts))
                # target word modified to reflect centroid assignment
                outwords = list(map(self.update_centroid_dict, examples))
                #print(outwords)
                for outword in outwords:
                    f.write(outword + ' ')

        self.init_glove(out_file, 'new_vectors.txt')

    def generate_regex_pattern(self, word_list):
        if len(word_list) == 1:
            return r"{}[0-9]+".format(word_list[0])

        pattern = r'('
        pattern += word_list[0] if word_list[0] not in STOP_WORDS else "<unk>"
        for word in word_list[1:]:
            if word not in STOP_WORDS:
                pattern += '|' + word

        pattern += ')' + '[0-9]+'
        return pattern

    def set_glove_file(self, vector_file):
        self.vector_file = vector_file

    def find_nearest_semantic_neighbor(self, word, context):
        homonym_pattern = self.generate_regex_pattern(word)
        glove_dict = self.read_glove(self.vector_file)

        candidates = self.predictor.predict_candidates(context, 10)
        candidate_pattern = self.generate_regex_pattern([candidate[1] for candidate in candidates])
<<<<<<< Updated upstream
        # TODO: retrain language model on the new dataset. For now we are averaging all centroids.
        candidates = np.array([vec for key, vec in glove_dict.items() if re.search(candidate_pattern, key)])
=======
        #print(candidates)
        #print(candidate_pattern)
        # TODO: retrain language model on the new dataset. For now we are averaging all centroids.
        candidates = np.array([vec for key, vec in glove_dict.items() if re.search(candidate_pattern, key)])
        #print(candidates.shape)
        #print(candidates)
>>>>>>> Stashed changes
        candidate_centroid = np.average(candidates, axis=0)

        homonyms = np.array([vec for key, vec in glove_dict.items() if re.search(homonym_pattern, key)])
        # Broadcast candidate_centroid to get a difference matrix, then calculate length
        result_idx = np.argmin(np.linalg.norm(homonyms - candidate_centroid))
        result = homonyms[result_idx]
        return result

    def predict(self, X):
        results = []
        for x in X:
            word1, context1, word2, context2 = x
            #print('word1...', word1)
            #print('word2...', word2)
            if word1 not in self.corpus.dictionary.word2idx.keys() or word2 not in self.corpus.dictionary.word2idx.keys():
<<<<<<< Updated upstream
                results.append((None, None))
                print(word1, 'or', word2, 'not in corpus. Skipping...')
=======
                results.append((np.zeros(self.glove_dim), np.zeros(self.glove_dim)))
                #print(word1, 'or', word2, 'not in corpus. Skipping...')
>>>>>>> Stashed changes
                continue
            print('=' * 89)
            result1 = self.find_nearest_semantic_neighbor(word1, context1)
            result2 = self.find_nearest_semantic_neighbor(word2, context2)
            results.append((result1, result2))

        return results

if __name__ == '__main__':
    args = parser.parse_args()
    in_file = args.in_file
    out_file = 'out_' + in_file
    generator = GloVeGenerator()
    generator.fit(in_file, out_file)
