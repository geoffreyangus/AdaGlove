from predictor import Predictor
from collections import defaultdict
from os.path import join

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
		self.glove = torchtext.vocab.GloVe(name='6B', dim=glove_dim)
		self.centroid_dict = defaultdict(list)
		self.predictor = Predictor(corpus)

	def update_centroid_dict(self, target, context):
		candidates = self.predictor.getCandidates(context, 10)

		for candidate in candidates:
			rank, word, score = candidate
			print('Candidate #{}: \"{}\" with score {}.'.format(rank, word, score))

		new_centroid = np.zeros(glove_dim)
		for candidate in candidates:
			new_centroid += self.glove[word]

		new_centroid /= len(candidates)
		is_homonym = True
		for old_centroid in self.centroid_dict[target]:
			# If it can be subsumed by another centroid's sphere of influence
			if np.linalg.norm(old_centroid, new_centroid) < self.threshold:
				is_homonym = False

		# TODO: Update logic for determining homonymy
		if is_homonym:
			centroid_dict[target].append(new_centroid)

	def train(self):
		reader = textReader(join(self.data_path, 'train'), regex_rules=r"(= +.*= +)")
		sentence = reader.get_next_sentence()
		while sentence:
			# returns sentence as list
			for i, word in enumerate(sentence):
				if word == '.':
					break
				target = word
				context = sentence[:i]
				self.update_centroid_dict(target, context)

			sentence = reader.get_next_sentence()

		print(self.centroid_dict[target])

if __name__ == '__main__':
	generator = GloVeGenerator()
	generator.train()