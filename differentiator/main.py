from generator import textGenerator

generator = textGenerator()
candidates = generator.getCandidates('I love soft fluffy teddy', 10)

for candidate in candidates:
	rank, word, score, glove = candidate
	print('Candidate #{}: \"{}\" with score {}.'.format(rank, word, score))