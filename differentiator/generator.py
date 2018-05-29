import torch
import torchtext
import data


class textGenerator(object):

    def __init__(
        self, 
        data_path='../data/wikitext-2', 
        checkpoint='../model/model.pt',
        temperature=1.0,
        seed=224):

        assert temperature > 1e-3, \
            'temperature has to be greater than or equal to 1e-3'

        # Set the random seed manually for reproducibility.
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_path = data_path
        self.corpus = data.Corpus(data_path)

        with open(checkpoint, 'rb') as f:
            self.model = torch.load(f).to(self.device)
        self.model.eval()
        hidden = self.model.init_hidden(1)

    def getCandidates(inputString, numCandidates):
        inputs = inputString.split(' ')
        curr_input = torch.zeros([1, 1], dtype=torch.long).to(self.device)

        with torch.no_grad():  # no tracking history
            for i in range(len(inputs) + 1):
                if i < len(inputs) - 1:
                    curr_input.fill_(self.corpus.dictionary.word2idx[inputs[i]])
                    print(inputs[i])

                output, hidden = model(curr_input, hidden)

                if i == len(inputs):
                    word_weights = output.squeeze().div(self.temperature).exp().cpu()
                    scores, indices = torch.topk(word_weights, numCandidates)
                    candidates = []
                    for place, word_idx in enumerate(indices):
                        candidate_word = lower(self.corpus.dictionary.idx2word[word_idx])
                        candidate = (place+1, candidate_word, scores[place], self.corpus.glove[candidate_word])
                        candidates.append(candidate)
                    return candidates
