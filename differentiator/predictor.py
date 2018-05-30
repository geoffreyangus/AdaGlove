import torch
import torchtext
import data


class Predictor(object):

    def __init__(
        self, 
        corpus,
        data_path='../data/wikitext-2', 
        checkpoint='../checkpoints/model.pt',
        temperature=1.0,
        seed=224):

        assert temperature > 1e-3, \
            'temperature has to be greater than or equal to 1e-3'

        self.temperature = temperature

        # Set the random seed manually for reproducibility.
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_path = data_path
        self.corpus = corpus
        with open(checkpoint, 'rb') as f:
            self.model = torch.load(f).to(self.device)
        self.model.eval()

    def predict_candidates(self, inputs, num_candidates):
        """
        Outputs the top-k most probable outcomes given some list of words.

        Params:
            inputs          A list of words representing the left context of some word.
            num_candidates  The number of top outcomes desired as output.

        Returns:
            A list of tuples where each tuple contains 
                1.) the rank of the word (1 indexed)
                2.) the word itself
                3.) unnormalized score from language model
        """
        curr_input = torch.zeros([1, 1], dtype=torch.long).to(self.device)
        hidden  = self.model.init_hidden(1)
        with torch.no_grad():  # no tracking history
            for i in range(len(inputs) + 1):
                if i < len(inputs):
                    if inputs[i] in self.corpus.dictionary.word2idx:
                        curr_input.fill_(self.corpus.dictionary.word2idx[inputs[i]])
                    else:
                        curr_input.fill_(self.corpus.dictionary.word2idx['<unk>'])

                output, hidden = self.model(curr_input, hidden)

                if i == len(inputs):
                    word_weights = output.squeeze().div(self.temperature).exp().cpu()
                    scores, indices = torch.topk(word_weights, num_candidates)
                    candidates = []
                    for place, word_idx in enumerate(indices):
                        candidate_word = self.corpus.dictionary.idx2word[word_idx]
                        candidate = (place+1, candidate_word, scores[place])
                        candidates.append(candidate)
                    return candidates
