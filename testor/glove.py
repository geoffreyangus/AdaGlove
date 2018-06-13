from os.path import join
import numpy as np

VSMDATA_HOME = '/Users/pierce/Documents/CS224U/cs224u/vsmdata/'
glove_home = join(VSMDATA_HOME, 'glove.6B')

def glove2dict(src_filename):
    """GloVe Reader.  From the CS224U course code.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors.

    """
    data = {}
    with open(src_filename,  encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

class GloveModel(object):
    def fit(self, X=None, y=None):
        self.glove_lookup = glove2dict(
            join(glove_home, 'glove.6B.50d.txt')
        )
        self.glove_lookup["<UNK>"] = np.random.normal(size=list(self.glove_lookup.values())[0].shape)

    def predict(self, X):
        y = []

        for x in X:
            word1, context1, word2, context2 = x

            word1_glove = self.glove_lookup.get(word1.lower(), np.zeros(50))
            word2_glove = self.glove_lookup.get(word2.lower(), np.zeros(50))

            y.append((word1_glove, word2_glove))

        return y
