from scws import SCWSHarness
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from generator import GloVeGenerator

harness = SCWSHarness()

# Generate the X/y pairs of our dataset
X = [
    (
        ex.word1,
        harness.get_context(ex, word="word1"),
        ex.word2,
        harness.get_context(ex, word="word2")
    ) for ex in harness.dataset]
y = [int(float(ex.avg_rating) > 5) for ex in harness.dataset]

# Split our dataset
# We want to train on only a small part of our data, so we leave a high success
# rate to evaluate on our scws task
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)

model = GloVeGenerator()
model.set_glove_file('new_vectors')

# Get homonym embeddings of X_train & X_test
X_train = model.predict(X_train)
X_test = model.predict(X_test)

train_idxs = []
num_removed_train = 0
for i, val in enumerate(X_train):
    if type(val[0]) == np.ndarray:
        train_idxs.append(i)
    else:
        num_removed_train += 1

test_idxs = []
num_removed_test = 0
for i, val in enumerate(X_test):
    if type(val[0]) == np.ndarray:
        test_idxs.append(i)
    else:
        num_removed_test += 1
# These feature vectors passed to our logistic regression should be the difference
# between the two within the current embedding space

X_train = [X_train[i][0] - X_train[i][1] for i in train_idxs]
y_train = [y_train[i] for i in train_idxs]

X_test = [X_test[i][0] - X_test[i][1] for i in test_idxs]
y_test = [y_test[i] for i in test_idxs]

regression = LogisticRegression()
regression.fit(X_train, y_train)
y_predicted = regression.predict(X_test)

print("Total test examples:", len(y_test))
print("Number of examples removed:", num_removed_test)

print("Number of zero predictions:", np.sum((np.array(y_predicted) == 0)))
print("Number of one predictions:", np.sum((np.array(y_predicted) == 1)))

print(np.mean(y_predicted == y_test))
