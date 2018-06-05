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
model.set_glove_file('new_vectors.txt')

# Get homonym embeddings of X_train & X_test
X_train = model.predict(X_train)
X_test = model.predict(X_test)

# These feature vectors passed to our logistic regression should be the difference
# between the two within the current embedding space
X_train = [a - b for a, b in X_train]
X_test = [a - b for a, b in X_test]

regression = LogisticRegression()
regression.fit(X_train, y_train)
y_predicted = regression.predict(X_test)

print("All zero", np.sum((np.array(y_test) == 0)))
print("All one", np.sum((np.array(y_test) == 1)))

print(np.mean(y_predicted == y_test))
