from scws import SCWSHarness
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# TODO: Get glove vectors of X_train & X_test
# These glove vectors should be the differences between the two words
#X_train = glove(X_train)
#X_test = glove(X_test)

regression = LogisticRegression()
regression.fit(X_train, y_test)
