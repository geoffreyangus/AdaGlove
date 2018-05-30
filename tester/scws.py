from collections import namedtuple
from os.path import dirname, realpath, join
import re

Example = namedtuple(
    "Example",
    [
        "id",
        "word1",
        "word1_pos",
        "word2",
        "word2_pos",
        "word1_context",
        "word2_context",
        "avg_rating",
        "all_ratings",
    ]
)

RATING_COUNT = 10

class SCWSHarness(object):
    def __init__(self):
        self.dataset = self.parse_dataset()

    def parse_dataset(self):
        """
        parse_dataset: Parses the ratings file that's conformant to the SCWS spec

        https://www.socher.org/index.php/Main/
            ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes
        """
        data_folder = join(dirname(realpath(__file__)), "store")
        ratings_data = join(data_folder, "ratings.txt")
        dataset = []

        with open(ratings_data, "r") as file:
            for line in file:
                components = line.strip().split("\t")
                all_body = components[:-RATING_COUNT]
                all_ratings = components[-RATING_COUNT:]

                example = Example(*all_body, all_ratings)
                dataset.append(example)

        return dataset

    def predict(self, model):
        pass

    def get_context(self, example, word="word1", direction="left"):
        """
        Gets the context surrounding a given example
        """
        context = getattr(example, f"{word}_context")
        searches = re.search("(.*)<b>(.*)<\/b>(.*)", context)

        if direction == "left":
            return searches.group(1).strip()
        elif direction == "right":
            return searches.group(3).strip()
