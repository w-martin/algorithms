from typing import Dict

import numpy as np


class PriorReader:

    def read_priors(self, filename: str, prior_map: Dict[str, float]) -> np.ndarray:
        text_priors = None
        with open(filename, 'r') as f:
            text_priors = list(map(lambda x: x.strip(), f.readlines()))
        result = np.empty(shape=(len(text_priors), max(len(x) for x in text_priors)), dtype=float)
        for row in range(len(text_priors)):
            text_prior_row = text_priors[row]
            for column in range(len(text_prior_row)):
                result[row][column] = prior_map[text_prior_row[column]]
        return result
