import numpy as np

class ObservationReader:

    def read_observations(self, filename:str) -> np.ndarray:
        with open(filename, 'r') as f:
            text_observations = map(lambda x: x.strip(), f.readlines())
        result = np.array([[int(x) for x in row] for row in text_observations])
        return result