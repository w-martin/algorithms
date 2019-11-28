from typing import Tuple

from hmm.observation_reader import ObservationReader
from hmm.prior_reader import PriorReader
import numpy as np


def main():
    prior_reader = PriorReader()
    priors_list = [prior_reader.read_priors('resources/creaks.txt', {'0': 0.1, 'x': 0.9}),
    prior_reader.read_priors('resources/bumps.txt', {'0': 0.1, 'x': 0.9})]
    priors = np.dstack(priors_list)
    observations = ObservationReader().read_observations('resources/observations.txt')
    print(f'Priors: {priors.shape}')
    print(f'Observations: {observations.shape}')
    viterbi(priors, observations)


def create_init_p(shape: Tuple[int]) -> np.ndarray:
    result = np.zeros(shape)
    result[0, :] = 1
    result[-1, :] = 1
    result[:, 0] = 1
    result[:, -1] = 1
    result /= np.sum(result)
    return result


def create_trans_p_masks(shape):
    base_mask = np.zeros(1 + np.array(shape))
    for i in range(0, 3):
        for j in range(0, 3):
            base_mask[i, j] = 1
    base_mask[1, 1] = 0
    base_mask /= np.sum(base_mask)
    trans_p_masks = np.zeros(list(shape) + list(shape))
    for i in range(shape[0]):
        for j in range(shape[1]):
            this_mask = np.roll(base_mask, (j, i))[1:, 1:]
            trans_p_masks[i, j] = this_mask
    i = 1



def viterbi(priors: np.ndarray, observations: np.ndarray):
    obs_p_sound = np.max(np.multiply(priors, observations[:, np.newaxis, np.newaxis, :]), axis=3)
    obs_p_no_sound = np.max(np.multiply(1 - priors, 1 - observations[:, np.newaxis, np.newaxis, :]), axis=3)
    base_p = np.tile(0.1, obs_p_sound.shape)
    obs_p = np.max([obs_p_sound, base_p], axis=0)
    init_p = create_init_p(priors[:, :, 1].shape)
    trans_p = init_p[np.newaxis]
    trans_p_masks = create_trans_p_masks(priors[:, :, 1].shape)
    indices = []
    probabilities = []
    for i, obs_p_sound_step in enumerate(obs_p_sound):
        step_p = trans_p * obs_p_sound_step
        mask = step_p > 0
        continuation_indices = np.argwhere(mask)
        indices += [continuation_indices[:, 1:]]
        probabilities += [step_p[mask]]
        trans_p = trans_p_masks[indices[-1]]
        i = 2
    i = 1



if __name__ == '__main__':
    main()