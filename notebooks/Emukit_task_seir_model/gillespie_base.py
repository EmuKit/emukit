# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple


from . import SIR


class GillespieBase:

    def __init__(self, model: SIR):
        """
        :param model: A SEIR model
        """
        self.model = model
        self.initial_state = self._get_initial_state()

    def _get_initial_state(self) -> np.ndarray:
        raise NotImplementedError

    def _get_state_index_infected(self) -> int:
        raise NotImplementedError

    def update_state(self) -> np.ndarray:
        """ possible updates of compartment counts """
        raise NotImplementedError

    def _current_rates(self, state: np.ndarray) -> np.ndarray:
        """
        Returns an array of the current rates of infection/recovery (1/2),
        i.e. the un-normalized probabilities for these events to occur next
        """
        raise NotImplementedError

    # implemented methods
    def _draw_next_event(self, state):
        """
        Draws which event of infection or recovery happens next
        """
        # Compute current rates and the sum thereof
        rates = self._current_rates(state)
        sum_of_rates = rates.sum()

        # draw timestep from exponential distribution
        dt = np.random.exponential(1.0 / sum_of_rates)

        # draw occurring event according to current rates
        event = np.random.choice(np.asarray([i for i in range(len(self.initial_state))], dtype=int),
                                 p=rates / sum_of_rates)

        return event, dt

    def draw_gillespie(self, max_time: float):
        """
        Draw one realisation of the gillespie simulation

        :param max_time: the maximum time
        :return: tuple of (times, compartments)
        """
        # initialize
        t = 0.
        times = np.asarray([t])
        states = self.initial_state.copy()[None, :]

        # Perform simulation
        while t < max_time:
            if self._current_rates(states[-1, :]).sum() == 0.:
                # make sure to stop when all individuals are recovered
                t = max_time
                states = np.append(states, states[-1:, :], axis=0)
                times = np.append(times, t)
                break

            # draw the event and time step
            event, dt = self._draw_next_event(states[-1, :])

            t += dt

            # Update the states and times
            states = np.append(states, (states[-1, :] + self.update_state()[event, :])[None, :], axis=0)
            times = np.append(times, t)

        recovered = (self.model.N - states.sum(axis=1))[:, None]
        return times, np.append(states, recovered, axis=1)

    def maximum_infected(self, gillespie_path: Tuple[np.array, np.ndarray]) -> Tuple:
        """
        compute the time when the maximum number of objects infected and how many

        :param gillespie_path: tuple of an array with times and and an array with states
        :return: time when maximum occurs, value of maximum
        """
        t, height = gillespie_path
        idx_of_max_time = np.argmax(height[:, self._get_state_index_infected()])
        return t[idx_of_max_time], height[idx_of_max_time, self._get_state_index_infected()]

    def mean_maximum_infected(self, num_sim: int, t_end: float):
        """
        Compute the mean of the location and value of maximum across simulations

        :param num_sim: the number of samples (simulation runs)
        :param t_end: end time of simulation
        :return:
        """
        t = np.zeros((num_sim,))
        state = np.zeros((num_sim,))

        for i in range(num_sim):
            path = self.draw_gillespie(t_end)
            t[i], state[i] = self.maximum_infected(path)

        # only count the maxima where at least one infection has occurred
        idx_of_peaks = (t != 0.)
        num_of_peaks = idx_of_peaks.sum()

        return t[idx_of_peaks].sum()/num_of_peaks, state[idx_of_peaks].sum()/num_of_peaks

    def run_gillespie_simulations(self, num_sim: int, t_eval: np.ndarray) -> np.ndarray:
        """
        Draw num_sim Gillespie samples

        :param num_sim: number of samples (simulation runs)
        :param t_eval: ordered time points at which to record the current state, shape (number of time points)
        :return: num_sim samples, shape (t_eval.size, number of compartments, num_sim)
        """
        num_compartments = len(self.initial_state) + 1
        samples = np.zeros((t_eval.size, num_compartments, num_sim))

        for i in range(num_sim):
            samples[:, :, i] = self.draw_gillespie_fixed_time(t_eval)
        return samples

    def draw_gillespie_fixed_time(self, t_eval: np.ndarray):
        """
        Draw one realisation of the SIR model with outputs evaluated at given times t_eval

        :param t_eval: array of ordered time points at which to record the current state
        :type t_eval: ordered np.ndarray of shape (number of time points)

        :return: a Gillespie sample, with shape (t_eval.size, number of compartments)
        """
        path = self.draw_gillespie(t_eval[-1])
        return self._map_paths_to_times(path, t_eval)

    def _map_paths_to_times(self, gillespie_path: Tuple[np.ndarray, np.ndarray], t_eval: np.ndarray):
        """
        Maps a given sample onto a given ordered time array

        :param gillespie_path: tuple containing times and state paths
        :param t_eval: array of ordered time points at which to record the current state, shape (number of time points)
        :return: the remapped states, with shape (t_eval.size, number of compartments)
        """
        # check that start of paths coincide
        assert t_eval[0] == 0

        t_path, state_path = gillespie_path
        idx = np.searchsorted(t_path, t_eval, side='right') - 1
        state_new = state_path[idx, :]
        return state_new
