# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from . import SIR


class GillespieBase:
    """Base class for a Gillespie simulations of a compartments model."""

    def __init__(self, model: SIR):
        """
        :param model: A SEIR model
        """
        self.model = model
        self.initial_state = self._get_initial_state()

    def _get_initial_state(self) -> np.ndarray:
        """returns the initial state"""
        raise NotImplementedError

    def _get_state_index_infected(self) -> int:
        """returns the index of the state that represents the infected"""
        raise NotImplementedError

    def _get_possible_state_updates(self) -> np.ndarray:
        """returns  possible updates of compartment counts"""
        raise NotImplementedError

    def _get_current_rates(self, state: np.ndarray) -> np.ndarray:
        """
        Returns an array of the current rates of infection/recovery (1/2),
        i.e. the un-normalized probabilities for these events to occur next
        """
        raise NotImplementedError

    def _draw_next_event(self, state: np.ndarray) -> Tuple[float, float]:
        """Draws which event of infection or recovery happens next"""
        # Compute current rates and the sum thereof
        rates = self._get_current_rates(state)
        sum_of_rates = rates.sum()

        # draw timestep from exponential distribution
        dt = np.random.exponential(1.0 / sum_of_rates)

        # draw occurring event according to current rates
        event = np.random.choice(
            np.asarray([i for i in range(len(self.initial_state))], dtype=int), p=rates / sum_of_rates
        )

        return event, dt

    def _draw_gillespie(self, time_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw one realisation of the gillespie simulation. Note that time intervals are given by the random draws of
        the state changes.

        :param time_end: the maximum time
        :return: tuple of (times, compartments) with shapes times (num_gil_times, ) and variable length and
        compartments of shape (num_gil_times, num_compartments). Note that each draw might yield a different
        num_gil_times.
        """
        # initialize
        t = 0.0
        times = np.asarray([t])
        states = self.initial_state.copy()[None, :]

        # Perform simulation
        while t < time_end:
            if self._get_current_rates(states[-1, :]).sum() == 0.0:
                # make sure to stop when all individuals are recovered
                t = time_end
                states = np.append(states, states[-1:, :], axis=0)
                times = np.append(times, t)
                break

            # draw the event and time step
            event, dt = self._draw_next_event(states[-1, :])

            # increment time that elapsed for this event to happen
            t += dt

            # Update the states and times
            states = np.append(states, (states[-1, :] + self._get_possible_state_updates()[event, :])[None, :], axis=0)
            times = np.append(times, t)

        recovered = (self.model.N - states.sum(axis=1))[:, None]
        compartments = np.append(states, recovered, axis=1)
        return times, compartments

    # fixed time steps methods
    def _draw_gillespie_fixed_time(self, times_fixed: np.ndarray) -> np.ndarray:
        """
        Draw one realisation of the SIR model with outputs evaluated at given times times_fixed. Note that the length
        of times_fixed might differ from the length of the gillespie path.

        :param times_fixed: array of ordered time points at which to sort into the gillespie compartments,
        shape (num_time_points, )
        :return: a Gillespie sample, with shape (num_time_points, num_compartments)
        """
        gillespie_path = self._draw_gillespie(times_fixed[-1])
        return self._map_gillespie_times_to_fixed_times(gillespie_path, times_fixed)

    def _map_gillespie_times_to_fixed_times(
        self, gillespie_path: Tuple[np.ndarray, np.ndarray], times_fixed: np.ndarray
    ) -> np.ndarray:
        """
        Maps a given sample onto a given ordered time array

        :param gillespie_path: tuple containing times and compartment paths
        :param times_fixed: ordered time points at which to record the current compartments, shape (num_time_points, )
        :return: the remapped compartments, with shape (num_time_points, num_compartments)
        """
        # check that start of paths coincide
        assert times_fixed[0] == 0

        times_path, compartments_path = gillespie_path
        idx_new = np.searchsorted(times_path, times_fixed, side="right") - 1
        compartments_new = compartments_path[idx_new, :]
        return compartments_new

    def run_simulation_fixed_time(self, num_gil: int, t_eval: np.ndarray) -> np.ndarray:
        """
        Draw num_gil Gillespie samples

        :param num_gil: number of samples (gillespie runs)
        :param t_eval: ordered time points at which to record the current state, shape (number of time points)
        :return: num_gil samples, shape (t_eval.size, number of compartments, num_gil)
        """
        num_compartments = len(self.initial_state) + 1
        gillespie_paths = np.zeros((t_eval.size, num_compartments, num_gil))

        for i in range(num_gil):
            gillespie_paths[:, :, i] = self._draw_gillespie_fixed_time(t_eval)
        return gillespie_paths

    # height and time occurrence of infection peak methods
    def _compute_height_and_time_of_peak(self, gillespie_path: Tuple[np.array, np.ndarray]) -> Tuple:
        """
        compute the time when the maximum number of objects infected and how many

        :param gillespie_path: tuple of an array with times and and an array with states
        :return: time when maximum occurs, value of maximum
        """
        t, height = gillespie_path
        idx_of_max_time = np.argmax(height[:, self._get_state_index_infected()])
        return t[idx_of_max_time], height[idx_of_max_time, self._get_state_index_infected()]

    def run_simulation_height_and_time_of_peak(self, num_gil: int, time_end: float):
        """
        Compute the mean of the location and value of maximum across simulations

        :param num_gil: the number of samples (simulation runs)
        :param time_end: end time of simulation
        :return:
        """
        t = np.zeros((num_gil,))
        state = np.zeros((num_gil,))

        for i in range(num_gil):
            gillespie_path = self._draw_gillespie(time_end)
            t[i], state[i] = self._compute_height_and_time_of_peak(gillespie_path)

        # only count the maxima where at least one infection has occurred
        idx_of_peaks = t != 0.0
        num_of_peaks = idx_of_peaks.sum()

        return t[idx_of_peaks].sum() / num_of_peaks, state[idx_of_peaks].sum() / num_of_peaks
