"""
Class to manage the arrivals times of tokes in the process.
"""

import numpy as np
from datetime import datetime, timedelta
from new_rims.parameters import Parameters
from new_rims.process import SimulationProcess
import new_rims.custom_function as custom
from scipy.interpolate import interp1d

class InterTriggerTimer(object):

    def __init__(self, params: Parameters, process: SimulationProcess, start: datetime):
        self._process = process
        self._start_time = start
        self._type = params.INTER_TRIGGER['type']
        self._previous = None
        if self._type == 'distribution' or self._type == 'histogram_sampling':
            """Define the distribution of token arrivals from specified in the file json"""
            self.params = params.INTER_TRIGGER['parameters']

    def get_next_arrival(self, env, case):
        """Generate a new arrival from the distribution and check if the new token arrival is inside calendar,
        otherwise wait for a suitable time."""
        next = 0
        if self._type == 'histogram_sampling':
            bin_midpoints = self.params["bin_midpoints"]
            cdf = self.params["histogram_data"]
            inverse_cdf = interp1d(cdf, bin_midpoints, bounds_error=False,
                                   fill_value=(bin_midpoints[0], bin_midpoints[-1]))
            random_samples = np.random.rand(1)
            next = inverse_cdf(random_samples)[0]
        return next

    def custom_arrival(self, case, previous):
        """
        Call to the custom functions in the file custom_function.py.
        """
        return custom.custom_arrivals_time(case, previous)

