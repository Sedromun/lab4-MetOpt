import math

import numpy as np


class SimulatedAnnealing:

    def __init__(
            self,
            cost_function,
            interval,
            initial_temperature=1000,
            cooling_rate=0.93,
            acceptance_rate_threshold=40,
            amplitude_rate_step=0.999
    ):
        self.cost_function = cost_function
        self.interval = interval
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.acceptance_rate_threshold = acceptance_rate_threshold
        self.amplitude_rate_step = amplitude_rate_step

    def annealing(self, start_point=None, n_iterations=10000):
        state = self.__random_start() if start_point is None else start_point
        cost = self.cost_function(state)
        states, costs = [state], [cost]
        amplitude = (self.interval[1] - self.interval[0]) / 10  # Initial amplitude
        acceptance_count = 0
        for step in range(n_iterations):
            T = self.__temperature(step)
            new_state = self.__random_neighbour(state, amplitude)
            new_cost = self.cost_function(new_state)
            if new_cost < cost or self.__acceptance_probability(cost, new_cost, T) > np.random.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
                acceptance_count += 1

            amplitude *= self.amplitude_rate_step

            if step % 100 == 0:
                if acceptance_count > self.acceptance_rate_threshold:
                    T *= 0.9  # Cool faster if too many moves are accepted
                else:
                    T *= 1.1  # Cool slower if too few moves are accepted
                acceptance_count = 0

        return state, self.cost_function(state), states, costs

    @staticmethod
    def __acceptance_probability(cost, new_cost, temperature):
        return np.exp(- (new_cost - cost) / temperature)

    def __temperature(self, step):
        return self.temperature * (self.cooling_rate ** step)

    def __random_neighbour(self, x, amplitude):
        delta = (-amplitude / 2) + amplitude * np.random.random_sample(size=len(x))
        return np.clip(x + delta, self.interval[0], self.interval[1])

    def __random_start(self):
        a, b = self.interval
        return a + (b - a) * np.random.random_sample(size=len(self.interval))
