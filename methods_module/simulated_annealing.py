import numpy as np


class SimulatedAnnealing:

    def __init__(self, cost_function, interval):
        self.cost_function = cost_function
        self.interval = interval

    def annealing(self, n_iterations=1000):
        state = self.__random_start()
        cost = self.cost_function(state)
        states, costs = [state], [cost]
        for step in range(n_iterations):
            fraction = step / float(n_iterations)
            T = self.__temperature(fraction)
            new_state = self.__random_neighbour(state, fraction)
            new_cost = self.cost_function(new_state)
            if new_cost < cost or self.__acceptance_probability(cost, new_cost, T) > np.random.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
        return state, self.cost_function(state), states, costs

    @staticmethod
    def __acceptance_probability(cost, new_cost, temperature):
        return np.exp(- (new_cost - cost) / temperature)

    @staticmethod
    def __temperature(fraction):
        return max(0.01, min(1, 1 - fraction))

    def __random_neighbour(self, x, fraction=1.0):
        a, b = self.interval
        amplitude = (b - a) * fraction / 10
        delta = (-amplitude / 2) + amplitude * np.random.random_sample()
        return self.__clip(x + delta)

    def __random_start(self):
        a, b = self.interval
        return a + (b - a) * np.random.random_sample()

    def __clip(self, x):
        a, b = self.interval
        res = []
        for i in range(len(a)):
            res.append(max(min(x[i], b[i]), a[i]))
        return np.array(res)
