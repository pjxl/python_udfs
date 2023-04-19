import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class MonteCarlo:
    def __init__(self, seed):
        self.seed = seed
        self.rng = None
        self.inputs = {}
        self.pdf = self.DistributionGenerator(self)

    
    # Initialize a random number generator based on the provided seed
    def _initialize_rng(self):
        self.rng = np.random.default_rng(self.seed)


    def _check(self, names):
        names = set(names)
        initialized_names = set(self.inputs.keys())
        
        if names.issubset(initialized_names):
            pass
        else:
            uninit_inputs = names.difference(initialized_names)
            msg = f"Parameter named '{list(uninit_inputs)[0]}' has not been defined as an input in {self.__class__.__name__} object."
            msg += f' Defined inputs are: {list(self.inputs.keys())}.'
            raise NameError(msg)


    def simulate(self, func, n=10000):
        self._initialize_rng()
        simulations = []
        samples = defaultdict(list)

        if callable(func):
            func_args = set(func.__code__.co_varnames)
        else:
            raise TypeError('Argument passed to `func` must be a callable.')

        self._check(func_args)

        for i in range(n):
            sample_i = {}
            for name in func_args:
                val = self.inputs[name].draw()
                sample_i.update({name: val})
                samples[name].append(val)
            
            simulations.append(func(**sample_i))

        return self.SimulationArray(a=simulations, samples=pd.DataFrame(samples))


    # An InputDict is a dict corresponding to a single user-defined input to the MonteCarlo model
    # An InputDict is typically created through method calls made to the MonteCarlo object's `pdf` attribute
    class InputDict(dict):
        def __init__(self, outer, obj):
            self.outer = outer
            return super().__init__(obj)


        # Draw a single random sample from the input's distribution
        def draw(self):
            if not self.outer.rng:
                self.outer._initialize_rng()

            f = self.get('function')
            kwargs = self.get('args')
            return f(**kwargs)


        # Plot the sampling distribution of the input
        def plot(self, n=10000, fig=None, ax=None):
            self.outer._initialize_rng()
            
            samples = []
            for i in range(n):
                samples.append(self.draw())            
            
            if not fig and not ax:
                fig, ax = plt.subplots()

            sns.histplot(samples, ax=ax, kde=True, stat='probability')

            return fig, ax


    # Each MonteCarlo instance has a `pdf` property that is an instance of DistributionGenerator
    # DistributionGenerator is an abstractive layer whose main purpose is to map a user's input definitions to the appropriate numpy Generator
    class DistributionGenerator:
        def __init__(self, outer):
            self.outer = outer


        def _compile(self, distr_func, distr_type, **kwargs):
            return self.outer.InputDict(outer=self.outer, obj={'distribution': distr_type, 'function': distr_func, 'args': {**kwargs}})

        
        def degenerate(self, val):
            def degenerate_distribution(**kwargs):
                if isinstance(val, int):
                    return self.outer.rng.integers(low=val, high=val, endpoint=True)
                else:
                    return self.outer.rng.uniform(low=val, high=val)

            return self._compile(degenerate_distribution, 'degenerate', val=val)


        def uniform(self, low, high):
            def uniform_distribution(**kwargs):
                if all([isinstance(i, int) for i in (low, high)]):
                    return self.outer.rng.integers(low, high, endpoint=True)
                else:
                    return self.outer.rng.uniform(low, high)
            
            return self._compile(uniform_distribution, 'uniform', low=low, high=high)


        def normal(self, center, stddev):
            def normal_distribution(**kwargs):
                return self.outer.rng.normal(loc=center, scale=stddev)
            
            return self._compile(normal_distribution, 'normal', center=center, stddev=stddev)


        def triangular(self, low, mode, high):
            def trianguler_distribution(**kwargs):
                return self.outer.rng.triangular(left=low, mode=mode, right=high)
            
            return self._compile(trianguler_distribution, 'triangular', low=low, mode=mode, high=high)


        def PERT(self, low, mode, high, lamb=4):
            def PERT_distribution(**kwargs):
                r = high - low
                alpha = 1 + lamb * (mode - low) / r
                beta = 1 + lamb * (high - mode) / r        
                return low + self.outer.rng.beta(alpha, beta) * r
            
            return self._compile(PERT_distribution, 'PERT', low=low, mode=mode, high=high, lamb=lamb)

    
    # A SimulationArray is an np.ndarray of simulation results with added attributes
    class SimulationArray(np.ndarray):
        def __new__(cls, a, samples=None):
            obj = np.asarray(a).view(cls)
            obj.samples = samples
            return obj


        def plot(self, kind='pdf', fig=None, ax=None):
            cume = True if kind=='cdf' else False

            if not fig and not ax:
                fig, ax = plt.subplots()
            
            sns.histplot(self, ax=ax, kde=True, stat='probability', cumulative=cume)
            
            return fig, ax
