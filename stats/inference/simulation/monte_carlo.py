import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class MonteCarlo:
    """
    TODO: Annotation
    """
    def __init__(self, seed=None):
        self.seed = seed
        self.variables = []
        self.distributions, self.args, self.pdfs = {}, {}, {}


    # Initialize a random number generator based on the provided seed
    def _initialize_rng(self):
        self.rng = np.random.default_rng(self.seed)
    
    # Map a variable to its specified distribution and arguments
    # These are stored as properties
    def _fit_var(self, var, distribution, **kwargs):
        self.distributions.update({var: distribution})
        self.args.update({var: {**kwargs}})
        if var not in self.variables:
            self.variables.append(var)
    
    # For a given variabe, draw a single random sample from its distribution
    def _draw(self, var):
        return self.distributions[var](**self.args[var])


    def _plot_pdf(self, x, n, fig=None, ax=None, title=None, xlab=None, confint=None):
        if not fig and not ax:
            fig, ax = plt.subplots()
        
        sns.histplot(x, ax=ax, kde=True, stat='probability')
        ax.set_xlabel('# resamples: {:,.0f}'.format(n))
        ax.set_title(title)
        
        if confint:
            ax.vlines(confint, *ax.get_ylim(), linestyle='dashed', colors=['black']*2)
        
        return fig, ax


    def exact_value(self, var, val):
        def degenerate_distribution(**kwargs):
            return self.rng.uniform(low=val, high=val)

        self._fit_var(var, degenerate_distribution, val=val)


    def uniform(self, var, low, high):
        def uniform_distribution(**kwargs):
            return self.rng.uniform(low, high)
        
        self._fit_var(var, uniform_distribution, low=low, high=high)


    def normal(self, var, center, stddev):
        def normal_distribution(**kwargs):
            return self.rng.normal(loc=center, scale=stddev)
        
        self._fit_var(var, normal_distribution, center=center, stddev=stddev)


    def triangle(self, var, low, mode, high):
        def triangle_distribution(**kwargs):
            return self.rng.triangular(left=low, mode=mode, right=high)
        
        self._fit_var(var, triangle_distribution, low=low, mode=mode, high=high)


    def PERT(self, var, low, mode, high, lamb=4):
        def PERT_distribution(**kwargs):
            r = high - low
            alpha = 1 + lamb * (mode - low) / r
            beta = 1 + lamb * (high - mode) / r        
            return low + self.rng.beta(alpha, beta) * r
        
        self._fit_var(var, PERT_distribution, low=low, mode=mode, high=high, lamb=lamb)


    def plot_sampling_distribution(self, var, n=1000, fig=None, ax=None):
        self._initialize_rng()

        if isinstance(var, str) and var in self.distributions:
            samples = []
            for i in range(n):
                samples.append(self._draw(var))            
            
            self._plot_pdf(samples, n, fig=fig, ax=ax)


    def simulate(self, func, n=1000):
        self._initialize_rng()
        self.simulations = []
        self.samples = defaultdict(list)

        if callable(func):
            for i in range(n):
                sample_i = {}
                for var in self.distributions.keys():
                    val = self._draw(var)
                    sample_i.update({var: val})
                    self.samples[var].append(val)
                
                self.simulations.append(func(**sample_i))
            
            for var in self.distributions.keys():
                p, _ = self._plot_pdf(self.samples[var], 
                                      len(self.samples[var]), 
                                      title=f'Monte Carlo sampling distribution: {var}')
                self.pdfs.update({var: p})
                plt.close()

        self.simulations = np.array(self.simulations)
        self.samples = dict(self.samples)

    
    # def simulation_confint(self, alpha):
    #     pctiles = alpha/2, 1-alpha/2
    #     ci_lower, ci_upper = np.quantile(self.simulations, np.array(pctiles))
    #     return ci_lower, ci_upper


    def plot_simulation_distribution(self, fig=None, ax=None, confint=None):
        if hasattr(self, 'simulations'):  
            return self._plot_pdf(self.simulations, n=len(self.simulations), confint=confint, fig=fig, ax=ax)
