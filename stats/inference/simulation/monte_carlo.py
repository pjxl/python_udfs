import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MonteCarlo:
    """
    TODO: Annotation
    """
    def __init__(self, seed=None):
        self.seed = seed
        self.distribution, self.distr_args, self.distr_samples, self.distr_pdf = {}, {}, {}, {}
        self.var, self.simulation = [], []


    def _initialize_rng(self):
        self.rng = np.random.default_rng(self.seed)
    

    def _fit_var(self, var, distribution, **kwargs):
        self.distribution.update({var: distribution})
        self.distr_args.update({var: {**kwargs}})
        self.var.append(var)
    

    def _draw(self, var, write=True):
        sample = self.distribution[var](**self.distr_args[var])

        if write:
            if var in self.distr_samples:
                self.distr_samples[var].append(sample)
            else:
                self.distr_samples.update({var: [sample]})        

        return sample


    def _plot_pdf(self, x, n, fig=None, ax=None, title=None, xlab=None, confint=None):
        if not fig and not ax:
            fig, ax = plt.subplots()
        
        sns.histplot(x, ax=ax, kde=True, stat='probability')
        ax.set_xlabel('# resamples: {:,.0f}'.format(n))
        ax.set_title(title)
        
        if confint:
            ax.vlines(self.confint, *ax.get_ylim(), linestyle='dashed', colors=['black']*2)
        
        return fig, ax


    def _confint(self, alpha):
        pctiles = alpha/2, 1-alpha/2
        ci_lower, ci_upper = np.quantile(self.simulation, np.array(pctiles))
        self.confint = ci_lower, ci_upper


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


    def plot_sampling_distribution(self, component, fig, ax, n=1000):
        self._initialize_rng()

        if isinstance(component, str) and component in self.distribution:
            samples = []
            for i in range(n):
                samples.append(self._draw(component, write=False))            
            
            self._plot_pdf(samples, n, fig=fig, ax=ax)


    def simulate(self, func=None, n=1000, confint_alpha=0.05):
        self._initialize_rng()

        if func and callable(func):
            for i in range(n):
                samples = {}
                for var in self.distribution.keys():
                    samples.update({var: self._draw(var)})
                
                self.simulation.append(func(**samples))
            
            for var in self.distribution.keys():
                p, _ = self._plot_pdf(self.distr_samples[var], n, title=f'Monte Carlo sampling distribution: {var}')
                self.distr_pdf.update({var: p})
                plt.close()

        self._confint(confint_alpha)


    def plot_simulation_results(self, fig, ax):
        if self.simulation:  
            return self._plot_pdf(self.simulation, n=len(self.simulation), confint=True, fig=fig, ax=ax)
