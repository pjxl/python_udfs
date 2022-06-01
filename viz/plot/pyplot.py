# Custom extensions to base matplotlib.pyplot functionality

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdate


# Subclass extending matplotlib.axes._subplots.AxesSubplot
# Workaround inspired by https://stackoverflow.com/questions/48593124/subclass-axes-in-matplotlib
class _AugmentedAxis(Axes):
    name = 'aug'


    def _get_spine(self, spine):
        return self.__getattribute__(spine + 'axis')


    def _format_tick_labels(self, axis, format, **kwargs):
        spine = self._get_spine(axis)

        def _as_percent(spine, **kwargs):
            return spine.set_major_formatter(mtick.PercentFormatter(**kwargs))

        if format == 'percent':
            return _as_percent(spine, **kwargs)
        else:
            pass


    def format_xtick_labels(self, format=None, **kwargs):
        return self._format_tick_labels(axis='x', format=format, **kwargs)

    
    def format_ytick_labels(self, format=None, **kwargs):
        return self._format_tick_labels(axis='y', format=format, **kwargs)


matplotlib.projections.register_projection(_AugmentedAxis)


def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs, subplot_kw=dict(projection='aug'))
