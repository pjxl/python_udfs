# Custom extensions to base matplotlib.pyplot functionality

import matplotlib.projections as proj
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdate


# Subclass extending matplotlib.axes._subplots.AxesSubplot
# Workaround inspired by https://stackoverflow.com/questions/48593124/subclass-axes-in-matplotlib
class _AugmentedAxis(Axes):
    name = 'aug'

    def _get_spines(self, axis):
        axmap = {'x': [self.xaxis],
                 'y': [self.yaxis],
                 'both': [self.xaxis, self.yaxis]}
        
        return axmap.get(axis)
        
    def format_tick_labels_numeric(self, axis, decimals, as_percent=False, prefix='', suffix=''):        
        cast_type = '%' if as_percent else 'f'
        fmt = f'{prefix}{{x:,.{decimals}{cast_type}}}{suffix}'
        
        for spine in self._get_spines(axis):
            spine.set_major_formatter(mtick.StrMethodFormatter(fmt))

    def format_tick_labels_date(self, axis, strftime):
        for spine in self._get_spines(axis):
            spine.set_major_formatter(mdate.DateFormatter(strftime))

    def set_tick_intervals(self, axis, interval):
        for spine in self._get_spines(axis):
            spine.set_major_locator(mtick.MultipleLocator(base=interval))


proj.register_projection(_AugmentedAxis)


def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs, subplot_kw=dict(projection='aug'))
