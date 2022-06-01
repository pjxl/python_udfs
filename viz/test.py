import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Iterable


class EtsyColors():
    """
    Instantiate an object with a `library` property containing Etsy's brand colors.
    See: https://drive.google.com/file/d/185hhTOMsBWTacLE8fZo44czjJloB1qCb/view
    """


    def __init__(self):
        self.library = {
            'orange': {'dark': '#CF4018', 'medium': '#F1641E', 'light': '#FAA077'},
            'denim': {'dark': '#232347', 'medium': '#2F466C', 'light': '#4D6BC6'},
            'grey': {'dark': '#222222', 'medium': '#595959', 'light': '#757575'},
            'lavender': {'dark': '#3E1C53', 'medium': '#654B77', 'light': '#9560B8'},
            'beeswax': {'dark': '#A66800', 'medium': '#FAA129', 'light': '#FDD95C'},
            'slime': {'dark': '#1C4A21', 'medium': '#258635', 'light': '#9EC063'},
            'brick': {'dark': '#540D17', 'medium': '#A61A2E', 'light': '#FD9184'},               
            'turquoise': {'dark': '#1A3B38', 'medium': '#2F766D', 'light': '#7ED4BD'},
            'bubblegum': {'dark': '#592642', 'medium': '#B54C82', 'light': '#F592B8'}
        }


    def __hex_fetcher(self, hue=None, tint=None):
        # If the input object isn't already a non-string iterable, make it one
        def to_iter(x):
            return [x] if not isinstance(x, Iterable) or isinstance(x, str) else x
        
        # Cast all str items in the input object to lowercase
        def parse_str(x):
            return [i.lower() if isinstance(i, str) else i for i in x]

        def parse_hue(hue):
            hue = parse_str(to_iter(hue))
            
            all_hues = self.library.keys()
            core_hues = ['orange', 'denim', 'grey']

            if any([i in (None, 'all') for i in hue]):
                hue = all_hues
            elif any([i == 'core' for i in hue]):
                hue = core_hues
            elif any([i == 'extended' for i in hue]):
                hue = [i for i in all_hues if i not in core_hues]
            else:
                hue = hue
            
            return hue

        def parse_tint(tint):
            tint = parse_str(to_iter(tint))

            all_tints = ['dark', 'medium', 'light']

            if any([i in (None, 'all') for i in tint]):
                tint = all_tints
            else:
                tint = tint
            
            return tint
        
        hue, tint = parse_hue(hue), parse_tint(tint)

        hexes = []
        for h in hue:
            for t in tint:
                try:
                    hexes.append(self.library.get(h).get(t))
                except:
                    pass
        
        return hexes


    def make_palette(self, hue=None, tint=None, n_colors=None):
        """
        Generate a custom color palette out of the options in `self.library`.
        """

        hexes = self.__hex_fetcher(hue, tint)

        pal = sns.color_palette(hexes, n_colors=n_colors)
        
        self.palette = pal if len(pal) > 0 else None

    
    def plot_palette(self):
        """
        Plot the colors in the `self.palette` property if it exists; return `None` otherwise.
        """

        try:
            pal = self.palette
        except:
            pal = None

        if pal:
            sns.palplot(pal)
            plt.show()
        else:
            return None


    def plot_library(self):
        """
        Plot the full library of Etsy colors in the `self.library` property.
        """
        pal = self.__hex_fetcher()
        sns.palplot(pal)
        plt.show()

      

def senstable(rows, cols, 
              dimnames=['x1', 'x2'], 
              func=lambda i, j: None,
              row_val_format=None,
              col_val_format=None,
              cell_val_format=None,
              gradient_color='royalblue',
              highlight_between=None,
              title=None,
              caption=None
             ):
  
    """
    Generate and format a 2-dimensional sensitivity table.
    
    Parameters
    ----------
        rows: A numeric sequence object representing the Y-axis variable.
        
        cols: A numeric sequence object representing the X-axis variable.
        
        func: A function object that returns the cell values for each (i, j) combination of (rows, cols).
        
        row_val_format: A function object that formats the row values (e.g. lambda rows: '{:.1f}%'.format(rows*100)).
        
        col_val_format: A function object that formats the column values (e.g. lambda cols: '{:.1f}%'.format(cols*100)).
        
        cell_val_format: A function object that formats the cell values (e.g. lambda vals: '{:.1f}%'.format(vals*100)).
        
        gradient_color: The primary color for the table's gradient color map (passed to pandas.io.formats.style.Styler.background_gradient).
        
        highlight_between: A dict object with keywords arguments to be passed to pandas.io.formats.style.Styler.highlight_between (e.g. {'right': 1, 'props': 'font-weight:bold;color:#e83e8c'}).
        
        title: A string object to be passed to the title of the table.
    """
        
    # Convert row/col variables to 1D DataFrame objects
    rows_df, cols_df = (pd.DataFrame(v) for i, v in enumerate([rows, cols]))
    
    # Get the Cartesian product of all row/col values
    xmat = rows_df.merge(cols_df, how='cross')
    
    # Append a third column 'y' containing the output of func (corresponds to the table cell values)
    xmat['y'] = [func(xmat.iloc[i, 0], xmat.iloc[i, 1]) for i in xmat.index]
    
    # Pivot to wide format
    t = xmat.pivot(xmat.columns[0], xmat.columns[1], 'y')
    
    # Apply any row/col value formatting
    if row_val_format:
        t.index = [row_val_format(i) for i in t.index]
        
    if col_val_format:
        t.columns = [col_val_format(i) for i in t.columns]
    
    # Name row/col dimensions
    t.index.name = dimnames[0]
    t.columns.name = dimnames[1]
    
    t = t.style.format(cell_val_format)
    
    # Designate and apply standard table styling
    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: white; font-weight:bold; background-color: #000066;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #000066; color: white;'
    }
    t = t.set_table_styles([index_names, headers])
    
    # Apply background gradient
    cmap = sns.light_palette(gradient_color, as_cmap=True)
    t = t.background_gradient(cmap, axis=None)
    
    # Apply any conditional highlighting
    if highlight_between:
        t = t.highlight_between(**highlight_between)
    
    return t.set_table_attributes("style='display:inline'").set_caption(title)



def plotDiD(df, group, period, y, agg_func, y_label=None, title=None):
    
    """
    Plot a difference-in-differences (DiD) slope comparison.
    
    Parameters
    ----------
        df: A pandas.DataFrame object containing the panel data of interest.
        
        group: A string representing the name of the column in `df` containing the variant labels.
        
        period: A string representing the name of the column in `df` containing the time period labels.
        
        y: A string representing the name of the column in `df` containing the dependent variable.
        
        agg_func: A string representing the pd.agg() function name (e.g. 'mean') to be performed on y when grouping by `group` and `period`.
        
        y_label: (Optional) A string to be passed to the y-axis name. If not specified, the y-axis name defaults to the value of `y`.
        
        title: (Optional) A string to be passed to the plot title.
    """
        
    periods = np.array(['Pre', 'Post'])
    variants = np.array(['Control', 'Treatment', 'Counterfactual'])

    df_agg = df.groupby([group, period]).agg(y=(y, agg_func)).reset_index()

    t0_ctrl, t1_ctrl, t0_treat, t1_treat = [df_agg[(df_agg[group]==i) & (df_agg[period]==j)].y.values[0] for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    
    t1_counterfact = t1_ctrl + t0_treat - t0_ctrl

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(periods, np.array([t0_ctrl, t1_ctrl]), color='steelblue', marker='o')
    ax.plot(periods, np.array([t0_treat, t1_treat]), color='darkorange', marker='o')
    ax.plot(periods, np.array([t0_treat, t1_counterfact]), color='darkorange', linestyle='dashed', marker='o')
    ax.set_xlabel('Period')
    ax.set_ylabel(y_label if y_label else y)
    ax.set_title(title)
    ax.legend(variants)
    plt.show()
