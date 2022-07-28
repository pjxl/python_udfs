# Various UDFs defining "quick-plots"
# These will typically be very specialized in nature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def senstable(rows, cols, 
              dimnames=['x1', 'x2'], 
              func=lambda i, j: None,
              row_val_format=None,
              col_val_format=None,
              cell_val_format=None,
              gradient_color='goldenrod',
              callout_area=None,
              highlight_between=None,
              title_props={'text': None, 'loc': 'center', 'fontsize': 16}
             ):
  
    """
    Generate and format a 2-dimensional sensitivity table.
    
    Parameters
    ----------
    rows : array-like of numerics
        A numeric sequence representing the Y-axis variable.
        
    cols : array-like of numerics
        A numeric sequence representing the X-axis variable.
    
    func : function
        A function that returns the cell values for each (i, j) combination of (rows, cols).
        For example: `lambda row, col: row * col`.
    
    row_val_format : function, default None
        A function that formats the row values. 
        For example: `lambda rows: '{:.1f}%'.format(rows*100)`.
    
    col_val_format : function, default None
        A function that formats the column values.
        For example: `lambda cols: '{:.1f}%'.format(cols*100)`.
    
    cell_val_format : function, default None
        A function that formats the cell values.
        For example: `lambda vals: '{:.1f}%'.format(vals*100)`.
    
    gradient_color : str, default "goldenrod"
        The primary color for the table's gradient color map (passed to pandas.io.formats.style.Styler.background_gradient).

    callout_area : dict, default None
        Callout a group of cell(s) with a prominent border.
        For example, this can be used to emphasize a 2D range of expected values.
        dict must contain all of the following keys: `row_start`, `row_stop`, `col_start`, and `col_stop`.
        Each `%_start` / `%_stop` key must have an int value, corresponding to a starting/stopping (inclusive) positional index for a given "corner" of the target region.
    
    highlight_between : dict, default None
        A dict with keywords arguments to be passed to pandas.io.formats.style.Styler.highlight_between().
        For example: `{'right': 1, 'props': 'font-weight:bold;color:#e83e8c'}`.
    
    title_props : dict, default `{'text': None, 'loc': 'center', 'fontsize': 16}`
        A dict containing properties used to create and customize the table's caption.
    """
        
    # Convert row/col variables to 1D DataFrame objects
    rows_df, cols_df = (pd.DataFrame(v) for i, v in enumerate([rows, cols]))
    
    # Get the Cartesian product of all row/col values
    xmat = rows_df.merge(cols_df, how='cross')
    
    # Append a third column 'y' containing the output of func (corresponds to the table cell values)
    xmat['y'] = [func(xmat.iloc[i, 0], xmat.iloc[i, 1]) for i in xmat.index]
    
    # Pivot to wide format
    df = xmat.pivot(xmat.columns[0], xmat.columns[1], 'y')
    
    # Apply any row/col value formatting
    if row_val_format:
        df.index = [row_val_format(i) for i in df.index]
        
    if col_val_format:
        df.columns = [col_val_format(i) for i in df.columns]
    
    # Apply row/col headers
    ylabel, xlabel = dimnames    
    df.index = pd.MultiIndex.from_product([[ylabel], df.index])
    df.columns = pd.MultiIndex.from_product([[xlabel], df.columns])

    # Format cell text
    style = df.style.format(cell_val_format)

    # Apply table styling
    # Resources on functionality here: https://pandas.pydata.org/docs/user_guide/style.html
    head_css = {
        'selector': 'th',
        'props': [('background-color', 'white')]
    }
    col_head_val_css = {
        'selector': 'th.col_heading.level1',
        'props': [('text-align', 'center'),
                  ('font-style', 'italic'),
                  ('font-family', 'Arial'),
                  ('color', 'black'),
                  ('background-color', 'whitesmoke'),
                  ('border-bottom', '2px solid black')]
    }
    row_head_val_css = {
        'selector': 'th.row_heading.level1',
        'props': [('text-align', 'center'),
                  ('font-style', 'italic'),
                  ('font-family', 'Arial'),
                  ('color', 'black'),
                  ('background-color', 'whitesmoke'),
                  ('border-right', '2px solid black')]
    }
    row_name_css = {
        'selector': 'th.row_heading.level0',
        'props': [('text-align', 'center'),
                  ('font-style', 'italic'),
                  ('font-family', 'Arial'),
                  ('color', 'black'),
                  ('background-color', 'white'),
                  ('writing-mode', 'vertical-lr'),
                  ('transform', 'rotate(180deg)'),
                  ('white-space', 'pre'),
                  ('line-height', '200%'),
                  ('border-top', '1px solid white'),
                  ('border-right', '1px solid white')]
    }
    col_name_css = {
        'selector': 'th.col_heading.level0',
        'props': [('text-align', 'center'),
                  ('font-style', 'italic'),
                  ('font-family', 'Arial'),
                  ('color', 'black'),
                  ('background-color', 'white')]
    }
    caption_css = {
        'selector': 'caption',
        'props': [('background-color', 'white'),
                  ('text-align', f"{title_props.get('loc', 'center')}"),
                  ('font-size', f"{str(title_props.get('fontsize', 16))+'px'}"),
                  ('color', 'black'),
                  ('padding-left', '10px'),
                  ('padding-bottom', '10px')]
    }

    style = style.set_table_styles([head_css, col_head_val_css, row_head_val_css, row_name_css, col_name_css, caption_css],
                                   overwrite=False)

    # Apply background gradient
    cmap = sns.light_palette(gradient_color, as_cmap=True)
    style = style.background_gradient(cmap, axis=None)

    # Emphasize any callout area by enclosing those cell(s) in a prominent border
    if callout_area:
    
        # Extract boundary markers
        # Each should be a positional int
        row_start, row_stop = callout_area.get('row_start'), callout_area.get('row_stop')
        col_start, col_stop = callout_area.get('col_start'), callout_area.get('col_stop')

        # Ensure we have all four markers
        if any([i is None for i in (row_start, row_stop, col_start, col_stop)]):
            raise KeyError('Argument must be a dict with keys `row_start`, `row_stop`, `col_start`, and `col_stop`')

        # .set_properties()'s `subset` param looks up slices using .loc, not .iloc
        # So swap each positional int for its corresponding index/column name
        row_start, row_stop = style.data.index[row_start], style.data.index[row_stop]
        col_start, col_stop = style.data.columns[col_start], style.data.columns[col_stop]
        
        # Initialize slicer
        idx = pd.IndexSlice

        # Carve out four directional slices
        # The name of each slice indicates the type of border to apply
        left = idx[row_start:row_stop, col_start]
        right = idx[row_start:row_stop, col_stop]
        top = idx[row_start, col_start:col_stop]
        bottom = idx[row_stop, col_start:col_stop]

        # Apply callout border to each directional slice
        for i in 'left', 'right', 'top', 'bottom':
            style.set_properties(**{'border-' + i: '4px solid #e83e8c'}, subset=eval(i))
    
    # Apply any conditional highlighting
    if highlight_between:
        style = style.highlight_between(**highlight_between)
    
    return style.set_table_attributes("style='display:inline'").set_caption(title_props.get('text', None))



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
