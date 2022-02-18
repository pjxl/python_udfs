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
