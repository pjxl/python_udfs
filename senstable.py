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
        
        highlight_between: A dict object with keywords arguments to be passed to pandas.io.formats.style.Styler.highlight_between.
        
        title: A string object to be passed to the title of the table.
    """
    
    import pandas as pd
    import seaborn as sns
    
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
