import matplotlib.font_manager
from IPython.core.display import HTML

def get_plt_options(param=None):
    
    """
    Print out available matplotlib options.
    
    Parameters
    ----------
        param: A string corresponding to the matplotlib parameter of interest. Options include: 'fontname' (default: None).
    """
    
    if param == 'fontname':
        
        def make_html(fontname):
            return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)
        
        code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

        return HTML("<div style='column-count: 2;'>{}</div>".format(code))
    
    else:
        return None
