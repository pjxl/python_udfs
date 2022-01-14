def conf_int(x, stat, se):  
  ci_lwr = x - stat * se
  ci_upr = x + stat * se
  
  return ci_lwr, ci_upr


def ztest_2prop(x_treat, n_treat, x_ctrl, n_ctrl, alpha=0.05, ha='two-sided'):
  """
    Conduct a two-proportion z-test.
    
    Parameters
    ----------
        x_treat: (numeric) The number of observations with the event of interest in the treatment group.
        
        n_treat: (numeric) The total number of observations in the treatment group.
        
        x_ctrl: (numeric) The number of observations with the event of interest in the control group.
        
        n_ctrl: (numeric) The total number of observations in the control group.
        
        alpha: (float; default=0.05) The desired significance level.
        
        ha: (string; default='two-sided') The desired directionality of the test. A valid input is one of ('two-sided', 't', 'greater', 'g', 'less', 'l').
    """  
    
  from math import sqrt
  import scipy.stats as stats
  import pandas as pd

  # Assert valid values for ha
  valid_ha = {'two-sided', 't', 'greater', 'g', 'less', 'l'}
  ha = ha.lower()
  
  # Raise an error if a valid ha value is not provided
  if ha not in valid_ha:
      raise ValueError('"ha" must be one of %r.' % valid_ha) 

  # Calculate relative proportions
  p_treat = x_treat/n_treat
  p_ctrl = x_ctrl/n_ctrl
  p_pooled = (x_treat+x_ctrl)/(n_treat+n_ctrl)
  
  # Compute the standard error of the sampling distribution of the difference between p_treat and p_ctrl
  se = p_pooled*(1-p_pooled)*(1/n_treat+1/n_ctrl)
  se = sqrt(se)

  # Calculate the z test statistic
  z = (p_treat-p_ctrl)/se
  
  # Calculate the p-value associated with z
  p = 1-stats.norm.cdf(abs(z))
  one_sided = ha[0] in {'g', 'l'}
  p *= 2-one_sided

  # Calculate the critical z-score
  z_critical = stats.norm.ppf(1 - alpha / (1 + (not one_sided)))
  if ha[0] == 'l':
      z_critical *= -1

  # Find the lower and upper CIs
  # n.b.: in units of the difference between p_treat and p_ctrl
#   ci_lwr = p_treat - p_ctrl - z_critical*se
#   ci_upr = p_treat - p_ctrl + z_critical*se

  ci_lwr, ci_upr = conf_int(p_treat-p_ctrl, z_critical, se)

  # Calculate the pct lift
  lift = p_treat/p_ctrl-1
  lift_lwr = ci_lwr/p_ctrl
  lift_upr = ci_upr/p_ctrl
  
  # Function to format decimals as percentages for print-out readability
  # Optionally, prepend a sign (+/-) before a percentage (e.g., when representing lift estimates)
  def format_pct_str(x, precision=None, sign=False):
      pct = str(round(x*100, precision)) + '%'
      return '+' + pct if x >= 0 else pct

  # Star indicator if the diff in proportions was statsig
  sig = '*' if p <= alpha else ''
  
  # Print a readout of the experiment conclusion
  print(f'{format_pct_str(lift, precision=2)}', 
        'lift in the treatment', 
        f'({format_pct_str(1-alpha)} CI:',
        f'{format_pct_str(lift_lwr, precision=2, sign=True)}',
        f'to {format_pct_str(lift_upr, precision=2, sign=True)})',
        sig)
  
  # DataFrame with all test outputs of interest
  out_df = pd.DataFrame(
      {'': [p_ctrl, p_treat, lift, z, p, p_treat-p_ctrl, ci_lwr, ci_upr]}, 
      index=['control', 'treatment', 'lift', 'z-score', 'p-value', 'diff',
             'diff ({0:.0f}% CI lower)'.format(100*(1-alpha)),
             'diff ({0:.0f}% CI upper)'.format(100*(1-alpha))])
  
  with pd.option_context('display.precision', 10):
      return out_df
