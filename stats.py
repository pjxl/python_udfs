from math import sqrt
import scipy.stats as stats
import numpy as np
import pandas as pd


def confint(x, stat, se, ha='two-sided'):
    ci_lwr = x - stat * se
    ci_upr = x + stat * se
#     if ha[0] == 'l':
#         ci_lwr = None
#     else:
#         ci_lwr = x - stat * se
    
#     if ha[0] == 'r':
#         ci_upr = None
#     else:
#         ci_upr = x + stat * se

    return ci_lwr, ci_upr


# https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1330
def _ztest_p(z, ha):
    if ha == 'less':
        p = stats.norm.cdf(z)
    elif ha == 'greater':
        p = stats.norm.sf(z)
    elif ha == 'two-sided':
        p = 2 * stats.norm.sf(abs(z), dof)
        
    return p


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

        ha: (string; default='two-sided') The desired directionality of the test. A valid input is one of ('two-sided', 'greater', 'less').
    """

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
    p = _ztest_p(z, ha)

    # Calculate the critical z-score
    one_sided = ha[0] in {'greater', 'less'}
    z_critical = stats.norm.ppf(1 - alpha / (1 + (not one_sided)))

    # Find the lower and upper CIs boundaries
    # n.b.: in units of the difference between p_treat and p_ctrl
    ci_lwr, ci_upr = confint(p_treat-p_ctrl, z_critical, se, ha=ha)

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
      {'': [p_ctrl, p_treat, z, p, lift, p_treat-p_ctrl, ci_lwr, ci_upr]},
      index=['control', 'treatment', 'z-score', 'p-value', 'lift', 'diff',
             'diff ({0:.0f}% CI lower)'.format(100*(1-alpha)),
             'diff ({0:.0f}% CI upper)'.format(100*(1-alpha))])

    with pd.option_context('display.precision', 10):
        return out_df


# https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L5661
def _ttest_p(t, d0f, ha):
    if ha == 'less':
        p = stats.t.cdf(t, dof)
    elif ha == 'greater':
        p = stats.t.sf(t, dof)
    elif ha == 'two-sided':
        p = 2 * stats.t.sf(abs(t), dof)
        
    return p


def welch_ttest(treat, ctrl, alpha=0.05, ha='two-sided'):
    
    # Assert valid values for ha
    valid_ha = {'two-sided', 'greater', 'less'}
    ha = ha.lower()

    # Raise an error if a valid ha value is not provided
    if ha not in valid_ha:
        raise ValueError('"ha" must be one of %r.' % valid_ha)

    treat, ctrl = [pd.Series(i) for i in (treat, ctrl)]
    
    # Get each group's sample size 
    n_treat, n_ctrl = [i.count() for i in (treat, ctrl)]
    
    # Get the mean of each group
    mean_treat, mean_ctrl = [i.mean() for i in (treat, ctrl)]
    
    # Get the variance of each group
    var_treat, var_ctrl = [i.var(ddof=1) for i in (treat, ctrl)]
    
    # Calculate the pooled standard error
    se = var_treat/n_treat + var_ctrl/n_ctrl
    se = sqrt(se)
    
    t = (mean_treat-mean_ctrl) / se
    
    # Welch-Satterthwaite degrees of freedom    
    dof = (var_treat/n_treat + var_ctrl/n_ctrl)**2 / ((var_treat/n_treat)**2 / (n_treat-1) + (var_ctrl/n_ctrl)**2 / (n_ctrl-1))
    
    # Calculate the p-value associated with t and dof
    p = _ttest_p(t, dof, ha)
    
    # Calculate the critical t-score
    one_sided = ha in {'greater', 'less'}
    t_critical = stats.t.ppf(1 - alpha / (1 + (not one_sided)), dof)
    
    # Find the lower and upper CIs boundaries
    # n.b.: in units of the difference between mean_treat and mean_ctrl
    ci_lwr, ci_upr = confint(mean_treat-mean_ctrl, t_critical, se, ha=ha)
    
    # Calculate the pct lift
    lift = mean_treat/mean_ctrl-1
    lift_lwr = ci_lwr/mean_ctrl
    lift_upr = ci_upr/mean_ctrl
    
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
      {'': [mean_ctrl, mean_treat, t, p, dof, lift, mean_treat-mean_ctrl, ci_lwr, ci_upr]},
      index=['control', 'treatment', 't-score', 'p-value', 'DoF', 'lift', 'diff',
             'diff ({0:.0f}% CI lower)'.format(100*(1-alpha)),
             'diff ({0:.0f}% CI upper)'.format(100*(1-alpha))])

    with pd.option_context('display.precision', 10):
        return out_df
