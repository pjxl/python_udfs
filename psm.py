# Functions and classes for conducting Propensity Score Matching
# Still needs a lot more annotation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import patsy

def _design_matrix_from_formula(f, df):
    """
    Helper function that converts a formula string to DataFrames containing response/predictor variables.
    
    Returns
    -------
    Tuple (Y, X) with `Y` corresponding to a DataFrame containing a y-variable and `X` corresponding to a DataFrame of x-variables.
    """
    Y, X = patsy.dmatrices(f, df, return_type='dataframe')
    return Y, X


def _is_binary_col(s):
    """
    Helper function that assesses whether a pandas.Series object `s` corresponds to a dichotomous variable.
    
    Returns
    -------
    bool
    """
    return s.dtype == bool or s.isin({0,1}).values.all()


# Not quite the methodology used by R's MatchIt
# See: https://stats.stackexchange.com/questions/472421/how-to-calculate-standardized-mean-difference-after-matching
def SMD(a, b, is_binary=False, abs=True):
    """
    Calculate the standardized mean difference (SMD) between two groups on a variable.
    Note that the formula differs depending on whether the variable of interest in continuous or binary.
    For more information, see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6351359/

    Parameters
    ----------
    a : array-like
        An array of values corresponding to the first group of interest.
    b : array-like
        An array of values corresponding to the second group of interest.
    is_binary : bool (default=False)
        Is the variable represent by `a` and `b` dichotomous in nature?
        If False, variable is assumed to be continuous.
    abs : bool (default=False)
        Should the SMD be returned as an absolute value?
    
    Returns
    -------
    float
    """
    m_a, m_b = np.mean(a), np.mean(b)
    s_a, s_b = np.std(a, ddof=2), np.std(b, ddof=1)

    if is_binary:
        sf = np.sqrt((m_a*(1-m_a) + m_b*(1-m_b))/2)
    else:
        sf = np.sqrt((s_a**2 + s_b**2)/2)
    
    return np.abs((m_a - m_b)/sf)


class PSM(object):
    """
    A PSM object is the general parent class from which both PropensityScore and _Matcher classes inherit.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame corresponding to the dataset of interest.
    formula : str
        A string representing an R-style formula, to be passed to statsmodels.api
    reset_data_index : bool (default=False)
        On instantiating, should the index of `data` be reset?
    """
    def __init__(self, data, formula, reset_data_index=False):
        if reset_data_index:
            self.data = data.reset_index(drop=True)
        else:
            self.data = data
        self.formula = formula
        self.Y, self.X = _design_matrix_from_formula(self.formula, self.data)


        def covar_balance():
            Y, X = self.Y.squeeze(), self.X.iloc[:,1:]
            df = pd.concat([Y, X], axis=1)
            
            balance = pd.DataFrame()

            for xcol in X.columns:
                ctrl_mean, treat_mean = df.loc[Y==0, xcol].mean(), df.loc[Y==1, xcol].mean()
                smd = SMD(a=df.loc[Y==0, xcol],
                        b=df.loc[Y==1, xcol],
                        is_binary=_is_binary_col(df[xcol]))

                xrow = pd.DataFrame([[xcol, ctrl_mean, treat_mean, smd]],
                                    columns=['Covariate', 'Mean (Control)', 'Mean (Treatment)', 'Abs. Std. Mean Diff'])
                
                balance = pd.concat([balance, xrow], ignore_index=True)
                
            return balance
        
        self.balance = covar_balance()


class _Matcher(PSM):
    """
    A _Matcher object contains post-matching data.
    _Matcher objects are instantiated via a `.get_matches_%` PropensityScore instance method.

    Attributes
    ----------
    data : pandas.DataFrame
        A DataFrame containing the post-matching dataset.
    matches : pandas.DataFrame
        A DataFrame showing all pairwise treatment : control matches, the associated propensity scores, and the distance between the scores.
        Useful for auditing the matching process.
    balance : pandas.DataFrame
        A DataFrame showing the post-matching balance across covariates.
    formula : str
        A string representing the R-style formula used to generate the propensity scores.
    """
    def __init__(self, data, formula, matches, orig_balance):
        super().__init__(data, formula)
        self.matches = matches
        self._orig_balance = orig_balance


    def love_plot(self, xvline_at=0.1):
        fig, ax = plt.subplots(figsize=(10,6))

        ax.grid(axis='y', c='gainsboro', linestyle='--')

        for b in self._orig_balance, self.balance:
            ax.scatter(b['Abs. Std. Mean Diff'], b['Covariate'], s=60)

        ax.legend(['Pre-matching', 'Post-matching'], facecolor='whitesmoke')
        
        if xvline_at:
            ax.axvline(x=xvline_at, c='black', alpha=0.5, linestyle=':')            

        plt.draw()
        labels = ax.get_yticklabels()
        ax.set_yticklabels(labels=labels, fontsize=12)
        ax.set_xlabel('Absolute Standardized Mean Difference', labelpad=12, fontsize=12)
        ax.set_title('PSM Covariate Balance', fontsize=16, pad=12)


class PropensityScore(PSM):
    """
    A PropensityScore object contains pre-matching data and propensity scores.

    Attributes
    ----------
    data : pandas.DataFrame
        A DataFrame containing the pre-matching dataset.
    balance : pandas.DataFrame
        A DataFrame showing the pre-matching balance across covariates.
    formula : str
        A string representing the R-style formula used to generate the propensity scores.
    scores : pandas.Series
        The propensity scores. Indices correspond to the indices of `data`.
    score_fit : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        The GLM fit that generated the propensity scores.
    """
    def __init__(self, data, formula, link='logit'):
        super().__init__(data, formula, reset_data_index=True)
        
        # Fit a GLM model, returning the model fit
        # Assign resulting propensity scores to `self.scores`
        def fit_glm(Y, X, link):
            link_map = {'logit': sm.families.links.logit(),
                        'probit': sm.families.links.probit()}
            
            mod = sm.GLM(Y, X, family=sm.families.Binomial(link=link_map.get(link, 'logit')))

            return mod.fit(), mod.endog_names, mod.exog_names[1:]

        self.score_fit, self.Yvar, self.Xvar = fit_glm(self.Y, self.X, link) 
        self.scores = self.score_fit.predict(self.X)


    # Prep data prior to implementing matching procedure
    def _initialize_matching(self, match_to):
        
        # Fetch the propensity scores
        # Re-order each Series: matching will be attempted first on observations with higher propensity scores
        # This allows the units that would have the hardest time finding close matches to be matched first (Rubin 1973)
        treat_ps = self.scores[self.data[self.Yvar] == 1].sort_values(ascending=False)
        ctrl_ps = self.scores[self.data[self.Yvar] == 0].sort_values(ascending=False)

        # One propensity score Series will be assigned to `mto`, the other to `mfrom`
        # This informs which set of scores will have primacy: candidates for matches will be drawn _from_ `mfrom` and matched _to_ a candidate in `mto`
        mto = {'group_name': None, 'scores': None}
        mfrom = mto.copy()

        match_to = match_to.lower()

        if match_to in ('treat', 'treatment', 'test'):
            mto['group_name'], mto['scores'] = 'treat', treat_ps
            mfrom['group_name'], mfrom['scores'] = 'ctrl', ctrl_ps
        
        elif match_to in ('control', 'ctrl'):
            mto['group_name'], mto['scores'] = 'ctrl', ctrl_ps
            mfrom['group_name'], mfrom['scores'] = 'treat', treat_ps
        
        elif match_to in ('min', 'minority'):
            score_n = np.array([treat_ps.size, ctrl_ps.size])

            mto['scores'], mto['group_name'] = (treat_ps, 'treat') if score_n.min() == treat_ps.size else (ctrl_ps, 'ctrl')
            mfrom['scores'], mfrom['group_name'] = (treat_ps, 'treat') if mto['group_name'] == 'ctrl' else (ctrl_ps, 'ctrl')

        else:
            raise AssertionError("Invalid `match_to` parameter. Must be one of ('treat' / 'treatment' / 'test', 'control' / 'ctrl', or 'minority' / 'min')")

        return mto, mfrom


    def get_matches_nearest(self, replace=False, nmatches: int=1, seed=None, match_to='treat'):

        mto, mfrom = self._initialize_matching(match_to)

        match_idxs = []
        matches = pd.DataFrame()
        
        for i in range(int(nmatches)):
            # Candidates for matches will be drawn from `mfrom` and matched to a candidate in `mto`
            for mto_idx in mto['scores'].index:
                
                # Handle matching w/ vs w/o replacement
                if replace:
                    mfrom_mask = mfrom['scores']
                else:
                    mfrom_mask = mfrom['scores'][~mfrom['scores'].index.isin(match_idxs)]

                # Stop matching once there are no more eligible matches in `mto`
                if len(mfrom_mask) == 0:
                    break
                else:
                    pass

                # Find the observation(s) in `mfrom` whose propensity score is the closet (Euclidean distance) to the match target from `mto`
                dist = abs(mfrom_mask - mto['scores'][mto_idx])
                min_dist = dist.min()
                matchable = dist[dist == min_dist].index

                # If there is more than one `mfrom` candidate for the current round of matching, try to avoid re-selecting a candidate that has already been successfully matched
                # This scenario should only occur when `replace`=True
                if any(matchable.isin(match_idxs)) and not all(matchable.isin(match_idxs)):
                    matchable = matchable[~matchable.isin(match_idxs)]
                else:
                    pass
                
                # If there is more than one `mfrom` candidate for the current round of matching, select one at random
                np.random.seed(seed)
                mfrom_idx = np.random.choice(matchable)

                # When `replace`=True, `mfrom` observations may appear more than once in the final matched dataset
                # However, `mto` observations should only ever appear once
                if mto_idx not in match_idxs:
                    match_idxs.append(mto_idx)

                match_idxs.append(mfrom_idx)

                # Create a DataFrame with details about matched pairs 
                match_cols = [mto_idx, mfrom_idx, mto['scores'][mto_idx], mfrom['scores'][mfrom_idx], min_dist]
                match_colnames = [mto['group_name']+'_idx', mfrom['group_name']+'_idx', mto['group_name']+'_score', mfrom['group_name']+'_score', 'abs_dist']
                match_row = pd.DataFrame([match_cols], columns=match_colnames)
                matches = pd.concat([matches, match_row], ignore_index=True)

        matched_data = self.data.iloc[match_idxs]

        return _Matcher(data = matched_data,
                        formula=self.formula,
                        matches=matches,
                        orig_balance=self.balance)
