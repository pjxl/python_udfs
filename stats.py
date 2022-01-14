def conf_int(x, stat, se):  
  ci_lwr = x - stat * se
  ci_upr = x + stat * se
  
  return ci_lwr, ci_upr
