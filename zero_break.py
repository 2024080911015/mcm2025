import pandas as pd
import numpy as np
import statsmodels.api as sm


def zero_break(team_size,sports_count,lagged_Medals,Is_Host):
    score=2.9686 -0.4066*team_size+0.4407*sports_count
    pi=1/(1+np.exp(-score))
    lambda1=np.exp(1.6741+0.0316*lagged_Medals+1.4404*Is_Host)
    p=(1-pi)*(1-np.exp(-lambda1))
    return p


