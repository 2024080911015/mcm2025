import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import random
def predict_medal(Athletes,Events,Is_Host,Lagged_Share,Sigma):
    score=-0.067612+0.014247*np.log(Athletes)+0.002205*np.log(Events)+0.078906*Is_Host+0.501635*Lagged_Share+np.random.normal(loc=0, scale=Sigma)
    return max(0,score)
def predict_gold(Athletes,Events,Is_Host,Lagged_Share,Sigma):
    score = -0.109768 + \
            0.020919 * np.log(Athletes) + \
            0.002084 * np.log(Events) + \
            0.082456 * Is_Host + \
            0.608438 * Lagged_Share + \
            np.random.normal(loc=0, scale=Sigma)  # 建议去掉 size=1，直接返回浮点数
    return max(0,score)
