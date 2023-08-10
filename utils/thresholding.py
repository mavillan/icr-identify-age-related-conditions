# %% [code]
import numpy as np
import pandas as pd

def threshold_probs(pred_proba, t_low, t_high, eps=1e-10):
    pred_proba = pred_proba.copy()
    
    mask_low = (pred_proba < t_low)
    mask_high = (pred_proba > t_high)
    
    pred_proba[mask_low] = eps
    pred_proba[mask_high] = 1-eps
    
    return pred_proba


def threshold_oof(oof_dfs, t_low, t_high):
    oof_dfs_thresholded = list()
    
    for oof in oof_dfs:
        oof = oof.copy(deep=True)
        thresh_p1 = threshold_probs(
            oof.pred_proba.values,
            t_low,
            t_high,
        )
        oof["pred_proba"] = thresh_p1
        oof_dfs_thresholded.append(oof)
        
    return oof_dfs_thresholded