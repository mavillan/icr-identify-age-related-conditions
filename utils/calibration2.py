import numpy as np
from functools import partial
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

#########################################################
# competition metric
#########################################################

def balanced_logloss_(y_pred, y_true, eps=1e-7):
    n0 = np.sum(1-y_true)
    n1 = np.sum(y_true)
    p1 = np.clip(y_pred, eps, 1-eps)
    p0 = 1-p1
    log_loss0 = - np.sum((1-y_true) * np.log(p0)) / (n0+eps)
    log_loss1 = - np.sum(y_true * np.log(p1)) / (n1+eps)
    return (log_loss0 + log_loss1)/2

#########################################################
# magic functions
#########################################################

def exp_func(x):
    return np.exp(-1/x)

def psi_func(x):
    out = np.empty(x.size)
    
    mask0 = (x <= 0)
    mask1 = (x >= 1)
    mask01 = np.logical_and(x>0, x<1)
    
    out[mask0] = 0
    out[mask1] = 1
    out[mask01] = exp_func(x[mask01]) / (exp_func(x[mask01]) + exp_func(1-x[mask01]))
    
    return out

def scale_func(x, alpha, beta, gamma):
    return (alpha-gamma) * psi_func(beta * x) + gamma


#########################################################
# calibration functions
#########################################################

def calibrate_probs(
        prob_class1,
        alpha,
        beta,
        gamma,
    ):
    p0 = (1-prob_class1).copy()
    p1 = prob_class1.copy()

    scale_values = scale_func(p1, alpha, beta, gamma)
    p1 *= scale_values

    normalized_p1 = p1 / (p0+p1)
    return normalized_p1
  

def calculate_overall_metric(
        oof_dfs:list, 
        alpha:float,
        beta:float,
        gamma:float,
    ) -> float:
    all_metrics = list()
    for oof in oof_dfs:
        calib_p1 = calibrate_probs(
            oof.pred_proba.values,
            alpha,
            beta,
            gamma,
        )
        metric = balanced_logloss_(calib_p1, oof.Class.values)
        all_metrics.append(metric)
    return np.mean(all_metrics)


def objective(oof_dfs, trial):
    alpha = trial.suggest_float("alpha", 1., 10.)
    beta = trial.suggest_float("beta", 1., 10.)
    gamma = trial.suggest_float("gamma", 0.001, 0.999)
    return calculate_overall_metric(oof_dfs, alpha, beta, gamma)


def optimize_calibration(oof_dfs):
    study = optuna.create_study(
        direction='minimize',
    )
    study.optimize(
        partial(objective, oof_dfs), 
        n_trials=1000,
        n_jobs=1, 
        gc_after_trial=True,
    )
    print("best_value:", study.best_value)
    print("best_params:", study.best_params)
    return dict(study.best_params)
