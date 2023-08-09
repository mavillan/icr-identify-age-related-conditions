# %% [code]
import numpy as np
from functools import partial
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def balanced_logloss_(y_pred, y_true, eps=1e-7):
    n0 = np.sum(1-y_true)
    n1 = np.sum(y_true)
    p1 = np.clip(y_pred, eps, 1-eps)
    p0 = 1-p1
    log_loss0 = - np.sum((1-y_true) * np.log(p0)) / (n0+eps)
    log_loss1 = - np.sum(y_true * np.log(p1)) / (n1+eps)
    return (log_loss0 + log_loss1)/2


def calibrate_probs(
        prob_class1,
        epsilon,
        alpha,
        beta
    ):
    p0 = (1-prob_class1).copy()
    p1 = prob_class1.copy()

    mask0 = (p1 < epsilon)
    mask1 = (p1 >= epsilon)    
    # scale down p1 below epsilon
    p1[mask0] *= alpha
    # scale up p1 above epsilon
    p1[mask1] *= beta
    # normalize probs
    normalized_p1 = p1 / (p0+p1)
    return normalized_p1
  

def calculate_overall_metric(
        oof_dfs:list, 
        epsilon:float,
        alpha:float,
        beta:float,
    ) -> float:
    all_metrics = list()
    for oof in oof_dfs:
        calib_p1 = calibrate_probs(
            oof.pred_proba.values,
            epsilon,
            alpha,
            beta
        )
        metric = balanced_logloss_(calib_p1, oof.Class.values)
        all_metrics.append(metric)
    return np.mean(all_metrics)


def objective(oof_dfs, trial):
    epsilon = trial.suggest_float("epsilon", 0.01, 0.99)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 1., 10.)
    return calculate_overall_metric(oof_dfs, epsilon, alpha, beta)


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
