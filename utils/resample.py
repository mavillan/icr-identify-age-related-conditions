import numpy as np
import pandas as pd
import math

def mixup(
        input_df:pd.DataFrame,
        input_cols:list,
        total_samples:int = 100_000,
        alpha:float = 0.3,
        pct_interclass:float = 0.4,
    ):
    np.random.seed(2112)
    input_df = input_df.copy()

    assert pct_interclass >= 0 and pct_interclass <=1, "pct_interclass must be between 0 and 1"

    samples_inter = int(total_samples*pct_interclass)
    samples_intra = int(total_samples*(1-pct_interclass) // 2)

    # split dataframe into class-0 and class-1
    input_df_c0 = input_df.query("Class == 0").reset_index(drop=True)
    input_df_c1 = input_df.query("Class == 1").reset_index(drop=True)

    output_dataframes = list()

    if pct_interclass < 1:

        # resample class-0 twice for mixup
        index_c0_left = np.random.choice(input_df_c0.index.values, size=samples_intra, replace=True)
        index_c0_right = np.random.choice(input_df_c0.index.values, size=samples_intra, replace=True)
        lambda_ = np.random.beta(alpha, alpha, size=samples_intra).reshape((-1,1))
        # mixup intra-class for C0
        x_intra_c0 = (
            (lambda_ * input_df_c0.loc[index_c0_left, input_cols].values) + 
            ((1-lambda_) * input_df_c0.loc[index_c0_right, input_cols].values)
        )
        y_intra_c0 = np.zeros(samples_intra)
        # put it into a dataframe
        df_intra_c0 = pd.DataFrame(x_intra_c0, columns=input_cols)
        df_intra_c0["Class"] = y_intra_c0
        output_dataframes.append(df_intra_c0)


        # resample class-1 twice for mixup
        index_c1_left = np.random.choice(input_df_c1.index.values, size=samples_intra, replace=True)
        index_c1_right = np.random.choice(input_df_c1.index.values, size=samples_intra, replace=True)
        lambda_ = np.random.beta(alpha, alpha, size=samples_intra).reshape((-1,1))
        # mixup intra-class for C1
        x_intra_c1 = (
            (lambda_ * input_df_c1.loc[index_c1_left, input_cols].values) +
            ((1-lambda_) * input_df_c1.loc[index_c1_right, input_cols].values)
        )
        y_intra_c1 = np.ones(samples_intra)
        # put it into a dataframe
        df_intra_c1 = pd.DataFrame(x_intra_c1, columns=input_cols)
        df_intra_c1["Class"] = y_intra_c1
        output_dataframes.append(df_intra_c1)

    if pct_interclass > 0:
        # resample class0 and class1 for mixup
        index_c0_left = np.random.choice(input_df_c0.index.values, size=samples_inter, replace=True)
        index_c1_right = np.random.choice(input_df_c1.index.values, size=samples_inter, replace=True)
        lambda_ = np.random.beta(alpha, alpha, size=samples_inter).reshape((-1,1))
        # mixup inter-class for C0 & C1
        x_inter = (
            (lambda_ * input_df_c0.loc[index_c0_left, input_cols].values) +
            ((1-lambda_) * input_df_c1.loc[index_c1_right, input_cols].values)
        )
        y_inter = (
            (lambda_.ravel() * input_df_c0.loc[index_c0_left, "Class"].values) +
            ((1-lambda_).ravel() * input_df_c1.loc[index_c1_right, "Class"].values)
        )
        # put it into a dataframe
        df_inter = pd.DataFrame(x_inter, columns=input_cols)
        df_inter["Class"] = y_inter
        output_dataframes.append(df_inter)

    return pd.concat(output_dataframes, ignore_index=True)
