import json
import joblib
from joblib import dump, load
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import fire
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import argparse
import itertools


def regression(input_csv_path=None,
               input_csv_paths=None,
               split_seed=42,
               output_path=None,
               remove_llm_score=False,
               partial_to_train=None,
               remove_previous_turns=False,
               leave_one_arch_out_name=None,
               leave_one_task_out_name=None):


    # Read the CSV file, first read the mutliple files
    if input_csv_paths is not None:
        dfs = []
        for input_csv_path in input_csv_paths:
            print(f"Reading {input_csv_path}")
            cur_df = pd.read_csv(input_csv_path)
            dfs.append(cur_df)
        df = pd.concat(dfs)
    else:
        print(f"Detect {input_csv_path} only, use it")
        df = pd.read_csv(input_csv_path)

    # Fill NaN values with -1, try whether the model can interpret it better (no better)
    # df.fillna(-1, inplace=True)

    # Extract features and target
    if remove_previous_turns:
        df = df[~df['config_name'].str.contains('turn_1')]
        df = df[~df['config_name'].str.contains('turn_2')]

    if leave_one_arch_out_name is not None or leave_one_task_out_name is not None:
        if leave_one_arch_out_name is not None:
            X_train = df[~df['config_name'].str.contains(leave_one_arch_out_name)]
            X_test = df[df['config_name'].str.contains(leave_one_arch_out_name)]

        if leave_one_task_out_name is not None:
            X_train = df[~df['config_name'].str.contains(leave_one_task_out_name)]
            X_test = df[df['config_name'].str.contains(leave_one_task_out_name)]

        y_train = X_train['target_score']
        y_test = X_test['target_score']
        if remove_llm_score:
            drop_columns = [column for column in df.columns if "persanal_score" in column or "pagerank" in column]

            # we do this because we need to save the config_name and target_score for the scaling behaviour, so restore it
            dropped_X_test_column = X_test[['config_name', 'target_score'] + drop_columns]
            X_train = X_train.drop(columns=['config_name', 'target_score'] + drop_columns)
            X_test = X_test.drop(columns=['config_name', 'target_score'] + drop_columns)
        else:
            dropped_X_test_column = X_test[['config_name', 'target_score']]
            X_train = X_train.drop(columns=['config_name', 'target_score'])
            X_test = X_test.drop(columns=['config_name', 'target_score'])
    else:

        y = df['target_score']
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=split_seed)

        if remove_llm_score:
            drop_columns = [column for column in df.columns if "personal_score" in column or "collective_score" in column or "pagerank" in column]
            dropped_X_test_column = X_test[['config_name', 'target_score'] + drop_columns]
            X_train = X_train.drop(columns=['config_name', 'target_score'] + drop_columns)
            X_test = X_test.drop(columns=['config_name', 'target_score'] + drop_columns)
        else:
            dropped_X_test_column = X_test[['config_name', 'target_score']]
            X_train = X_train.drop(columns=['config_name', 'target_score'])
            X_test = X_test.drop(columns=['config_name', 'target_score'])


    print(f"Leave one arc out {leave_one_arch_out_name}")
    print(f"Leave one task out {leave_one_task_out_name}")
    print("partial to train rate: ", partial_to_train)
    print("pre X_train shape: ", X_train.shape)

    # Sample a subset of the training data, using partial to train rate
    if partial_to_train is not None:
        sampled_indices = X_train.sample(frac=partial_to_train, random_state=split_seed).index
        X_train = X_train.iloc[sampled_indices]
        y_train = y_train.iloc[sampled_indices]
    print("after sample X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    # Define the model and hyperparameter grid
    model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=1)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [3, 5, 7]
    }

    # Perform Grid Search to find the best hyperparameters
    print("xgboost training ...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=16, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_y_pred = best_model.predict(X_test)

    # in order to calculate the scaling behaviour of indicators, we need to find the test set that the error is less than 0.05
    error_threshold = 0.05
    errors = np.abs(y_test - best_y_pred)
    indices_for_scaling_behaviour = np.where(errors < error_threshold)[0]

    config_target = dropped_X_test_column.iloc[indices_for_scaling_behaviour]
    saved_X_test = X_test.iloc[indices_for_scaling_behaviour]
    recombined_saved_X_test = pd.concat([config_target, saved_X_test], axis=1)

    best_spearman_corr, best_p_value = spearmanr(y_test, best_y_pred)

    # After we search the best parameters, we use the best parameters to cross validate
    model.set_params(**grid_search.best_params_)
    cv_results = cross_validate(model, X_train, y_train, cv=5, n_jobs=16, return_estimator=True, scoring='neg_mean_squared_error')

    estimators = cv_results['estimator']

    test_spearmans = []
    test_pearsons = []
    test_kendalls = []
    test_rmse = []
    test_r2 = []

    # for calculating scaling behaviour of indicators
    for estimator in estimators:
        y_pred = estimator.predict(X_test)
        spearman_corr, p_value = spearmanr(y_test, y_pred)
        test_spearmans.append(spearman_corr)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_rmse.append(rmse)

        r2 = r2_score(y_test, y_pred)
        test_r2.append(r2)

        pearson = pearsonr(y_test, y_pred)
        test_pearsons.append(pearson)

        kendall = kendalltau(y_test, y_pred)
        test_kendalls.append(kendall)

    mean_test_spearmans = np.mean(test_spearmans)
    std_test_spearmans = np.std(test_spearmans)

    mean_rmse = np.mean(test_rmse)
    std_rmse = np.std(test_rmse)

    mean_r2 = np.mean(test_r2)
    std_r2 = np.std(test_r2)

    mean_pearson = np.mean(test_pearsons)
    std_pearson = np.std(test_pearsons)

    mean_kendall = np.mean(test_kendalls)
    std_kendall = np.std(test_kendalls)

    print("Spearman Rank Correlation Coefficient between y_test and predictions:", test_spearmans)
    print("RMSE between y_test and predictions:", test_rmse)
    print("R2 between y_test and predictions:", test_r2)
    print("Pearson Correlation Coefficient between y_test and predictions:", test_pearsons)
    print("Kendall Tau Correlation Coefficient between y_test and predictions:", test_kendalls)


    # Print feature importances
    feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    # solve the sklean's best model

    if partial_to_train is not None:
        output_path = os.path.join(output_path, f"{partial_to_train}")

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_path, "best_model.joblib"))
    feature_importances.to_csv(os.path.join(output_path, "feature_importances.csv"))
    recombined_saved_X_test.to_csv(os.path.join(output_path, "scaling_behaviour_test_set.csv"))
    with open(os.path.join(output_path, "best_spearman_corr.json"), "w") as f:
        json.dump({"spearman_corr": best_spearman_corr, "p_value": best_p_value, "best_param": grid_search.best_params_}, f)

    with open(os.path.join(output_path, "cross_valid_error_bar.json"), "w") as f:
        json.dump({
            "mean_test_spearmans": mean_test_spearmans,
            "std_test_spearmans": std_test_spearmans,
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_pearson": mean_pearson,
            "std_pearson": std_pearson,
            "mean_kendall": mean_kendall,
            "std_kendall": std_kendall
        }, f)

    with open(os.path.join(output_path, "cross_valid_results.json"), "w") as f:
        json.dump(test_spearmans, f)

    with open(os.path.join(output_path, "per_instance_results.json"), "w") as f:
        json.dump({"y_test": y_test.to_list(), "y_pred": best_y_pred.tolist()}, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str)
    parser.add_argument('--input_csv_paths', nargs='+', type=str)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--partial_to_train', type=float, default=None)
    parser.add_argument('--remove_llm_score', action='store_true')
    parser.add_argument('--remove_previous_turns', action='store_true')
    parser.add_argument('--leave_one_arch_out_name', type=str, default=None)
    parser.add_argument('--leave_one_task_out_name', type=str, default=None)
    args = parser.parse_args()

    regression(input_csv_path=args.input_csv_path,
               input_csv_paths=args.input_csv_paths,
               split_seed=args.split_seed,
               output_path=args.output_path,
               partial_to_train=args.partial_to_train,
               remove_llm_score=args.remove_llm_score,
               remove_previous_turns=args.remove_previous_turns,
               leave_one_arch_out_name=args.leave_one_arch_out_name,
               leave_one_task_out_name=args.leave_one_task_out_name )


