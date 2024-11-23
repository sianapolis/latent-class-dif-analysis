import pandas as pd
import multiprocessing as mp
import numpy as np

from src.utils.plots import Plot
from src.functions.difalgorithm import DIFAlgorithm


class Compare:
    def __init__(self, nrespondents, nreps, class_threshold, max_iteration, tolerance, pi_list, j_list, p_list, step=1, seed=776155):
        self.n = nrespondents
        self.nreps = nreps
        self.threshold = class_threshold
        self.max_iter = max_iteration
        self.tolerance = tolerance
        self.pi_list = pi_list
        self.j_list = j_list
        self.p_list = p_list
        self.step = step
        self.seed = seed
        self.plot = Plot()

    def combine_df(self, mse_list, roc_est, roc_true, x, pi, p):
        df = self.plot.get_df_MSE(mse_list)
        df["Pi"] = pi
        df["P"] = p
        auc_true, auc_est = self.plot.get_mean_auc(roc_est, roc_true)
        df["AUC est"] = auc_est
        df["AUC true"] = auc_true
        df["J_total"] = x
        return df

    def process_combination(self, params):
        jj, pis, p, self.n, self.nreps, self.threshold, self.max_iter, self.tolerance = params
        print(jj, pis, p)
        p_act = round(jj * p)
        auc_list, mse_list, bic_list = DIFAlgorithm(
            self.n, jj, p_act, self.nreps, self.threshold, pis, self.step, self.max_iter, self.tolerance, self.seed).run()
        interm_res = self.combine_df(
            mse_list, auc_list[0], auc_list[1], jj, pis, p_act)
        return interm_res

    def main(self):
        results = pd.DataFrame()

        # Prepare a list of all parameter combinations
        param_combinations = [
            (jj, pis, p, self.n, self.nreps,
             self.threshold, self.max_iter, self.tolerance)
            for jj in self.j_list
            for pis in self.pi_list
            for p in self.p_list
        ]

        # Create a pool of workers and execute the process_combination function in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            interm_results = pool.map(
                self.process_combination, param_combinations)

        # Concatenate all intermediate results
        results = pd.concat(interm_results, ignore_index=True)
        return results

    @staticmethod
    def evaluate_best(results, threshold):
        if 'Seed' in results.columns:
            parameters_evaluation = results[[
                "Pi", "J_total", "AUC est", "AUC true", "P", "Seed"]].drop_duplicates()
        else:
            parameters_evaluation = results[[
                "Pi", "J_total", "AUC est", "AUC true", "P"]].drop_duplicates()
        parameters_evaluation["AUC diff"] = parameters_evaluation["AUC true"] - \
            parameters_evaluation["AUC est"]

        parameters_evaluation_best = parameters_evaluation[
            parameters_evaluation["AUC diff"] <= threshold]
        top_10 = parameters_evaluation_best.sort_values(
            "AUC diff", ascending=True).reset_index(drop=True).iloc[:10, :]

        return top_10

    @staticmethod
    def evaluate_worst(results, threshold):
        if 'Seed' in results.columns:
            parameters_evaluation = results[[
                "Pi", "J_total", "AUC est", "AUC true", "P", "Seed"]].drop_duplicates()
        else:
            parameters_evaluation = results[[
                "Pi", "J_total", "AUC est", "AUC true", "P"]].drop_duplicates()

        parameters_evaluation["AUC diff"] = parameters_evaluation["AUC true"] - \
            parameters_evaluation["AUC est"]

        parameters_evaluation_best = parameters_evaluation[
            parameters_evaluation["AUC diff"] >= threshold]
        top_10 = parameters_evaluation_best.sort_values(
            "AUC diff", ascending=False).reset_index(drop=True).iloc[:10, :]

        return top_10

    def test_seeds(self, df, seed_list):
        final_df = pd.DataFrame()
        for s in seed_list:
            np.random.seed(s)
            for i in range(len(df)):
                print(s, i)
                auc_list, mse_list, bic_list = DIFAlgorithm(self.n, df.loc[i, "J_total"], df.loc[i, "P"], self.nreps,
                                                            self.threshold, df.loc[i, "Pi"], self.step, self.max_iter, self.tolerance, s).run()
                results_df = self.combine_df(
                    mse_list,
                    auc_list[0],
                    auc_list[1],
                    df.loc[i, "J_total"],
                    df.loc[i, "Pi"],
                    df.loc[i, "P"],
                )
                results_df["Seed"] = s
                final_df = pd.concat([final_df, results_df])
        if len(seed_list) == 1:
            return auc_list, mse_list
        else:
            return final_df.reset_index(drop=True)
