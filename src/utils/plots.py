import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Plot:
    def __init__(self):
        pass

    @staticmethod
    def MSE(results, parameter):
        rel_mse = np.zeros_like(results)
        for i in range(results.shape[0]):
            rel_mse[i, :] = (results[i, :] - parameter[i]) ** 2
        res = np.mean(rel_mse, axis=1)
        return res

    def get_df_MSE(self, mse_list):
        bias_avec = mse_list[0]
        a_vec = mse_list[1]
        bias_dvec = mse_list[2]
        d_vec = mse_list[3]
        bias_dif = mse_list[8]
        delta_vec = mse_list[9]
        j_items = mse_list[10]

        estimated_d_vec = bias_dvec + d_vec
        estimated_a_vec = bias_avec + a_vec
        estimated_delta_vec = bias_dif + delta_vec

        mse_d_vec = self.MSE(estimated_d_vec, d_vec)
        mse_a_vec = self.MSE(estimated_a_vec, a_vec)
        mse_delta_vec = self.MSE(estimated_delta_vec, delta_vec)
        df = pd.DataFrame(
            {
                "Item": np.arange(1, j_items + 1),
                "MSE d": mse_d_vec,
                "MSE a": mse_a_vec,
                "MSE delta": mse_delta_vec,
            }
        )

        return df

    @staticmethod
    def get_mean_auc(roc_est, roc_true):
        n_thresholds = 100
        fpr_list, tpr_list = [], []

        for i in range(n_thresholds):
            mean_fpr = np.mean([roc["FPR"][i] for roc in roc_est])
            mean_tpr = np.mean([roc["TPR"][i] for roc in roc_est])
            fpr_list.append(mean_fpr)
            tpr_list.append(mean_tpr)

        # Area under the curve (AUC) for the estimated model
        fpr_sorted, tpr_sorted = zip(*sorted(zip(fpr_list, tpr_list)))
        auc_est = np.trapz(tpr_sorted, fpr_sorted)

        fpr_list_true, tpr_list_true = [], []

        for i in range(n_thresholds):
            mean_fpr_true = np.mean([roc["FPR"][i] for roc in roc_true])
            mean_tpr_true = np.mean([roc["TPR"][i] for roc in roc_true])
            fpr_list_true.append(mean_fpr_true)
            tpr_list_true.append(mean_tpr_true)

        # Area under the curve (AUC) for the true model
        fpr_true_sorted, tpr_true_sorted = zip(
            *sorted(zip(fpr_list_true, tpr_list_true)))
        auc_true = np.trapz(tpr_true_sorted, fpr_true_sorted)
        return auc_true, auc_est

    @staticmethod
    def plot_auc(roc_est, roc_true):
        n_thresholds = 100
        mean_fpr_list, mean_tpr_list = [], []

        for i in range(n_thresholds):
            fpr_i = [roc["FPR"][i] for roc in roc_est]
            tpr_i = [roc["TPR"][i] for roc in roc_est]
            mean_fpr_list.append(np.mean(fpr_i))
            mean_tpr_list.append(np.mean(tpr_i))

        average_est = pd.DataFrame()
        average_est['FPR Est'] = mean_fpr_list
        average_est['TPR Est'] = mean_tpr_list
        sorted_roc = average_est.sort_values(
            by='FPR Est').reset_index(drop=True)
        auc_est = np.sum((sorted_roc['FPR Est'].values[1:] - sorted_roc['FPR Est'].values[:-1]) *
                         (sorted_roc['TPR Est'].values[1:] + sorted_roc['TPR Est'].values[:-1]) / 2)

        fpr_list_true, tpr_list_true = [], []

        for i in range(n_thresholds):
            mean_fpr_true = np.mean([roc["FPR"][i] for roc in roc_true])
            mean_tpr_true = np.mean([roc["TPR"][i] for roc in roc_true])
            fpr_list_true.append(mean_fpr_true)
            tpr_list_true.append(mean_tpr_true)

        sorted_roc_true = pd.DataFrame()
        sorted_roc_true['TPR True'] = tpr_list_true
        sorted_roc_true['FPR True'] = fpr_list_true
        sorted_roc_true = sorted_roc_true.sort_values(
            by='FPR True').reset_index(drop=True)
        auc_true = np.sum((sorted_roc_true['FPR True'].values[1:] - sorted_roc_true['FPR True'].values[:-1]) *
                          (sorted_roc_true['TPR True'].values[1:] + sorted_roc_true['TPR True'].values[:-1]) / 2)

        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr_list, mean_tpr_list,
                 label=f"Estimated ROC (AUC = {auc_est:.3f})")
        plt.plot(
            fpr_list_true,
            tpr_list_true,
            label=f"True ROC (AUC = {auc_true:.3f})",
            color="purple",
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Based on Estimated Parameters")
        plt.legend()
        plt.show()
        return None

    @staticmethod
    def plot_MSE(mse):

        df_long = mse.melt(
            id_vars="Item", var_name="MSE_type", value_name="MSE_value")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_long, x="Item", y="MSE_value", hue="MSE_type", style="MSE_type", s=100
        )
        plt.xlabel("Item")
        plt.ylabel("Mean Squared Error")
        plt.title("Mean Squared Error by Item")
        plt.legend(title="Estimate")
        plt.show()
        return None
