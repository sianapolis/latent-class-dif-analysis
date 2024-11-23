from src.functions.difalgorithm import DIFAlgorithm
from src.utils.plots import Plot
from src.utils.compare import Compare

import pandas as pd


def run():
    # setting up hyperparameters
    respondents = 50
    items = 50
    difitems = 20
    reps = 50
    class_thresh = 0.5
    outlier_proba = 0.4
    step = 1
    max_iter = 100
    tol = 1e-6
    seed = 776155

    # Running the function
    auc, mse, bic = DIFAlgorithm(respondents, items, difitems, reps,
                                 class_thresh, outlier_proba, step, max_iter, tol, seed).run()

    # Plotting the results
    Plot().plot_auc(auc[0], auc[1])
    Plot().plot_MSE(mse)

    # Setting up hyperparameters for comparison
    n = 1000
    n_replications = 50
    class_threshold = 0.5
    max_iter = 100
    tol = 1e-6

    # Change hyperparameters as you wish
    pi_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    j_list = [10, 100, 150, 200]
    p_list = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]

    results = Compare(nrespondents=n, nreps=n_replications, class_threshold=class_threshold, max_iteration=max_iter,
                      tolerance=tol, pi_list=pi_list, j_list=j_list, p_list=p_list).main()
    results.to_csv('Comparison_results.csv', index=False)

    return None


if __name__ == '__main__':
    run()
