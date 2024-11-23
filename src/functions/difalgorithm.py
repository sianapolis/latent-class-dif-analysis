import numpy as np
import pandas as pd
from scipy.stats import norm

class DIFAlgorithm:
    def __init__(self,nrespondents,nitems,ndifitems,nreps,class_threshold,outlier_probability,step=0.1,max_iteration=100,tolerance=1e-6,seed=776155):
        self.n = nrespondents
        self.nitems = nitems
        self.ndifitems = ndifitems
        self.nreps = nreps
        self.class_threshold = class_threshold
        self.outlier_probability = outlier_probability
        self.step = step
        self.max_iter = max_iteration
        self.tolerance = tolerance
        self.seed = seed

    @staticmethod
    def update_outlier(post):
        return np.sum(np.mean(post[:, 32:], axis=0))

    @staticmethod
    def soft_threshold(x, lambd):
        temp = np.copy(x)
        
        temp[np.abs(x) <= lambd] = 0 
        temp[x > lambd] = temp[x > lambd] - lambd
        temp[x < -lambd] = temp[x < -lambd] + lambd
        
        return temp

    def prox_grad(self, x, grad, lbd, step):
        return self.soft_threshold(x - step * grad, lbd * step)

    @staticmethod
    def roc_curve(class_assign, xi_vec, thresholds):

        fpr = np.zeros(len(thresholds))
        tpr = np.zeros(len(thresholds))
        
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            predicted = np.where(class_assign > threshold, 1, 0)
            tp = np.sum((predicted == 1) & (xi_vec == 1))
            fp = np.sum((predicted == 1) & (xi_vec == 0))
            fn = np.sum((predicted == 0) & (xi_vec == 1))
            tn = np.sum((predicted == 0) & (xi_vec == 0))
            fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    
    @staticmethod
    def prob_calc(vec1, vec2, integral):
        temp = vec1 @ integral.T + (vec2)
        prob = 1 / (1 + np.exp(-temp))
        return prob
    
    @staticmethod
    def log_calc(x, prob):
        return x @ np.log(prob) + (1 - x) @ np.log(1 - prob)

    def llik(self,data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi):

        log_likelihood1 = self.log_calc(data,self.prob_calc(vec_a,vec_d,grid))

        grid1 = grid * sigma + mu
        log_likelihood2 = self.log_calc(data,self.prob_calc(vec_a,vec_d + vec_delta, grid1))

        max_log_likelihood = np.maximum(log_likelihood1, log_likelihood2)
        log_likelihood_combined = max_log_likelihood + np.log(
            (1 - pi) * np.exp(log_likelihood1 - max_log_likelihood) +
            pi * np.exp(log_likelihood2 - max_log_likelihood)
        )

        weighted_log_likelihood = np.dot(np.exp(log_likelihood_combined), weight)

        return np.sum(np.log(np.abs(weighted_log_likelihood)))

    def post_matr(self,data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi):
        log_likelihood1 = self.log_calc(data,self.prob_calc(vec_a,vec_d,grid))

        temp = np.exp(np.array(log_likelihood1)).T * weight
        log_likelihood1 = (temp).T * (1-pi)

        grid1 = grid * sigma + mu
        log_likelihood2 = self.log_calc(data,self.prob_calc(vec_a,vec_d + vec_delta, grid1))

        temp = np.exp(np.array(log_likelihood2)).T * weight
        log_likelihood2 = (temp).T * (pi)

        post = np.hstack((log_likelihood1, log_likelihood2))

        post = post / np.sum(post, axis=1, keepdims=True)
        return post

    def classify_respondents(self,data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi):

        post = self.post_matr(data, vec_a, vec_d, vec_delta,
                        mu, sigma, grid, weight, pi)

        return np.sum(post[:, 32:], axis=1, keepdims=True)

    def line_search(self, data, vec_a, vec_d, vec_delta, mu, sigma, grad_a, grad_d, 
                    grad_delta, grad_sigma, grad_mu, lbd, grid, weight, pi, step):

        step_size = step

        obj_value = -self.llik(
            data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi
        ) + lbd * np.sum(abs(vec_delta))

        iter = 0

        while iter <= self.max_iter:

            a_new = vec_a - step_size * grad_a
            d_new = vec_d - step_size * grad_d
            delta_new = self.prox_grad(vec_delta, grad_delta, lbd, step_size)
            sigma_new = sigma - step_size * grad_sigma
            mu_new = mu - step_size * grad_mu

            # recalculate objective function
            new_obj_value = -self.llik(
                data, a_new, d_new, delta_new, mu_new, sigma_new, grid, weight, pi
            ) + lbd * np.sum(np.abs(delta_new))

            if abs(new_obj_value - obj_value) < self.tolerance:
                return step_size

            else:
                step_size = step_size / 2
                obj_value = new_obj_value
                iter += 1

        return step_size
    
    def update_para(self, post, data, vec_a, vec_d, vec_delta, mu, sigma, lbd, step, grid, weight, pi):
        temp1 = post[:, :31]
        temp2 = post[:, 31:]

        prob = self.prob_calc(vec_a,vec_d,grid)

        grid1 = grid * sigma + mu
        prob2 = self.prob_calc(vec_a,vec_d+vec_delta,grid1)

        ## calculating gradients
        # first derivative of Q wrt slope
        grad_a = (
            data.T @ temp1 @ grid
            - prob @ (np.sum(temp1, axis=0).reshape(-1, 1) * grid)
            + data.T @ temp2 @ grid1
            - prob2 @ (np.sum(temp2, axis=0).reshape(-1, 1) * grid1)
        )
        grad_a = -grad_a / self.n

        grad_d = (
            np.sum(data.T @ temp2, axis=1).reshape(-1, 1)
            - prob @ np.sum(temp1, axis=0).reshape(-1, 1)
            + np.sum(data.T @ temp2, axis=1).reshape(-1, 1)
            - prob2 @ np.sum(temp2, axis=0).reshape(-1, 1)
        )
        grad_d = -grad_d / self.n

        # first derivative of Q wrt DIF effect
        grad_delta = np.sum(data.T @ temp2, axis=1) - prob2 @ np.sum(temp2, axis=0)
        grad_delta = -grad_delta.reshape(-1, 1) / self.n

        # first derivative of Q wrt standard deviation
        grad_sigma = vec_a.T @ (data.T @ temp2 @ grid) - vec_a.T @ prob2 @ (
            np.sum(temp2, axis=0).reshape(-1, 1) * grid
        )
        grad_sigma = -grad_sigma / self.n

        # firt derivative of Q wrt mean
        grad_mu = np.sum(vec_a.T @ (data.T @ temp2)) - vec_a.T @ prob2 @ np.sum(
            temp2, axis=0
        )
        grad_mu = -grad_mu.reshape(-1, 1) / self.n
        # use line search to find step size
        step = self.line_search(data,vec_a,vec_d,vec_delta,mu,sigma,grad_a,grad_d,grad_delta,grad_sigma,grad_mu,lbd,grid,weight,pi,step)

        # updating via gradient descent parameters
        new_a = vec_a - step * grad_a
        new_d = vec_d - step * grad_d
        new_delta = self.prox_grad(vec_delta,grad_delta,lbd,step)
        new_sigma = sigma - step * grad_sigma
        new_mu = mu - step * grad_mu

        return {
            "vec_a": new_a,
            "vec_d": new_d,
            "delta": new_delta,
            "mu": new_mu[0],
            "sigma": new_sigma[0],
        }

    def EM(self, data, vec_a, vec_d, vec_delta, mu, sigma, lbd, step, grid, weight, pi):

        obj0 = -self.llik(
            data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi
        ) + lbd * np.sum(np.abs(vec_delta))

        post = self.post_matr(data, vec_a, vec_d, vec_delta,
                        mu, sigma, grid, weight, pi)

        pi = self.update_outlier(post)

        res = self.update_para(post,data,vec_a,vec_d,vec_delta,mu,sigma,lbd,step,grid,weight,pi)

        vec_a = res["vec_a"]
        vec_d = res["vec_d"]
        vec_delta = res["delta"]
        mu = res["mu"]
        sigma = res["sigma"]

        obj1 = -self.llik(
            data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi
        ) + lbd * np.sum(np.abs(vec_delta))

        while np.abs(obj0 - obj1) > 1e-4:
            obj0 = obj1
            post = self.post_matr(data, vec_a, vec_d, vec_delta,
                            mu, sigma, grid, weight, pi)
            pi = self.update_outlier(post)

            res = self.update_para(post,data,vec_a,vec_d,vec_delta,mu,sigma,lbd,step,grid,weight,pi)
            
            vec_a = res["vec_a"]
            vec_d = res["vec_d"]
            vec_delta = res["delta"]
            mu = res["mu"]
            sigma = res["sigma"]

        obj1 = -self.llik(
            data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi
        ) + lbd * np.sum(np.abs(vec_delta))

        return {
            "vec_a": vec_a,
            "vec_d": vec_d,
            "delta": vec_delta,
            "mu": mu,
            "sigma": sigma,
            "post": post,
            "pi":pi
        }

    def line_search_conf(self,data,vec_a,vec_d,vec_delta,mu,sigma,grad_a,grad_d,grad_delta,grad_sigma,grad_mu,grid,weight,pi,step):
        step_size = step

        obj_value = -self.llik(data, vec_a, vec_d, vec_delta,
                        mu, sigma, grid, weight, pi)

        iter = 0

        while iter < self.max_iter:
            
            # gradient descent update parameters
            a_new = vec_a - step_size * grad_a
            d_new = vec_d - step_size * grad_d
            delta_new = vec_delta - step_size * grad_delta
            sigma_new = sigma - step_size * grad_sigma
            mu_new = mu - step_size * grad_mu

            # recalculate objective function
            new_obj_value = -self.llik(
                data, a_new, d_new, delta_new, mu_new, sigma_new, grid, weight, pi
            )


            if abs(new_obj_value - obj_value) < self.tolerance:
                return step_size
            else: 
                step_size = step_size / 2
                obj_value = new_obj_value
                iter += 1
        return step_size

    def update_para_conf(self,post,data,vec_a,vec_d,delvec,mu,sigma,step,grid,weight,pi):
        vec_delta = np.where(delvec != 0, delvec, 0)
        temp1 = post[:, :31]
        temp2 = post[:, 31:]

        prob = self.prob_calc(vec_a,vec_d,grid)
        grid1 = grid * sigma + mu
        prob2 = self.prob_calc(vec_a,vec_d+vec_delta,grid1)

        grad_a = (
            data.T @ temp1 @ grid
            - prob @ (np.sum(temp1, axis=0).reshape(-1, 1) * grid)
            + data.T @ temp2 @ grid1
            - prob2 @ (np.sum(temp2, axis=0).reshape(-1, 1) * grid1)
        )
        grad_a = -grad_a / self.n

        grad_d = (
            np.sum(data.T @ temp1, axis=1).reshape(-1, 1)
            - prob @ np.sum(temp1, axis=0).reshape(-1, 1)
            + np.sum(data.T @ temp2, axis=1).reshape(-1, 1)
            - prob2 @ np.sum(temp2, axis=0).reshape(-1, 1)
        )
        grad_d = -grad_d / self.n

        grad_delta = np.sum(data.T @ temp2, axis=1) - prob2 @ np.sum(temp2, axis=0)
        grad_delta = -grad_delta / self.n
        grad_delta = grad_delta.reshape(-1, 1)

        grad_sigma = vec_a.T @ (data.T @ temp2 @ grid) - vec_a.T @ prob2 @ (
            np.sum(temp2, axis=0).reshape(-1, 1) * grid
        )
        grad_sigma = -grad_sigma / self.n

        grad_mu = np.sum(vec_a.T @ (data.T @ temp2)) - vec_a.T @ prob2 @ np.sum(
            temp2, axis=0
        )
        grad_mu = -grad_mu / self.n

        step = self.line_search_conf(data,vec_a,vec_d,vec_delta,mu,sigma,grad_a,grad_d,grad_delta,grad_sigma,grad_mu,grid,weight,pi,step)

        new_a = vec_a - step * grad_a
        new_d = vec_d - step * grad_d
        new_delta = np.where(vec_delta != 0, vec_delta - step * grad_delta, 0)
        new_sigma = sigma - step * grad_sigma
        new_mu = mu - step * grad_mu

        return {
            "vec_a": new_a,
            "vec_d": new_d,
            "delta": new_delta,
            "mu": new_mu[0],
            "sigma": new_sigma[0],
        }

    def EM_conf(self,data,vec_a,vec_d,vec_delta,mu,sigma,step,grid,weight,pi):

        obj0 = -self.llik(data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi)

        post = self.post_matr(data, vec_a, vec_d, vec_delta,
                        mu, sigma, grid, weight, pi)

        pi = self.update_outlier(post)

        for k in range(5):
            res = self.update_para_conf(post,data,vec_a,vec_d,vec_delta,mu,sigma,step,grid,weight,pi)
            vec_a = res["vec_a"]
            vec_d = res["vec_d"]
            vec_delta = res["delta"]
            mu = res["mu"]
            sigma = res["sigma"]

        obj1 = -self.llik(data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi)

        while abs((obj0 - obj1)) > 1e-4:
            obj0 = obj1
            post = self.post_matr(data, vec_a, vec_d, vec_delta,
                            mu, sigma, grid, weight, pi)
            pi = self.update_outlier(post)
            for k in range(5):
                res = self.update_para_conf(post,data,vec_a,vec_d,vec_delta,mu,sigma,step,grid,weight,pi)
                vec_a = res["vec_a"]
                vec_d = res["vec_d"]
                vec_delta = res["delta"]
                mu = res["mu"]
                sigma = res["sigma"]

            obj1 = -self.llik(data, vec_a, vec_d, vec_delta,mu, sigma, grid, weight, pi)

        bic = -2 * self.llik(
            data, vec_a, vec_d, vec_delta, mu, sigma, grid, weight, pi
        ) + np.log(self.n) * (self.nitems + self.nitems + 3 + np.sum(vec_delta != 0))

        return {
            "vec_a": vec_a,
            "vec_d": vec_d,
            "delta": vec_delta,
            "mu": mu,
            "sigma": sigma,
            "bic": bic,
            "post": post,
            "pi":pi
        }

    def run(self):
        rng = np.random.default_rng(seed=self.seed)

        # Creating variables
        bias_mean = []
        bias_sd = []
        bias_pi = []
        bias_dif = np.zeros((self.nitems, self.nreps))
        bias_dvec = np.zeros((self.nitems, self.nreps))
        bias_avec = np.zeros((self.nitems, self.nreps))
        rmse_mean = []
        rmse_sd = []
        rmse_pi = []
        rmse_dif = np.zeros((self.nitems, self.nreps))
        rmse_dvec = np.zeros((self.nitems, self.nreps))
        rmse_avec = np.zeros((self.nitems, self.nreps))

        true_positive_delta = np.zeros(self.nreps)
        false_positive_delta = np.zeros(self.nreps)

        true_negative_delta = np.zeros(self.nreps)

        classification_error = np.zeros(self.nreps)
        classification_error_true = np.zeros(self.nreps)

        roc_est = []
        roc_true = []

        a_vec = np.array(rng.uniform(0.5, 1.5, self.nitems)).reshape(-1, 1)
        d_vec = np.array(rng.uniform(-2, 2, self.nitems)).reshape(-1, 1)


        delta_vec = np.zeros((self.nitems, 1))
        delta_vec[self.nitems -
                self.ndifitems:] = rng.uniform(0.5, 1.5, self.ndifitems).reshape(-1, 1)

        delta_vec = np.array(delta_vec)
        delta_vec_binary = np.zeros((self.nitems, 1))
        delta_vec_binary[delta_vec > 0] = 1


        xi_vec = np.array(rng.binomial(1, self.outlier_probability, self.n)).reshape(-1, 1)
        theta_vec = rng.normal(0, 1, self.n).reshape(-1, 1)

        mu = 0.5
        sigma = 1.5
        theta_vec[xi_vec == 1] = rng.normal(mu, sigma, sum(xi_vec == 1))

        grid = np.linspace(-4, 4, 31).reshape(-1, 1)
        grid = np.array(grid, dtype='float32')

        weight = norm.pdf(grid)
        weight = weight / np.sum(weight)
        weight = np.array(weight, dtype='float32')

        temp = theta_vec @ a_vec.T + \
                np.ones((self.n, 1)) @ d_vec.T + xi_vec @ delta_vec.T
        prob = 1 / (1 + np.exp(-temp))
        bic_list = []

        # Running the Loop
        for j in range(self.nreps):
            print(f"Running loop {j} of {self.nreps}")

            pi = self.outlier_probability
            step = self.step
            data = np.random.binomial(1, prob, size=(self.n, self.nitems))

            class_assign_tmp_true = self.classify_respondents(
                data, a_vec, d_vec, delta_vec, mu, sigma, grid, weight, pi)
            
            
            class_assign_truparval = np.where(
                class_assign_tmp_true > self.class_threshold, 1, 0)

            incorrect_predictions_true = np.sum(xi_vec != class_assign_truparval)
            total_preds = len(xi_vec)
            classification_error_true[j] = incorrect_predictions_true / total_preds

            delta_vec0 = delta_vec + \
                rng.normal(0, 0.1, self.nitems).reshape(-1, 1)
            a_vec0 = np.array(a_vec + rng.normal(0, 0.1,
                            self.nitems).reshape(-1, 1))
            d_vec0 = np.array(d_vec + rng.normal(0, 0.1,
                            self.nitems).reshape(-1, 1))
            mu0 = np.array(mu + rng.normal(0, 0.1, 1))
            sigma0 = np.array(sigma + rng.normal(0, 0.1, 1))
            lambda_vec = np.array(np.arange(0.01, 0.0005, -0.001))

            res = self.EM(data,a_vec0, d_vec0, delta_vec0, mu0, sigma0, lambda_vec[0],step,grid,weight,pi)

            bic_ls = []
            bic = float("inf")

            for i in range(1, len(lambda_vec)):
                res = self.EM(data,res["vec_a"], res["vec_d"], res["delta"], res["mu"], res["sigma"],
                        lambda_vec[i], step,grid,weight,pi)
    
                delta_lambda0 = res["delta"]
                delta_lambda0[delta_lambda0 < 0.5] = 0
                res_conf = self.EM_conf(data,res["vec_a"], res["vec_d"], delta_lambda0, res["mu"],
                                res["sigma"], step, grid, weight, pi)

                bic_new = res_conf["bic"]
                print(f"Running {i} Lambda Loop of {len(lambda_vec)}: BIC {bic} and BIC New {bic_new}")
                bic_ls.append([bic,bic_new])

                if bic_new < bic:
                    bic = bic_new
                    delta_bic = res_conf["delta"]
                    vec_a_bic = res_conf["vec_a"]
                    vec_d_bic = res_conf["vec_d"]
                    mu_bic = res_conf["mu"]
                    sigma_bic = res_conf["sigma"]
                    pi_bic = res_conf["pi"]

            class_assign_tmp = self.classify_respondents(data,vec_a_bic,vec_d_bic, delta_bic, mu_bic, sigma_bic, grid, weight, pi_bic)
            pred_label = np.where(np.sum(class_assign_tmp, axis=1, keepdims=True) > self.class_threshold, 1, 0)
            incorrect_pred = np.sum(xi_vec != pred_label)
            classification_error[j] = incorrect_pred / self.n

            delta_bic_binary = np.where(delta_bic != 0, 1, 0)

            true_positive_delta[j] = np.sum(
                (delta_vec_binary == 1) & (delta_bic_binary == 1)
            ) / np.sum(delta_vec_binary == 1)
            true_negative_delta[j] = np.sum(
                (delta_vec_binary == 0) & (delta_bic_binary == 0)
            ) / np.sum(delta_vec_binary == 0)
            false_positive_delta[j] = np.sum(
                (delta_vec_binary == 0) & (delta_bic_binary == 1)
            ) / np.sum(delta_vec_binary == 0)

            # bias
            bias_mean.append(mu_bic - mu)
            bias_sd.append(sigma_bic - sigma)
            bias_pi.append(pi_bic - pi)

            bias_dif[:, j] = delta_bic.reshape(-1,) - delta_vec.reshape(-1,)
            bias_dvec[:, j] = vec_d_bic.reshape(-1,) - d_vec.reshape(-1,)
            bias_avec[:, j] = vec_a_bic.reshape(-1,) - a_vec.reshape(-1,)

            # mse
            rmse_mean.append(np.power(mu_bic - mu, 2))
            rmse_sd.append(np.power(sigma_bic - sigma, 2))
            rmse_pi.append(np.power(pi_bic - pi, 2))
            rmse_dif[:, j] = np.power(
                delta_bic.reshape(-1,) - delta_vec.reshape(-1,), 2)
            rmse_dvec[:, j] = np.power(
                vec_d_bic.reshape(-1,) - d_vec.reshape(-1,), 2)
            rmse_avec[:, j] = np.power(
                vec_a_bic.reshape(-1,) - a_vec.reshape(-1,), 2)

            thresholds = np.linspace(0, 1, num=100)

            roc_est.append(self.roc_curve(class_assign_tmp, xi_vec, thresholds))
            roc_true.append(self.roc_curve(class_assign_tmp_true, xi_vec, thresholds))
            bic_list.append(bic_ls)
        plot_auc_list = [roc_est, roc_true]
        plot_mse_list = [bias_avec, a_vec, bias_dvec, d_vec,
                        bias_mean, mu, bias_sd, sigma, bias_dif, delta_vec, self.nitems]
        return plot_auc_list, plot_mse_list, bic_list