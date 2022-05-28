import numpy as np
import svt
from bayes_opt import BayesianOptimization
import pickle, os
import pandas as pd
import matrix_construct as MC
from sklearn.model_selection import KFold

train_triplet_path = "data/train_triplets.pkl"
test_triplet_path = "data/test_triplets.pkl"
data_size = 120

def eval_distM(M, triplets):
    """ evaluate the triplet accuracy of a distance matrix """
    correct = 0
    for a,p,n in triplets:
        if M[a,p] < M[a,n]: correct += 1
    return correct/len(triplets)

def svt_solve_bayesian_opt(X, mask, opt_iter=50):
    """ searches for the best svt hyperparameters using bayesian optimization """
    def svt_solve(threshold, eps):
        X_hat = svt.svt_solve(X, mask, threshold=threshold, eps=eps)
        return  - np.linalg.norm(X - X_hat)

    S = np.linalg.svd(X)[1]
    pbounds = {"threshold": (S.min(), S.max()/2),
        "eps": (1e-6, 1000)}

    optimizer = BayesianOptimization(f=svt_solve,pbounds=pbounds, verbose=0, random_state=1)
    optimizer.maximize(init_points=2, n_iter=opt_iter)

    threshold = optimizer.max["params"]["threshold"]
    eps = optimizer.max["params"]["eps"]
    X_hat = svt.svt_solve(X, mask, threshold=threshold, eps=eps, max_iters=1000)

    return X_hat    

def main():
    train_triplets = np.array(pickle.load(open(train_triplet_path,"rb")))
    test_triplets = np.array(pickle.load(open(test_triplet_path,"rb")))
    len_train = len(train_triplets)

    for name, construc_distM_func in zip(["incre_1", "incre_rand", "incre_pos"], [MC.incre_1, MC.incre_rand, MC.incre_pos]):
        baseline_data = {}
        acc_data = {}

        for p in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            acc = []
            baseline = []
            
            idx = np.random.choice(np.arange(len_train), int(p*len_train), replace=False)
            X, mask = construc_distM_func(train_triplets[idx], data_size)
            baseline.append(eval_distM(X, test_triplets))

            X_hat = svt_solve_bayesian_opt(X, mask)
            acc.append(eval_distM(X_hat, test_triplets))

            baseline = np.array(baseline).mean()
            acc = np.array(acc).mean()

            baseline_data[p] = [baseline]
            acc_data[p] = [acc]


        df = pd.concat([pd.DataFrame(baseline_data), pd.DataFrame(acc_data)])
        df.insert(0, "method", ["baseline","SVT"])
        df.to_csv(f"results/{name}.csv", index=False)




main()