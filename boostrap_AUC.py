import numpy as np
from scipy.stats import norm

def calculate_Z_score(alfa):
    Z_score = norm.ppf(1-alfa/2.)
    Z_score = abs(round(Z_score, 3))

    return Z_score

def boostrap_ACC (x_train, y_train, classifier):
    random_rng = np.random.RandomState()
    idx = np.arange(y_train.shape[0])

    boostrap_train_accuracies = []
    boostrap_rounds = 10

    for i in range (boostrap_rounds):
        train_idx = random_rng.choice(idx, size = idx.shape[0], replace=True)
        valid_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

        boot_train_X, boot_train_Y = x_train.iloc[train_idx], y_train.iloc[train_idx]
        boot_valid_X, boot_valid_Y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

        classifier.fit(boot_train_X, boot_train_Y)
        acc = classifier.score(boot_valid_X,boot_valid_Y)
        boostrap_train_accuracies.append(acc)

    boostrap_train_mean = np.mean(boostrap_train_accuracies)

    alfa = 0.05
    z_value = calculate_Z_score(alfa)

    se = 0.0

    for acc in boostrap_train_accuracies:
        se += (acc - boostrap_train_mean) **2
    se = np.sqrt((1.0 / (boostrap_rounds - 1)) * se)

    ci_lenght = z_value*se

    boostrap_train_mean = round(boostrap_train_mean, 3)
    ci_lenght = round(ci_lenght, 3)

    return [boostrap_train_mean, '+/-', ci_lenght]