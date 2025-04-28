import numpy as np
class GradientMethod:

    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.X = None # regressor
        self.y = None # regressand
        self.betha = None # Matrix
        self.linear_regression_model = None # Matrix representation

    def read_dataset(self):
        return None

    def init_regression(dataset):
        """
        Paramètres :
        - dataset : numpy.ndarray of the forme (n_samples, n_features + 1),
                    où la dernière colonne est la variable cible y.

        Retour :
        - X : numpy.ndarray de forme (n_samples, n_features)
        - y : numpy.ndarray de forme (n_samples, )
        - beta_hat : numpy.ndarray de forme (n_features, )
        """
        # Supposer que dataset est une matrice numpy avec la dernière colonne = y
        X = dataset[:, :-1]  # Toutes les colonnes sauf la dernière
        y = dataset[:, -1]  # Dernière colonne

        # Add a col of "1" elements x_0j correspind to Betha_0 (interception)
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))


        # Calculer beta_hat = (X^T X)^(-1) X^T y
        beta_hat = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)

        return X, y, beta_hat
    def gradient_method(self):
        return None

    def


