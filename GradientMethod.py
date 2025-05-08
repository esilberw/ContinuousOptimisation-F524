import numpy as np
import pandas as pd


class GradientMethod:

    def __init__(self, dataset_file):
        self.dataset = dataset_file
        self.X = None  # regressor
        self.y = None  # regressand
        self.beta_hat = None  # Matrix
        self.linear_regression_model = None  # Matrix representation

    def csv_to_numpy_array_dataset(self):
        df = pd.read_csv(self.dataset, sep=';')  # separaotors of the files "winequality-red" and "winequality-white is ;-symbol
        df = df.dropna()  # supprime les lignes contenant des valeurs manquantes
        self.dataset = df.astype(np.float64)

    def init_regression(self):
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
        self.X = self.dataset.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
        self.y = self.dataset.iloc[:, -1]  # Dernière colonne

        # Add a col of "1" elements x_0j correspind to Betha_0 (interception)
        self.beta_hat = np.ones(self.X.shape[1] + 1)

        

    def gradient_descent(self, t = 0.00001, n_iterations=1000000, tolerance=1e-2):
        """
        Applique la méthode de descente de gradient pour optimiser beta_hat.

        Paramètres :
        - learning_rate : float, le taux d’apprentissage.
        - n_iterations : int, nombre maximal d’itérations.
        - tolerance : float, critère d’arrêt basé sur la norme du gradient.

        Résultats :
        - Met à jour self.beta_hat avec les nouveaux coefficients optimisés.
        - Retourne l’historique du coût à chaque itération.
        """
        n_samples = self.X.shape[0]

        # Ajout du biais si nécessaire
        X_b = np.c_[np.ones(self.X.shape[0]), self.X]       
        # if self.X.shape[1] + 1 == self.beta_hat.shape[0]:
        #     X_b = np.hstack((np.ones((n_samples, 1)), self.X))
        # else:
        #     X_b = self.X

        beta = self.beta_hat.copy()
        convergence_rate = []

        for i in range(n_iterations):
            residual = self.y - X_b @ beta

            gradient = -(1 / n_samples) * (X_b.T @ residual)
            beta = beta - t * gradient

            cost = (1 / (2 * n_samples)) * np.sum(residual ** 2)
            convergence_rate.append(cost)

            if np.linalg.norm(gradient) < tolerance:
                print(f"Convergence atteinte à l’itération {i}")
                print(f"Final cost: {convergence_rate[-1]}")
                break

        self.beta_hat = beta
        print(f"Final cost: {convergence_rate[-1]}")
        return convergence_rate

    def soft_thresholding(self, beta, threshold):
        return np.sign(beta) * np.maximum(np.abs(beta) - threshold, 0)

    def proximal_gradient_descent(self, lambda_reg=0.1, t=0.01, n_iterations=1000, tolerance=1e-6):
        """
        Applique la méthode de gradient proximal pour un Lasso (régularisation L1).

        Paramètres :
        - lambda_reg : float, le coefficient de régularisation L1.
        - learning_rate : float, pas de gradient.
        - n_iterations : int, nombre max d’itérations.
        - tolerance : float, critère d’arrêt.

        Retour :
        - Met à jour self.beta_hat.
        - Retourne la liste des valeurs de coût total à chaque étape.
        """
        n_samples = self.X.shape[0]

        # Ajout du biais si nécessaire
        if self.X.shape[1] + 1 == self.beta_hat.shape[0]:
            X_b = np.hstack((np.ones((n_samples, 1)), self.X))
        else:
            X_b = self.X

        beta = self.beta_hat.copy()
        convergence_rate = []

        for i in range(n_iterations):
            # Gradient de la partie lisse
            residual = self.y - X_b @ beta
            gradient = -(1 / n_samples) * (X_b.T @ residual)

            # Descente + opérateur proximal
            beta = self.soft_thresholding(beta - t * gradient, t * lambda_reg)

            # Coût = partie lisse + régularisation
            cost = (1 / (2 * n_samples)) * np.sum(residual ** 2) + lambda_reg * np.sum(np.abs(beta))
            convergence_rate.append(cost)

            if np.linalg.norm(gradient) < tolerance:
                print(f"Convergence atteinte à l’itération {i}")
                break

        self.beta_hat = beta
        return convergence_rate

    def ista(self, lambda_reg=0.1, t=0.01, n_iterations=1000, tolerance=1e-6):
        """
        Implémente l'algorithme ISTA pour la régression Lasso.

        Paramètres :
        - lambda_reg : coefficient de régularisation L1
        - learning_rate : pas de gradient (souvent fixé à 1 / L)
        - n_iterations : nombre d’itérations maximum
        - tolerance : arrêt si le gradient devient petit

        Retour :
        - Met à jour self.beta_hat
        - Retourne l’historique du coût total
        """
        n_samples = self.X.shape[0]

        # Ajout du biais si nécessaire
        if self.X.shape[1] + 1 == self.beta_hat.shape[0]:
            X_b = np.hstack((np.ones((n_samples, 1)), self.X))
        else:
            X_b = self.X

        beta = self.beta_hat.copy()
        convergence_rate = []

        for i in range(n_iterations):
            # Gradient de la perte quadratique
            residual = self.y - X_b @ beta
            gradient = -(1 / n_samples) * (X_b.T @ residual)

            # Mise à jour ISTA
            beta_next = self.soft_thresholding(beta - t * gradient, t * lambda_reg)

            # Calcul du coût total
            cost = (1 / (2 * n_samples)) * np.sum((self.y - X_b @ beta_next) ** 2) + lambda_reg * np.sum(
                np.abs(beta_next))
            convergence_rate.append(cost)

            # Condition d'arrêt
            if np.linalg.norm(beta_next - beta, ord=2) < tolerance:
                print(f"Convergence atteinte à l’itération {i}")
                break

            beta = beta_next

        self.beta_hat = beta
        return convergence_rate

    def fista(self, lambda_reg=0.1, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        """
        Implémente l'algorithme FISTA (accéléré) pour la régression Lasso.

        Paramètres :
        - lambda_reg : régularisation L1
        - learning_rate : pas de gradient (souvent 1 / L)
        - n_iterations : nombre d’itérations
        - tolerance : critère de convergence

        Résultat :
        - Met à jour self.beta_hat
        - Retourne la liste des coûts
        """
        n_samples = self.X.shape[0]

        # Ajout du biais si nécessaire
        if self.X.shape[1] + 1 == self.beta_hat.shape[0]:
            X_b = np.hstack((np.ones((n_samples, 1)), self.X))
        else:
            X_b = self.X

        # Initialisation
        beta = self.beta_hat.copy()
        z = beta.copy()
        t = 1
        convergence_rate = []

        for i in range(n_iterations):
            # Gradient sur z
            residual = self.y - X_b @ z
            gradient = -(1 / n_samples) * (X_b.T @ residual)

            # Mise à jour beta avec l'opérateur proximal (soft-thresholding)
            beta_next = self.soft_thresholding(z - learning_rate * gradient, learning_rate * lambda_reg)

            # Mise à jour t et z (accélération de Nesterov)
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            z = beta_next + ((t - 1) / t_next) * (beta_next - beta)

            # Calcul du coût
            cost = (1 / (2 * n_samples)) * np.sum((self.y - X_b @ beta_next) ** 2) + lambda_reg * np.sum(
                np.abs(beta_next))
            convergence_rate.append(cost)

            # Condition d'arrêt
            if np.linalg.norm(beta_next - beta, ord=2) < tolerance:
                print(f"Convergence atteinte à l’itération {i}")
                break

            beta = beta_next
            t = t_next

        self.beta_hat = beta
        return convergence_rate


dataset_test = np.array([
    [1, 2, 3],   # x1=1, x2=2, y=3
    [4, 5, 6],   # x1=4, x2=5, y=6
    [7, 8, 9],   # x1=7, x2=8, y=9
])

dataset_file = "./winequality-red.csv"
gm = GradientMethod(dataset_file)
gm.csv_to_numpy_array_dataset()
#print(gm.dataset)

gm.init_regression()
#print(gm.beta_hat)
gm.gradient_descent();
#print(gm.proximal_gradient_descent())
