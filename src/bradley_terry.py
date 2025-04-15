import numpy as np
import pandas as pd

class BradleyTerry():
    """
    Bradley-Terry model for pairwise comparisons.
    Reference: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    """
    def __init__(self):
        self.p = None
        self.models = None
        self.X = None
        self.match_results = None
        self.model_to_index = None

    def fit(
        self, 
        X: pd.DataFrame, 
        model1_column: str = 'model1', 
        model2_column: str = 'model2', 
        outcome_column: str = 'outcome',
        tol: float =1e-6,
        maxiter: int =1000,
):
        """
        Estimates the Bradley-Terry model parameters.

        Args:
            X (pd.DataFrame): DataFrame containing the pairwise comparison data.
            model1_column (str): Name of the first model column.
            model2_column (str): Name of the second model column.
            outcome_column (str): Name of the outcome column.
            tol (float): Tolerance for convergence.
            maxiter (int): Maximum number of iterations to perform.
        
        Returns:
            BradleyTerry: The fitted Bradley-Terry model.
        """
        self.X = X
        self.models = set(X[model1_column]) | set(X[model2_column]) # union of both columns
        self.p = {model: 1.0 for model in self.models} # initial values
        self.match_results = self._build_match_result_matrix(model1_column, model2_column, outcome_column)
        self.model_to_index = {model: i for i, model in enumerate(self.models)}
        
        for _ in range(maxiter):
            p_new = {}

            for i, model_i in enumerate(self.models):
                num = 0.0
                denom = 0.0
                for j, model_j in enumerate(self.models):
                    if i != j:
                        w_ij = self.match_results[self.model_to_index[model_i], self.model_to_index[model_j]]
                        w_ji = self.match_results[self.model_to_index[model_j], self.model_to_index[model_i]]

                        # faster formula (https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model, Equation 5)
                        num += w_ij * self.p[model_i] / (self.p[model_i] + self.p[model_j])
                        denom += w_ji / (self.p[model_i] + self.p[model_j] + 1e-12)
                
                p_new[model_i] = num / (denom + 1e-12)
            
            p_vals = np.array(list(p_new.values()))
            # normalize the parameters
            geom_mean = np.exp(np.mean(np.log(p_vals + 1e-12)))
            for model in self.models:
                p_new[model] /= (geom_mean + 1e-12)
            
            # convergence
            old_vals = np.array(list(self.p.values()))
            new_vals = np.array(list(p_new.values()))
            if np.all(np.abs(new_vals - old_vals) < tol):
                break

            self.p = p_new
        
        return self

    def predict(self, model1: str, model2: str) -> float:
        """
        Predicts the probability of model1 winning against model2.

        Args:
            model1 (str): Name of the first model.
            model2 (str): Name of the second model.
        
        Returns:
            float: Probability of model1 winning against model2.
        """
        pi = self.p[model1]
        pj = self.p[model2]
        return pi / (pi + pj)

    def rank(self):
        """
        Ranks the models based on their estimated parameters.

        Returns:
            List[Tuple[str, float]]: List of tuples containing model names and 
            their estimated parameters, sorted by parameter values.
        """
        return sorted(self.p.items(), key=lambda x: x[1], reverse=True)

    def _build_match_result_matrix(
            self, 
            model1_column: str, 
            model2_column: str, 
            outcome_column: str) -> np.ndarray:
        """
        Builds the matrix containing the number of wins for each model against each other.

        Args:
            model1_column (str): Name of the first model column.
            model2_column (str): Name of the second model column.
            outcome_column (str): Name of the outcome column.
        
        Returns:
            np.ndarray: Matrix containing the number of wins for each model against each other.
        """
        mat = np.zeros((len(self.models), len(self.models)))

        for i, model_i in enumerate(self.models):
            for j, model_j in enumerate(self.models):
                if i != j:
                    mat[i][j] = self._get_wins_i_vs_j(model_i, model_j, model1_column, model2_column, outcome_column)
        
        return mat

    def _get_wins_i_vs_j(
            self, 
            i: str, 
            j: str, 
            model1_column: str, 
            model2_column: str, 
            outcome_column: str
    ) -> int:
        """
        Computes the number of times model i won against model j.

        Args:
            i (str): Name of the first model.
            j (str): Name of the second model.
            model1_column (str): Name of the first model column.
            model2_column (str): Name of the second model column.
            outcome_column (str): Name of the outcome column.
        
        Returns:
            int: Number of wins for model i against model j.
        """
        return np.sum((self.X[outcome_column] == i) & (self.X[model1_column] == i) & (self.X[model2_column] == j))