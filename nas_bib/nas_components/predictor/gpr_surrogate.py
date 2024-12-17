from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from nas_bib.nas_components.predictor.surrogate_model import SurrogateModel
from nas_bib.utils.registre import register_class, get_registered_class

@register_class(registry="surrogate_models")
class GaussianProcessSurrogate(SurrogateModel):
    def __init__(self):
        """
        Initialize the Gaussian process surrogate model with the specified input dimension.
        """
        super().__init__()
        self.model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-5, n_restarts_optimizer=10)

    def fit(self, X, y):
        """
        Fit the Gaussian process surrogate model to the given input-output data.
        """
        X_np = X.detach().numpy()

        self.model.fit(X_np, y)


    def predict(self, X):
        """
        Predict using the Gaussian process surrogate model.
        """     
        X_np = X.detach().numpy()
        mean, var = self.model.predict(X_np, return_std=True)
        return mean, var