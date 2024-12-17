from nas_bib.utils.registre import register_class


@register_class(registry="surrogate_models")
class SurrogateModel:
    def __init__(self):
        """
        Initialize the surrogate model with the specified input dimension.
        """

    def fit(self, X, y):
        """
        Fit the surrogate model to the given input-output data.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("fit method not implemented")

    def predict(self, X):
        """
        Predict using the surrogate model.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("predict method not implemented")