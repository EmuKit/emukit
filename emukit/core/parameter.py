class Parameter(object):
    def transform_to_model(self, x):
        return x

    def transform_to_user_function(self, x):
        return x

    @property
    def model_dim(self):
        return 1