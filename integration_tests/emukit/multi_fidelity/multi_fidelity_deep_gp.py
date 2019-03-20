import numpy as np

from emukit.multi_fidelity.models import MultiFidelityDeepGP


def test():
    np.random.seed(123)
    x = [np.random.rand(5, 1), np.random.rand(5, 1)]
    y = [np.random.rand(5, 1), np.random.rand(5, 1)]

    model = MultiFidelityDeepGP(x, y, n_iter=20)
    model.optimize()

    x_predict = np.random.rand(10, 1)
    x_predict = np.concatenate([x_predict, np.ones((10, 1))], axis=1)
    y_mean, y_var = model.predict(x_predict)

    assert y_mean.shape == (10, 1)
    assert y_var.shape == (10, 1)
