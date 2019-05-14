from emukit.examples.profet.meta_benchmarks import meta_forrester, meta_svm, meta_xgboost, meta_fcnet


def test_meta_forrester():

    fcn, cs = meta_forrester("./profet_data/samples/forrester/sample_objective_0.pkl")

    x = cs.sample_uniform(10)
    y = fcn(x)

    assert y.shape[0] == 10
    assert y.shape[1] == 1


def test_meta_svm():

    fcn, cs = meta_svm("./profet_data/samples/svm/sample_objective_0.pkl")

    x = cs.sample_uniform(10)
    y, c = fcn(x)

    assert y.shape[0] == 10
    assert y.shape[1] == 1
    assert c.shape[0] == 10
    assert c.shape[1] == 1


def test_meta_fcnet():

    fcn, cs = meta_fcnet("./profet_data/samples/fcnet/sample_objective_0.pkl")

    x = cs.sample_uniform(10)
    y, c = fcn(x)

    assert y.shape[0] == 10
    assert y.shape[1] == 1
    assert c.shape[0] == 10
    assert c.shape[1] == 1


def test_meta_xgboost():

    fcn, cs = meta_xgboost("./profet_data/samples/xgboost/sample_objective_0.pkl")

    x = cs.sample_uniform(10)
    y, c = fcn(x)

    assert y.shape[0] == 10
    assert y.shape[1] == 1
    assert c.shape[0] == 10
    assert c.shape[1] == 1
