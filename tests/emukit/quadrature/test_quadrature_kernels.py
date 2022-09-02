# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad, sample_uniform

from emukit.model_wrappers.gpy_quadrature_wrappers import (
    BrownianGPy,
    ProductBrownianGPy,
    ProductMatern12GPy,
    ProductMatern32GPy,
    ProductMatern52GPy,
    RBFGPy,
)
from emukit.quadrature.interfaces import IStandardKernel
from emukit.quadrature.kernels import (
    QuadratureBrownianLebesgueMeasure,
    QuadratureProductBrownianLebesgueMeasure,
    QuadratureProductMatern12LebesgueMeasure,
    QuadratureProductMatern32LebesgueMeasure,
    QuadratureProductMatern52LebesgueMeasure,
    QuadratureRBFGaussianMeasure,
    QuadratureRBFLebesgueMeasure,
)
from emukit.quadrature.measures import GaussianMeasure, LebesgueMeasure


# the following classes and functions are also used to compute the ground truth integrals with MC
@dataclass
class DataBox:
    D = 2
    bounds = [(-1, 2), (-3, 3)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[-1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[-1, 1], [0, 0.2], [0.8, -0.1], [1.3, 2.8]])
    N = 3
    M = 4
    dat_bounds = bounds


@dataclass
class DataBoxPositiveDomain:
    D = 2
    bounds = [(0.1, 2), (0.2, 3)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[0.3, 1], [0.8, 0.5], [1.5, 2.8]])
    x2 = np.array([[0.25, 1.1], [0.5, 0.3], [1.0, 1.3], [1.2, 1.2]])
    N = 3
    M = 4
    dat_bounds = bounds


@dataclass
class DataIntervalPositiveDomain:
    D = 1
    bounds = [(0.2, 1.6)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[0.3], [0.8], [1.5]])
    x2 = np.array([[0.25], [0.5], [1.0], [1.2]])
    N = 3
    M = 4
    dat_bounds = bounds


@dataclass
class DataGaussianSpread:
    D = 2
    measure_mean = np.array([0.2, 1.3])
    measure_var = np.array([0.3, 1.4])
    x1 = np.array([[-1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[-1, 1], [0, 0.2], [0.8, -0.1], [1.3, 2.8]])
    N = 3
    M = 4
    dat_bounds = [(m - 2 * np.sqrt(v), m + 2 * np.sqrt(v)) for m, v in zip(measure_mean, measure_var)]


@dataclass
class EmukitRBF:
    variance = 0.5
    lengthscales = np.array([0.8, 1.3])
    kern = RBFGPy(GPy.kern.RBF(input_dim=2, lengthscale=lengthscales, variance=variance, ARD=True))


@dataclass
class EmukitProductMatern12:
    variance = 0.7
    lengthscales = np.array([0.4, 1.2])
    kern = ProductMatern12GPy(lengthscales=lengthscales)


@dataclass
class EmukitProductMatern32:
    variance = 0.7
    lengthscales = np.array([0.4, 1.2])
    kern = ProductMatern32GPy(lengthscales=lengthscales)


@dataclass
class EmukitProductMatern52:
    variance = 0.7
    lengthscales = np.array([0.4, 1.2])
    kern = ProductMatern52GPy(lengthscales=lengthscales)


@dataclass
class EmukitBrownian:
    variance = 0.5
    kern = BrownianGPy(GPy.kern.Brownian(input_dim=1, variance=variance))


@dataclass
class EmukitProductBrownian:
    variance = 0.7
    offset = -1.8
    kern = ProductBrownianGPy(variance=variance, input_dim=2, offset=offset)


# gaussian
def get_gaussian_qrbf():
    dat = DataGaussianSpread()
    measure = GaussianMeasure(mean=dat.measure_mean, variance=dat.measure_var)
    qkern = QuadratureRBFGaussianMeasure(EmukitRBF().kern, measure=measure)
    return qkern, dat


# lebesgue
def get_lebesgue_qrbf():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=False)
    qkern = QuadratureRBFLebesgueMeasure(EmukitRBF().kern, measure=measure)
    return qkern, dat


def get_lebesgue_qmatern12():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=False)
    qkern = QuadratureProductMatern12LebesgueMeasure(EmukitProductMatern12().kern, measure=measure)
    return qkern, dat


def get_lebesgue_qmatern32():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=False)
    qkern = QuadratureProductMatern32LebesgueMeasure(EmukitProductMatern32().kern, measure=measure)
    return qkern, dat


def get_lebesgue_qmatern52():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=False)
    qkern = QuadratureProductMatern52LebesgueMeasure(EmukitProductMatern52().kern, measure=measure)
    return qkern, dat


def get_lebesgue_qbrownian():
    dat = DataIntervalPositiveDomain()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=False)
    qkern = QuadratureBrownianLebesgueMeasure(EmukitBrownian().kern, measure=measure)
    return qkern, dat


def get_lebesgue_qprodbrownian():
    dat = DataBoxPositiveDomain()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=False)
    qkern = QuadratureProductBrownianLebesgueMeasure(EmukitProductBrownian().kern, measure=measure)
    return qkern, dat


# lebesgue normalized
def get_lebesgue_normalized_qrbf():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=True)
    qkern = QuadratureRBFLebesgueMeasure(EmukitRBF().kern, measure=measure)
    return qkern, dat


def get_lebesgue_normalized_qmatern12():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=True)
    qkern = QuadratureProductMatern12LebesgueMeasure(EmukitProductMatern12().kern, measure=measure)
    return qkern, dat


def get_lebesgue_normalized_qmatern32():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=True)
    qkern = QuadratureProductMatern32LebesgueMeasure(EmukitProductMatern32().kern, measure=measure)
    return qkern, dat


def get_lebesgue_normalized_qmatern52():
    dat = DataBox()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=True)
    qkern = QuadratureProductMatern52LebesgueMeasure(EmukitProductMatern52().kern, measure=measure)
    return qkern, dat


def get_lebesgue_normalized_qbrownian():
    dat = DataIntervalPositiveDomain()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=True)
    qkern = QuadratureBrownianLebesgueMeasure(EmukitBrownian().kern, measure=measure)
    return qkern, dat


def get_lebesgue_normalized_qprodbrownian():
    dat = DataBoxPositiveDomain()
    measure = LebesgueMeasure.from_bounds(bounds=dat.bounds, normalized=True)
    qkern = QuadratureProductBrownianLebesgueMeasure(EmukitProductBrownian().kern, measure=measure)
    return qkern, dat


# == fixtures Gaussian start here
@pytest.fixture
def gaussian_qrbf():
    qkern, dat = get_gaussian_qrbf()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


# == fixtures Lebesgue start here
@pytest.fixture
def lebesgue_qrbf():
    qkern, dat = get_lebesgue_qrbf()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_qmatern12():
    qkern, dat = get_lebesgue_qmatern12()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_qmatern32():
    qkern, dat = get_lebesgue_qmatern32()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_qmatern52():
    qkern, dat = get_lebesgue_qmatern52()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_qbrownian():
    qkern, dat = get_lebesgue_qbrownian()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_qprodbrownian():
    qkern, dat = get_lebesgue_qprodbrownian()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


# == fixtures Lebesgue normalized start here
@pytest.fixture
def lebesgue_normalized_qrbf():
    qkern, dat = get_lebesgue_normalized_qrbf()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_normalized_qmatern12():
    qkern, dat = get_lebesgue_normalized_qmatern12()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_normalized_qmatern32():
    qkern, dat = get_lebesgue_normalized_qmatern32()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_normalized_qmatern52():
    qkern, dat = get_lebesgue_normalized_qmatern52()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_normalized_qbrownian():
    qkern, dat = get_lebesgue_normalized_qbrownian()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def lebesgue_normalized_qprodbrownian():
    qkern, dat = get_lebesgue_normalized_qprodbrownian()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


gaussian_embeddings_test_dict = {
    "qrbf": lazy_fixture("gaussian_qrbf"),
}

lebesgue_embeddings_test_dict = {
    "qrbf": lazy_fixture("lebesgue_qrbf"),
    "qmatern12": lazy_fixture("lebesgue_qmatern12"),
    "qmatern32": lazy_fixture("lebesgue_qmatern32"),
    "qmatern52": lazy_fixture("lebesgue_qmatern52"),
    "qbrownian": lazy_fixture("lebesgue_qbrownian"),
    "qprodbrownian": lazy_fixture("lebesgue_qprodbrownian"),
}

lebesgue_normalized_embeddings_test_dict = {
    "qrbf": lazy_fixture("lebesgue_normalized_qrbf"),
    "qmatern12": lazy_fixture("lebesgue_normalized_qmatern12"),
    "qmatern32": lazy_fixture("lebesgue_normalized_qmatern32"),
    "qmatern52": lazy_fixture("lebesgue_normalized_qmatern52"),
    "qbrownian": lazy_fixture("lebesgue_normalized_qbrownian"),
    "qprodbrownian": lazy_fixture("lebesgue_normalized_qprodbrownian"),
}

embeddings_test_list = (
    list(gaussian_embeddings_test_dict.values())
    + list(lebesgue_embeddings_test_dict.values())
    + list(lebesgue_normalized_embeddings_test_dict.values())
)


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D, _ = kernel_embedding

    # kernel shapes
    assert emukit_qkernel.K(x1, x2).shape == (N, M)

    # embedding shapes
    assert emukit_qkernel.qK(x2).shape == (1, M)
    assert emukit_qkernel.Kq(x1).shape == (N, 1)
    assert np.shape(emukit_qkernel.qKq()) == ()
    assert isinstance(emukit_qkernel.qKq(), float)


# === tests for kernel embedding start here


@pytest.mark.parametrize(
    "kernel_embedding,interval",
    [
        (gaussian_embeddings_test_dict["qrbf"], [0.22019471616760106, 0.22056701590213823]),
        (lebesgue_embeddings_test_dict["qrbf"], [38.267217898004176, 38.32112525041843]),
        (lebesgue_embeddings_test_dict["qmatern12"], [23.989038590657223, 24.018860660258476]),
        (lebesgue_embeddings_test_dict["qmatern32"], [33.68147561344138, 33.72674814040212]),
        (lebesgue_embeddings_test_dict["qmatern52"], [36.31179230918318, 36.36244064795965]),
        (lebesgue_embeddings_test_dict["qbrownian"], [0.6527648875308305, 0.6539297075650101]),
        (lebesgue_embeddings_test_dict["qprodbrownian"], [147.14888857464945, 147.3118404349691]),
        (lebesgue_normalized_embeddings_test_dict["qrbf"], [0.11810869721606222, 0.11827507793339022]),
        (lebesgue_normalized_embeddings_test_dict["qmatern12"], [0.07404024256375687, 0.07413228598845208]),
        (lebesgue_normalized_embeddings_test_dict["qmatern32"], [0.10395517164642398, 0.10409490166790775]),
        (lebesgue_normalized_embeddings_test_dict["qmatern52"], [0.11207343305303447, 0.11222975508629518]),
        (lebesgue_normalized_embeddings_test_dict["qbrownian"], [0.3330433099647094, 0.3336376059005152]),
        (lebesgue_normalized_embeddings_test_dict["qprodbrownian"], [5.199166451419295, 5.204923979414083]),
    ],
)
def test_qkernel_qKq(kernel_embedding, interval):
    # To test the integral value of the kernel embedding, we check if it lies in some confidence interval.
    # These intervals were computed as follows: The kernel emukit_qkernel.qK was integrated by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a small chance that the true integrals lies outside the
    # specified intervals.
    emukit_qkernel = kernel_embedding[0]
    qKq = emukit_qkernel.qKq()
    assert interval[0] < qKq < interval[1]


@pytest.mark.parametrize(
    "kernel_embedding,intervals",
    [
        (
            gaussian_embeddings_test_dict["qrbf"],
            np.array(
                [
                    [0.13947611219369957, 0.14011691061322437],
                    [0.2451677448342577, 0.24601121206543616],
                    [0.18304341553566297, 0.1838963624576538],
                    [0.11108022737109795, 0.11170403633365646],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_dict["qrbf"],
            np.array(
                [
                    [1.5229873955135163, 1.537991178335842],
                    [2.828702323003249, 2.843692090253037],
                    [2.937171398518155, 2.9515842742084026],
                    [1.4729457662097771, 1.4865846411248254],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_dict["qmatern12"],
            np.array(
                [
                    [0.8454460426328465, 0.8580996295972143],
                    [1.6744048023965732, 1.689032824857501],
                    [1.7005411269221822, 1.7152507205903402],
                    [0.9958472553429656, 1.0089809081559042],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_dict["qmatern32"],
            np.array(
                [
                    [1.1766117360187398, 1.1935736584098973],
                    [2.3869071010842884, 2.409142999694124],
                    [2.4148816892041203, 2.437462244987469],
                    [1.3630166608625143, 1.380811565518279],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_dict["qmatern52"],
            np.array(
                [
                    [1.26728433492353, 1.2866546055251578],
                    [2.5895905423583936, 2.610587138492034],
                    [2.6142699903775943, 2.636482161926817],
                    [1.4609721776184352, 1.47994291152272],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_dict["qbrownian"],
            np.array(
                [
                    [0.1743636401938356, 0.17438715306808195],
                    [0.3273539016153389, 0.3276550528251683],
                    [0.5394612581143717, 0.5405487424242873],
                    [0.5893093858085915, 0.5906872718055994],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_dict["qprodbrownian"],
            np.array(
                [
                    [20.967420743471486, 20.980449036984133],
                    [17.640458953546258, 17.644744818957854],
                    [27.766348676441904, 27.796613394625194],
                    [28.15902659895897, 28.19052362974676],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_dict["qrbf"],
            np.array(
                [
                    [0.08461041086186201, 0.08544395435199122],
                    [0.157150129055736, 0.15798289390294645],
                    [0.1631761888065642, 0.16397690412268906],
                    [0.08183032034498759, 0.08258803561804583],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_dict["qmatern12"],
            np.array(
                [
                    [0.04696922459071368, 0.04767220164428967],
                    [0.09302248902203182, 0.09383515693652782],
                    [0.09447450705123234, 0.09529170669946335],
                    [0.05532484751905366, 0.05605449489755025],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_dict["qmatern32"],
            np.array(
                [
                    [0.06536731866770777, 0.06630964768943874],
                    [0.13260595006023823, 0.13384127776078464],
                    [0.13416009384467334, 0.13541456916597047],
                    [0.07572314782569524, 0.07671175363990439],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_dict["qmatern52"],
            np.array(
                [
                    [0.07040468527352946, 0.07148081141806432],
                    [0.143866141242133, 0.145032618805113],
                    [0.14523722168764414, 0.14647123121815653],
                    [0.08116512097880196, 0.08221905064015111],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_dict["qbrownian"],
            np.array(
                [
                    [0.12454545728131114, 0.12456225219148712],
                    [0.23382421543952772, 0.2340393234465488],
                    [0.385329470081694, 0.38610624458877657],
                    [0.42093527557756527, 0.4219194798611424],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_dict["qprodbrownian"],
            np.array(
                [
                    [3.9412445006525356, 3.9436934280045364],
                    [3.315875743147793, 3.316681356946966],
                    [5.219238473015395, 5.224927329816765],
                    [5.293050112586275, 5.298970607095257],
                ]
            ),
        ),
    ],
)
def test_qkernel_qK(kernel_embedding, intervals):
    # See test_qkernel_qKq on how the intervals were computed.
    emukit_qkernel, _, x2, _, _, _, _ = kernel_embedding
    qK = emukit_qkernel.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


# == tests for kernel gradients start here


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_gradient_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D, _ = kernel_embedding

    # gradient of kernel
    assert emukit_qkernel.dK_dx1(x1, x2).shape == (D, N, M)
    assert emukit_qkernel.dK_dx2(x1, x2).shape == (D, N, M)
    assert emukit_qkernel.dKdiag_dx(x1).shape == (D, N)

    # gradient of embeddings
    assert emukit_qkernel.dKq_dx(x1).shape == (N, D)
    assert emukit_qkernel.dqK_dx(x2).shape == (D, M)


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_gradient_values(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D, dat_bounds = kernel_embedding

    np.random.seed(42)
    x1 = sample_uniform(in_shape=(N, D), bounds=dat_bounds)
    x2 = sample_uniform(in_shape=(M, D), bounds=dat_bounds)

    # dKdiag_dx
    in_shape = x1.shape
    func = lambda x: np.diag(emukit_qkernel.K(x, x))
    dfunc = lambda x: emukit_qkernel.dKdiag_dx(x)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dK_dx1
    in_shape = x1.shape
    func = lambda x: emukit_qkernel.K(x, x2)
    dfunc = lambda x: emukit_qkernel.dK_dx1(x, x2)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dK_dx2
    in_shape = x2.shape
    func = lambda x: emukit_qkernel.K(x1, x)
    dfunc = lambda x: emukit_qkernel.dK_dx2(x1, x)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dqK_dx
    in_shape = x2.shape
    func = lambda x: emukit_qkernel.qK(x)
    dfunc = lambda x: emukit_qkernel.dqK_dx(x)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dKq_dx
    in_shape = x1.shape
    func = lambda x: emukit_qkernel.Kq(x).T
    dfunc = lambda x: emukit_qkernel.dKq_dx(x).T
    check_grad(func, dfunc, in_shape, dat_bounds)


# == tests specific to Brownian motion kernel starts here


def test_brownian_qkernel_raises():

    # measure has wrong dimensionality
    wrong_bounds = [(1, 2), (1, 2)]
    measure = LebesgueMeasure.from_bounds(bounds=wrong_bounds)
    with pytest.raises(ValueError):
        QuadratureBrownianLebesgueMeasure(BrownianGPy(GPy.kern.Brownian()), measure)

    # bounds are negative
    wrong_bounds = [(-1, 2)]
    measure = LebesgueMeasure.from_bounds(bounds=wrong_bounds)
    with pytest.raises(ValueError):
        QuadratureBrownianLebesgueMeasure(BrownianGPy(GPy.kern.Brownian()), measure)

    # bounds are smaller thn offset (product kernel)
    offset = -2
    wrong_bounds = [(offset - 1, offset + 1), (offset + 1, offset + 2)]
    measure = LebesgueMeasure.from_bounds(bounds=wrong_bounds)
    with pytest.raises(ValueError):
        QuadratureProductBrownianLebesgueMeasure(ProductBrownianGPy(input_dim=2, offset=offset), measure)


# == tests specific to mixins start here


@pytest.mark.parametrize(
    "qkernel_type",
    [
        QuadratureRBFLebesgueMeasure,
        QuadratureProductMatern32LebesgueMeasure,
        QuadratureProductMatern52LebesgueMeasure,
        QuadratureBrownianLebesgueMeasure,
        QuadratureBrownianLebesgueMeasure,
    ],
)
def test_quadrature_kernel_lebesgue_mixin(qkernel_type):
    bounds = [(1, 3)]
    kern = IStandardKernel()

    # un-normalized measure
    normalized = False
    qkern = qkernel_type.from_integral_bounds(kern, bounds, normalized)

    assert isinstance(qkern.measure, LebesgueMeasure)
    assert qkern.measure.domain.bounds == bounds
    assert qkern.measure.density == 1.0

    # normalized measure
    normalized = True
    qkern = qkernel_type.from_integral_bounds(kern, bounds, normalized)

    assert isinstance(qkern.measure, LebesgueMeasure)
    assert qkern.measure.domain.bounds == bounds
    assert qkern.measure.density == 0.5


@pytest.mark.parametrize(
    "qkernel_type",
    [
        QuadratureRBFGaussianMeasure,
    ],
)
def test_quadrature_kernel_gaussian_mixin(qkernel_type):
    mean = np.array([0.0, 1.0])
    kern = IStandardKernel()

    # diagonal covariance
    variance = np.array([1.0, 2.0])
    qkern = qkernel_type.from_measure_params(kern, mean, variance)

    assert isinstance(qkern.measure, GaussianMeasure)
    assert not qkern.measure.is_isotropic
    assert np.all(qkern.measure.mean == mean)
    assert np.all(qkern.measure.variance == variance)

    # isotropic covariance
    variance = 2.0
    qkern = qkernel_type.from_measure_params(kern, mean, variance)

    assert isinstance(qkern.measure, GaussianMeasure)
    assert qkern.measure.is_isotropic
    assert np.all(qkern.measure.mean == mean)
    assert np.all(qkern.measure.variance == np.full(mean.shape, variance))
