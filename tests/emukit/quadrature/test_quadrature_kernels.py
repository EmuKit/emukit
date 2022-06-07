# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Todo: test class methods of kernels
from dataclasses import dataclass

import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad, sample_uniform

from emukit.model_wrappers.gpy_quadrature_wrappers import (
    BrownianGPy,
    ProductBrownianGPy,
    ProductMatern32GPy,
    ProductMatern52GPy,
    RBFGPy,
)
from emukit.quadrature.interfaces import IStandardKernel
from emukit.quadrature.kernels import (
    QuadratureBrownianLebesgueMeasure,
    QuadratureKernel,
    QuadratureProductBrownianLebesgueMeasure,
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
    var = 0.5
    kern = BrownianGPy(GPy.kern.Brownian(input_dim=1, variance=var))


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
def lebesque_qprodbrownian():
    qkern, dat = get_lebesgue_qprodbrownian()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


# == fixtures Lebesgue normalized start here
@pytest.fixture
def lebesgue_normalized_qrbf():
    qkern, dat = get_lebesgue_normalized_qrbf()
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
def lebesque_normalized_qprodbrownian():
    qkern, dat = get_lebesgue_normalized_qprodbrownian()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


gaussian_embeddings_test_list = [
    lazy_fixture("gaussian_qrbf"),
]

lebesgue_embeddings_test_list = [
    lazy_fixture("lebesgue_qrbf"),
    lazy_fixture("lebesgue_qmatern32"),
    lazy_fixture("lebesgue_qmatern52"),
    lazy_fixture("lebesgue_qbrownian"),
    lazy_fixture("lebesque_qprodbrownian"),
]

lebesgue_normalized_embeddings_test_list = [
    lazy_fixture("lebesgue_normalized_qrbf"),
    lazy_fixture("lebesgue_normalized_qmatern32"),
    lazy_fixture("lebesgue_normalized_qmatern52"),
    lazy_fixture("lebesgue_normalized_qbrownian"),
    lazy_fixture("lebesque_normalized_qprodbrownian"),
]

embeddings_test_list = gaussian_embeddings_test_list + lebesgue_embeddings_test_list + lebesgue_normalized_embeddings_test_list


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
        (gaussian_embeddings_test_list[0], [0.22019471616760106, 0.22056701590213823]),
        (lebesgue_embeddings_test_list[0], [38.267217898004176, 38.32112525041843]),
        (lebesgue_embeddings_test_list[1], [33.6816570527734, 33.726646173769595]),
        (lebesgue_embeddings_test_list[2], [36.311780552275614, 36.36134818184079]),
        (lebesgue_embeddings_test_list[3], [0.6528048146871609, 0.653858667201299]),
        (lebesgue_embeddings_test_list[4], [147.1346099151547, 147.3099172678195]),
        (lebesgue_normalized_embeddings_test_list[0], [0.11810869721606222, 0.11827507793339022]),
        (lebesgue_normalized_embeddings_test_list[1], [0.10395573164436231, 0.10409458695607898]),
        (lebesgue_normalized_embeddings_test_list[2], [0.11207339676628277, 0.11222638327728639]),
        (lebesgue_normalized_embeddings_test_list[3], [0.3330636809628371, 0.3336013608169892]),
        (lebesgue_normalized_embeddings_test_list[4], [5.198661947932144, 5.204856028740301]),
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
            gaussian_embeddings_test_list[0],
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
            lebesgue_embeddings_test_list[0],
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
            lebesgue_embeddings_test_list[1],
            np.array(
                [
                    [1.1759685512006572, 1.1943216573095992],
                    [2.3867473290884305, 2.4085441010798943],
                    [2.414541592697165, 2.435863015779601],
                    [1.3627594837426076, 1.3814476744322477],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_list[2],
            np.array(
                [
                    [1.26726828845331, 1.2871778987069316],
                    [2.587496144727501, 2.6113016123209984],
                    [2.6135191767118013, 2.636703247174619],
                    [1.4607696933721623, 1.4808984088931223],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_list[3],
            np.array(
                [
                    [0.17436285054037512, 0.1743870565968362],
                    [0.3273488543068163, 0.3276377242105884],
                    [0.5394358272402537, 0.5405072045782628],
                    [0.5892821601114948, 0.5906529816223602],
                ]
            ),
        ),
        (
            lebesgue_embeddings_test_list[4],
            np.array(
                [
                    [20.968328255845666, 20.979953254056394],
                    [17.64001267572194, 17.645133556673045],
                    [27.768537209548015, 27.794406087839107],
                    [28.16022039838578, 28.189287432609166],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_list[0],
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
            lebesgue_normalized_embeddings_test_list[1],
            np.array(
                [
                    [0.06533158617781429, 0.06635120318386661],
                    [0.13259707383824618, 0.1338080056155497],
                    [0.13414119959428691, 0.13532572309886673],
                    [0.07570886020792265, 0.07674709302401377],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_list[2],
            np.array(
                [
                    [0.07040379380296166, 0.07150988326149618],
                    [0.14374978581819456, 0.14507231179561106],
                    [0.1451955098173223, 0.14648351373192323],
                    [0.08115387185400903, 0.08227213382739569],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_list[3],
            np.array(
                [
                    [0.12454489324312505, 0.12456218328345438],
                    [0.23382061021915443, 0.23402694586470593],
                    [0.3853113051716099, 0.38607657469875917],
                    [0.4209158286510677, 0.42189498687311444],
                ]
            ),
        ),
        (
            lebesgue_normalized_embeddings_test_list[4],
            np.array(
                [
                    [3.9414150856852763, 3.9436002357248863],
                    [3.315791856338711, 3.3167544279460612],
                    [5.219649851418801, 5.224512422526148],
                    [5.293274510974769, 5.298738239212249],
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


def test_qkernel_uniform_finite_correct_box(qrbf_uniform_finite):
    emukit_qkernel = qrbf_uniform_finite[0]
    # integral bounds are [(-1, 2), (-3, 3)]
    # measure bounds are [(1, 2), (-4, 2)]
    # this test checks that the reasonable box is the union of those boxes
    assert emukit_qkernel.reasonable_box.bounds == [(1, 2), (-3, 2)]


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
