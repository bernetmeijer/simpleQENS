# generate model with all lorentzian contributions of adamantane

from fitting import leastSquares
import numpy as np
import sys
from scipy.special import spherical_jn
import lmfit
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
from qef.models.deltadirac import DeltaDiracModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.operators.convolve import Convolve
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')

# list of all hopping distances
# CH protons
r1 = [0.0, 4.25, 4.25, 4.25, 4.25,
      4.25, 4.25, 5.205, 3.005, 5.205,
      3.005, 5.205, 3.005, 3.005, 3.005,
      3.005]
# CH2 protons
r2 = [2.486, 3.698, 3.698, 4.250, 1.764,
      4.923, 4.923, 4.429, 2.781, 4.429,
      2.781, 5.079, 4.763, 1.247, 3.481,
      3.481]


def j0(Q, r, r_idx):
    """ Returns spherical bessel function of Q*r, with r from the list
    of distances using r_idx.
    Parameters
    ----------
    Q: float
        momentum transfer
    r: list
        list of hopping distances, one of r1 or r2
    r_idx: int
        index of the rotation"""
    return spherical_jn(0, Q*r[r_idx])


def amplitude(Q, r):
    """ Calculate amplitude of the 5 (in)elastic contributions at a certain Q-value.
    Parameters
    ----------
    Q: float
        momentum transfer
    r: list
        list of hopping distances, one of r1 or r2
    """

    A = 0
    for nu in range(1, 5):
        A += j0(Q, r, nu-1)

    B = 0
    for nu in range(5, 8):
        B += j0(Q, r, nu-1)

    C = 0
    for nu in range(8, 14):
        C += j0(Q, r, nu - 1)

    D = 0
    for nu in range(14, 17):
        D += j0(Q, r, nu - 1)

    ampl1 = 1.0 / 24 * (1 + 2 * A + B + C + 2 * D)
    ampl2 = 1.0 / 24 * (1 + 2 * A + B - C - 2 * D)
    ampl3 = 1.0 / 24 * (4 - 4 * A + 4 * B)
    ampl4 = 1.0 / 24 * (9 - 3 * B + 3 * C - 6 * D)
    ampl5 = 1.0 / 24 * (9 - 3 * B - 3 * C + 6 * D)

    return np.array([ampl1, ampl2, ampl3, ampl4, ampl5])


def generateModel(res_data, Qarray):
    """ Returns model and parameters for adamantane C4 rotations.
    We fix the relative Lorentzian widths and all amplitudes. (elastic and lorentzian)
    The free parameters are now fwhm, f, m, background and overall intensity.
    Parameters
    ----------
    res_data: dict
        resolution spectra
    Qarray: bytearray
        array of Q-values of the spectra.
    """

    n_L = 4  # number of lorentzians

    # Resolution
    resolution1Dy = res_data['I'][0]  # Q, E
    resolution1Dx = res_data['E']  # Q, E

    # to do: add all contributions. then make model. then add global params. then tie global param to each sp and lor
    # put all individual models of the spectra into one global parameters.
    l_model = list()
    g_params = lmfit.Parameters()
    # make global fwhm = 1/tau_C4 parameter
    g_params.add('fwhm', value=0.07, min=0.0)
    g_params.add('f', value=0.9, min=0.0, max=1.0)
    g_params.add('m', value=0.9, min=0.0, max=1.0)

    # Model components
    for qq, Q in Qarray:
        intensity = ConstantModel(prefix='I_{}'.format(qq))  # I_amplitude
        elastic = DeltaDiracModel(prefix='e_{}'.format(qq))  # e_amplitude, e_center
        reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_{}'.format(qq))
        # you can vary r_centre and r_amplitude
        background = LinearModel(prefix='b_{}'.format(qq))  # b_slope, b_intercept

        inelastic = 0
        for ll in range(n_L):
            lor = LorentzianModel(prefix='l{}_{}_'.format(ll, qq))
            # l_amplitude, l_center, l_sigma (also l_fwhm, l_height)
            inelastic += lor

        # Putting it all together
        mymodel = intensity * Convolve(reso, elastic + inelastic) + background
        parameters = mymodel.make_params()  # model parameters are a separate entity.

        # add spectrum model to global model
        l_model.append(mymodel)
        for p in parameters.values():
            g_params.add(p)

        # set constraints
        # standard
        sp = '{}_'.format(qq)
        parameters['e_' + sp + 'amplitude'].set(min=0.0, max=1.0)
        # allowing the HWHM to get closer to zero than this makes the EISF and QISF too correlated
        parameters['r_' + sp + 'amplitude'].set(vary=False)
        for ll in range(n_L):
            parameters['l{}_'.format(ll) + sp + 'center'].set(expr='e_' + sp + 'center')  # centers tied
            parameters['l{}_'.format(ll) + sp + 'amplitude'].set(min=0.00001)
            parameters['l{}_'.format(ll) + sp + 'sigma'].set(min=0.000001)

        # Some initial sensible values
        init_vals = {'I_' + sp + 'c': 1000, 'e_' + sp + 'amplitude': 0.9,
                     'b_' + sp + 'slope': 0, 'b_' + sp + 'intercept': 0, 'e_' + sp + 'center': 0.0}
        # 'l_'+sp+'center': 0.0, 'r_'+sp+'center': 0.0}
        for p, v in init_vals.items():
            parameters[p].set(value=v)

        # OPTIONAL, if you don't want to model the background
        # parameters['b_'+sp+'slope'].set(vary=False)
        # parameters['b_'+sp+'intercept'].set(vary=False)

        # specifically for adamantane model: fix amplitudes and lorentzian widths
        amplitudes = 1.0/16*(4.0*amplitude(Q, r1)+12.0*amplitude(Q, r2))
        parameters['e_{}_amplitude'.format(qq)].set(expr='1.0-f+f*m*{}'.format(amplitudes[0]))
        for ll in range(n_L):
            parameters['l{}_{}_amplitude'.format(ll, qq)].set(expr='f*({}+1.0/{}*(1-m)*e_{}_amplitude'.format(amplitudes[ll+1], n_L, qq))
        # tie each width to the global fwhm (remember sigma is 0.5*fwhm)
        parameters['l{}_{}_sigma'.format(0, qq)].set(expr='0.5*2.0*fwhm')
        parameters['l{}_{}_sigma'.format(1, qq)].set(expr='0.5*fwhm')
        parameters['l{}_{}_sigma'.format(2, qq)].set(expr='0.5*4.0/3.0*fwhm')
        parameters['l{}_{}_sigma'.format(3, qq)].set(expr='0.5*2.0/3.0*fwhm')

    return l_model, g_params


def get_initial_fit(data, resolution, Qarray):
    l_model, g_params = generateModel(resolution, Qarray)
    global_fit, minimizer = leastSquares.minim(g_params, data, l_model)
    return global_fit, l_model, minimizer
