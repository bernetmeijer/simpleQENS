# generate model with all lorentzian contributions of adamantane

import sys
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')
from simpleQENS.fitting import leastSquares
import numpy as np
from scipy.special import spherical_jn
import lmfit
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
from qef.models.deltadirac import DeltaDiracModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.operators.convolve import Convolve


def generateModel(res_data, Qarray, n_L):
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

    # Resolution
    resolution1Dy = res_data['I'][0]  # Q, E
    resolution1Dx = res_data['E']  # Q, E

    # to do: add all contributions. then make model. then add global params. then tie global param to each sp and lor
    # put all individual models of the spectra into one global parameters.
    l_model = list()
    g_params = lmfit.Parameters()
    # make global fwhms (one for each lorentzian):
    for ll in range(n_L):
        g_params.add('fwhm_{}'.format(ll), value=0.07, min=0.0)

    # Model components
    for qq, Q in enumerate(Qarray):
        intensity = ConstantModel(prefix='I_{}_'.format(qq))  # I_amplitude
        elastic = DeltaDiracModel(prefix='e_{}_'.format(qq))  # e_amplitude, e_center
        reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_{}_'.format(qq))
        # you can vary r_centre and r_amplitude
        background = LinearModel(prefix='b_{}_'.format(qq))  # b_slope, b_intercept

        inelastic = LorentzianModel(prefix='l{}_{}_'.format(0, qq))  # create first, then add other lors
        if n_L > 1:
            for ll in range(1, n_L):
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

        # first lorentzian amplitude is 1-(the rest)
        lastamplitude = '1-e_{}_amplitude'.format(qq)
        if n_L > 1:
            for ll in range(1, n_L):
                lastamplitude += '-l{}_{}_amplitude'.format(ll, qq)
        parameters['l{}_{}_amplitude'.format(0, qq)].set(
                expr=lastamplitude)

        # tie each width to the global fwhm (remember sigma is 0.5*fwhm)
        for ll in range(n_L):
            parameters['l{}_{}_sigma'.format(ll, qq)].set(expr='0.5*fwhm_{}'.format(ll))

    return l_model, g_params


def get_initial_fit(data, resolution, Qarray, n_L):
    l_model, g_params = generateModel(resolution, Qarray, n_L)
    global_fit, minimizer = leastSquares.minim(g_params, data, l_model)
    return global_fit, l_model, minimizer
