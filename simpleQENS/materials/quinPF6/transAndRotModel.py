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


def generate_model_and_params_transRot(res_data, spectrum_index=0, init_vals=None):
    """Produce an LMFIT model with two Lorentzians and related set of fitting parameters.
    This model has 2 Lorentzians: one with globally fixed width (rotational) and one with free width (translational).
    We fix the global Lorentzian width for the rotational process."""

    sp = '' if spectrum_index is None else '{}_'.format(spectrum_index)  # prefix if spectrum_index passed

    # Resolution
    resolution1Dy = res_data['I'][0]  # Q, E
    resolution1Dx = res_data['E']  # Q, E

    # Model components
    intensity = ConstantModel(prefix='I_'+sp)  # I_amplitude
    elastic = DeltaDiracModel(prefix='e_'+sp)  # e_amplitude, e_center
    inelastic = LorentzianModel(prefix='l_'+sp)  # l_amplitude, l_center, l_sigma (also l_fwhm, l_height)
    inelastic2 = LorentzianModel(prefix='l2_'+sp)
    reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_'+sp)  # you can vary r_centre and r_amplitude
    background = LinearModel(prefix='b_'+sp)  # b_slope, b_intercept

    # Putting it all together
    mymodel = intensity * Convolve(reso, elastic + inelastic + inelastic2) + background
    parameters = mymodel.make_params()  # model parameters are a separate entity.

    # Ties and constraints
    parameters['e_'+sp+'amplitude'].set(min=0.005, max=1.0)
    parameters['l_'+sp+'amplitude'].set(min=0.0001)
    parameters['l2_'+sp+'amplitude'].set(min=0.0001)
    parameters['l_'+sp+'sigma'].set(min=0.0001)
    parameters['l2_'+sp+'sigma'].set(min=0.0001)
    # allowing the HWHM to get closer to zero than this makes the EISF and QISF too correlated

    parameters['l_'+sp+'center'].set(expr='e_'+sp+'center')  # centers tied
    #parameters['l_'+sp+'amplitude'].set(expr='1 - e_'+sp+'amplitude')
    parameters['l2_'+sp+'center'].set(expr='e_'+sp+'center')  # centers tied
    parameters['l2_'+sp+'amplitude'].set(expr='1 - e_'+sp+'amplitude - l_'+sp+'amplitude')

    # Some initial sensible values
    if init_vals is None:
        init_vals = {'I_'+sp+'c': 1000, 'e_'+sp+'amplitude': 0.9, 'l_'+sp+'sigma': 0.04,
                     'l2_'+sp+'sigma': 0.04,
                     'b_'+sp+'slope': 0, 'b_'+sp+'intercept': 0, 'e_'+sp+'center': 0.0}
                     #'l_'+sp+'center': 0.0, 'r_'+sp+'center': 0.0}
    for p, v in init_vals.items():
        parameters[p].set(value=v)

    # OPTIONAL, if you don't want to model the background
    #parameters['b_'+sp+'slope'].set(vary=False)
    #parameters['b_'+sp+'intercept'].set(vary=False)

    parameters['r_'+sp+'amplitude'].set(vary=False)

    #parameters['l_'+sp+'fwhm'].set(max=1.0)
    #parameters['I_'+sp+'c'].set(max=10000.0)

    return mymodel, parameters


def make_global_model(res, n_spectra, fwhm=0.1):
    """ Put all individual spectrum model into one big model and set of parameters.
    """
    l_model = list()
    g_params = lmfit.Parameters()

    # make global fwhm for rotational process
    g_params.add('fwhm', value=fwhm, min=0.0)

    for i in range(0, n_spectra):
        # model and parameters for one of the spectra
        m, ps = generate_model_and_params_transRot(res, spectrum_index=i)
        ps['l_{}_sigma'.format(i)].set(expr='0.50000*fwhm')  # fix width of translation process (1st Lorentzian)
        l_model.append(m)
        for p in ps.values():
            g_params.add(p)

    return l_model, g_params


def get_initial_fit(data, resolution, init_fwhm=0.1):
    l_model, g_params = make_global_model(resolution, len(data['Q']), fwhm=init_fwhm)
    global_fit, minimizer = leastSquares.minim(g_params, data, l_model)
    return global_fit, l_model, minimizer
