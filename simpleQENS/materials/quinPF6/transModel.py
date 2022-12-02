# generate model with all lorentzian contributions of adamantane

import sys
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')
from simpleQENS.fitting import leastSquares
import numpy as np
import lmfit
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
from qef.models.deltadirac import DeltaDiracModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.operators.convolve import Convolve


def generate_model_and_params_trans(res_data, spectrum_index=0, init_vals=None):
    """Produce an LMFIT model with one Lorentzian and related set of fitting parameters.
    This model has 1 Lorentzians, with the width following a Chudley-Elliot model for translational diffusion.
    """

    sp = '' if spectrum_index is None else '{}_'.format(spectrum_index)  # prefix if spectrum_index passed

    # Resolution
    resolution1Dy = res_data['I'][0]  # Q, E
    resolution1Dx = res_data['E']  # Q, E

    # Model components
    inelastic = LorentzianModel(prefix='l_'+sp)  # l_amplitude, l_center, l_sigma (also l_fwhm, l_height)
    reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_'+sp)  # you can vary r_centre and r_amplitude
    background = LinearModel(prefix='b_'+sp)  # b_slope, b_intercept

    # Putting it all together
    mymodel = Convolve(reso, inelastic) + background  # I do not need to multiply by intensity -
    # since we have only one Lorentzian, the amplitude of the Lorentzian is the intensity
    parameters = mymodel.make_params()  # model parameters are a separate entity.

    # Ties and constraints
    parameters['l_'+sp+'amplitude'].set(min=0.0001)
    parameters['l_'+sp+'sigma'].set(min=0.0001)
    # allowing the HWHM to get closer to zero than this makes the EISF and QISF too correlated

    # Some initial sensible values
    if init_vals is None:
        init_vals = {'b_'+sp+'slope': 0, 'b_'+sp+'intercept': 0}
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


def make_global_model(res, Qvalues, init_params):
    """ Put all individual spectrum model into one big model and set of parameters.
    """
    n_spectra = len(Qvalues)
    l_model = list()
    g_params = lmfit.Parameters()

    # and global sine wave for fwhm of translational process
    g_params.add('fwhm_trans_a', value=init_params['fwhm_trans_a'], min=0.0)
    g_params.add('fwhm_trans_l', value=init_params['fwhm_trans_l'], min=0.0)

    for i in range(0, n_spectra):
        # model and parameters for one of the spectra
        m, ps = generate_model_and_params_trans(res, spectrum_index=i)
        ps['l_{}_sigma'.format(i)].set(expr='0.50000*fwhm_trans_a*(1-sin(fwhm_trans_l*{})/({}*fwhm_trans_l))'.format(Qvalues[i], Qvalues[i]))  # fwhm = a* (1-sin(l * Q)/(lQ))
        l_model.append(m)
        for p in ps.values():
            g_params.add(p)

    return l_model, g_params


def get_fit(data, resolution, init_params=None):
    """ Fit the data to the translational model.
    Parameters
    ----------
    data: dict
        dictionary containing the data, with keys 'I', 'E', 'Q', 'dI'.
    resolution: dict
        dictionary containing the resolution data, same keys as data.
    init_params: dict
        dictionary containing one or more of the initial values you might want to set
        can define fwhm_trans_l, fwhm_trans_a
        If you want to replace (one of the) the defaults.

    Returns
    -------
    MinimizerResult, globalModel, minimizer
    """

    default_params = {'fwhm_trans_a': 0.2, 'fwhm_trans_l': 1.5}

    # set custom parameters if given
    if init_params is not None:
        for param in init_params.keys():
            default_params[param] = init_params[param]

    l_model, g_params = make_global_model(resolution, data['Q'], default_params)
    global_fit, minimizer = leastSquares.minim(g_params, data, l_model)
    return global_fit, l_model, minimizer
