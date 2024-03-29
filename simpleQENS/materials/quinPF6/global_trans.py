# functions to do an initial least-squares fit of the data

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
from simpleQENS.materials.quinPF6 import transModel


def make_global_model(temps, res, Qvalues, init_params):
    """ Put all individual spectrum model into one big model and set of parameters.
    """
    n_spectra = len(Qvalues)
    l_models = list()
    g_params = lmfit.Parameters()

    # make global fwhm with Arrhenius behaviour. Ea is in electron volt.
    # fwhm = fwhm_0 * exp(-Ea/kT)
    g_params.add('fwhm_trans_a_0', value=init_params['fwhm_trans_a_0'], min=0.0)
    g_params.add('fwhm_trans_a_Ea', value=init_params['fwhm_trans_a_Ea'])
    g_params.add('fwhm_trans_l', value=init_params['fwhm_trans_l'], min=0.0)  # also set jump distance globally

    for temp in temps:

        print('making global model for temp {}'.format(temp))

        # make global fwhm for rotational process and tie to Arrhenius behaviour
        # kB in meV, 1/kB = 11.6049669 K/meV
        g_params.add('fwhm_trans_a_{}'.format(temp), min=0.0,
                     expr='fwhm_trans_a_0 * exp(-11.6049669 * fwhm_trans_a_Ea/({}))'.format(temp))

        temp_models = list()
        for i in range(0, n_spectra):
            # model and parameters for one of the spectra
            spectrum_index = '{}_{}K'.format(i, temp)  # add temperature to prefix
            m, ps = transModel.generate_model_and_params_trans(res, spectrum_index=spectrum_index, init_vals=init_params)
            ps['l_{}_sigma'.format(spectrum_index)].set(
                expr='0.50000*fwhm_trans_a_{}*(1-sin(fwhm_trans_l*{})/({}*fwhm_trans_l))'.format(temp, Qvalues[i],
                                                                                                 Qvalues[
                                                                                                     i]))  # translational fwhm = a* (1-sin(l * Q)/(lQ))
            ps['l_{}_amplitude'.format(spectrum_index)].set(value=init_params['I'])
            # l2 is the rotational and translational lorentzian convolved,
            # which gives a lorentzian with fwhm = fwhm_rot + fwhm_trans
            temp_models.append(m)
            for p in ps.values():
                g_params.add(p)

        l_models.append(temp_models)

    return l_models, g_params


def get_fit(data, resolution, temps, init_params=None, init_fixes=None, minim_method='leastsq'):
    """ Fit the data to the translational model.
    Parameters
    ----------
    data: list of dicts
        one dict for each temperature corresponding to temps.
        dictionary containing the data, with keys 'I', 'E', 'Q', 'dI'.
    resolution: dict
        dictionary containing the resolution data, same keys as data.
    temps: list
        containing temperatures (int or float)
    init_params: dict
        dictionary containing one or more of the initial values you might want to set
        can define fwhm_rot_0, fwhm_rot_Ea, fwhm_trans_l, fwhm_trans_a_0, fwhm_trans_a_Ea.
        Ea is in electronVolt!
        If you want to replace (one of the) the defaults.
        You can now also define any of the other parameters in here, such as 'I_0_c' etc.
    init_fixes: list
        dictionary containing names of the parameters that you want to fix to their initial values.
        such as 'fwhm_rot'
    minim_method: string
        minimisation method (see https://lmfit.github.io/lmfit-py/fitting.html for options)

    Returns
    -------
    MinimizerResult, globalModel, minimizer
    """

    default_params = {'fwhm_trans_a_0': 0.1, 'fwhm_trans_a_Ea': 12.0, 'fwhm_trans_l': 3.0,
                      'I': 1.0,
                      }

    # set custom parameters if given
    if init_params is not None:
        for param in init_params.keys():
            default_params[param] = init_params[param]

    l_models, g_params = make_global_model(temps, resolution, data[0]['Q'], default_params)

    if init_fixes is not None:
        for init_fix_param in init_fixes:
            g_params[init_fix_param].set(vary=False)

    global_fit, minimizer = leastSquares.minim_globaltemp(g_params, data, temps, l_models, method=minim_method)
    return global_fit, l_models, minimizer
