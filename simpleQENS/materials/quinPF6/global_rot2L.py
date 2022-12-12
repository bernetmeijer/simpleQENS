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


def generate_model_and_params_rot2L(res_data, spectrum_index=0, init_vals=None):
    """Produce an LMFIT model with two Lorentzians and related set of fitting parameters.
    This model has 2 Lorentzians: one with globally fixed width (rotational) and one with the width following a sine wave (translational).
    We fix the global Lorentzian width for the rotational process."""

    sp = '' if spectrum_index is None else '{}_'.format(spectrum_index)

    # Resolution
    resolution1Dy = res_data['I'][0]  # Q, E
    resolution1Dx = res_data['E']  # Q, E

    # Model components
    intensity = ConstantModel(prefix='I_' + sp)  # I_amplitude
    elastic = DeltaDiracModel(prefix='e_' + sp)  # e_amplitude, e_center
    inelastic = LorentzianModel(prefix='l_' + sp)  # l_amplitude, l_center, l_sigma (also l_fwhm, l_height)
    inelastic2 = LorentzianModel(prefix='l2_' + sp)
    reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_' + sp)  # you can vary r_centre and r_amplitude
    background = LinearModel(prefix='b_' + sp)  # b_slope, b_intercept

    # Putting it all together
    mymodel = intensity * Convolve(reso, elastic + inelastic + inelastic2) + background
    parameters = mymodel.make_params()  # model parameters are a separate entity.

    # Ties and constraints
    parameters['e_' + sp + 'amplitude'].set(min=0.0, max=1.0)
    parameters['l_' + sp + 'amplitude'].set(min=0.0001)
    parameters['l2_' + sp + 'amplitude'].set(min=0.0001)
    parameters['l_' + sp + 'sigma'].set(min=0.00001)
    parameters['l2_' + sp + 'sigma'].set(min=0.00001)
    # allowing the HWHM to get closer to zero than this makes the EISF and QISF too correlated

    parameters['l_' + sp + 'center'].set(expr='e_' + sp + 'center')  # centers tied
    # parameters['l_'+sp+'amplitude'].set(expr='1 - e_'+sp+'amplitude')
    parameters['l2_' + sp + 'center'].set(expr='e_' + sp + 'center')  # centers tied
    parameters['l2_' + sp + 'amplitude'].set(expr='1 - e_' + sp + 'amplitude - l_' + sp + 'amplitude')

    # Some initial sensible values
    default_vals = {'I_' + sp + 'c': 1000, 'e_' + sp + 'amplitude': 0.9, 'l_' + sp + 'sigma': 0.04,
                     'l2_' + sp + 'sigma': 0.04,
                     'b_' + sp + 'slope': 0, 'b_' + sp + 'intercept': 0, 'e_' + sp + 'center': 0.0}
    # 'l_'+sp+'center': 0.0, 'r_'+sp+'center': 0.0}
    init_keys = default_vals.keys()
    # set custom parameters if given
    vals = default_vals
    if init_vals is not None:
        for param in init_vals.keys():
            vals[param] = init_vals[param]
    for p in init_keys:
        try:
            parameters[p].set(value=vals[p])
        except:
            continue

    # OPTIONAL, if you don't want to model the background
    # parameters['b_'+sp+'slope'].set(vary=False)
    # parameters['b_'+sp+'intercept'].set(vary=False)

    parameters['r_' + sp + 'amplitude'].set(vary=False)

    # parameters['l_'+sp+'fwhm'].set(max=1.0)
    # parameters['I_'+sp+'c'].set(max=10000.0)

    return mymodel, parameters


def make_global_model(temps, res, Qvalues, init_params):
    """ Put all individual spectrum model into one big model and set of parameters.
    """
    n_spectra = len(Qvalues)
    l_model = list()
    g_params = lmfit.Parameters()

    # make global fwhm with Arrhenius behaviour. Ea is in electron volt.
    # fwhm = fwhm_0 * exp(-Ea/kT)
    g_params.add('fwhm_rot_0', value=init_params['fwhm_rot_0'], min=0.0)
    g_params.add('fwhm_rot_Ea', value=init_params['fwhm_rot_Ea'])
    g_params.add('fwhm_rot2_0', value=init_params['fwhm_rot2_0'], min=0.0)
    g_params.add('fwhm_rot2_Ea', value=init_params['fwhm_rot2_Ea'])

    for temp in temps:

        print('making global model for temp {}'.format(temp))

        # make global fwhm for rotational process and tie to Arrhenius behaviour
        # kB in meV, 1/kB = 11.6049669 K/meV
        g_params.add('fwhm_rot', min=0.0, expr='fwhm_rot_0 * exp(-11.6049669 * fwhm_rot_Ea/({}))'.format(temp))
        g_params.add('fwhm_rot2', min=0.0, expr='fwhm_rot2_0 * exp(-11.6049669 * fwhm_rot2_Ea/({}))'.format(temp))

        for i in range(0, n_spectra):
            # model and parameters for one of the spectra
            spectrum_index = '{}_{}K'.format(i, temp)  # add temperature to prefix
            m, ps = generate_model_and_params_rot2L(res, spectrum_index=spectrum_index, init_vals=init_params)
            ps['l_{}_sigma'.format(spectrum_index)].set(expr='0.50000*fwhm_rot')
            ps['l2_{}_sigma'.format(spectrum_index)].set(expr='0.50000*fwhm_rot2')
            ps['I_{}_c'.format(spectrum_index)].set(value=init_params['I'])
            # l2 is the rotational and translational lorentzian convolved,
            # which gives a lorentzian with fwhm = fwhm_rot + fwhm_trans
            l_model.append(m)
            for p in ps.values():
                g_params.add(p)

    return l_model, g_params


def get_fit(data, resolution, temps, init_params=None, init_fixes=None, minim_method='leastsq'):
    """ Fit the data to the translational - rotational model.
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

    default_params = {'fwhm_rot_0': 0.1, 'fwhm_rot_Ea': 12.0, 'fwhm_rot2_0': 0.2,
                      'fwhm_rot2_Ea': -6.0, 'I': 1.0,
                      }

    # set custom parameters if given
    if init_params is not None:
        for param in init_params.keys():
            default_params[param] = init_params[param]

    l_model, g_params = make_global_model(temps, resolution, data[0]['Q'], default_params)
    if init_fixes is not None:
        for init_fix_param in init_fixes:
            g_params[init_fix_param].set(vary=False)
    global_fit, minimizer = leastSquares.minim_globaltemp(g_params, data, temps, l_model, method=minim_method)
    return global_fit, l_model, minimizer
