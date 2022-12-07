# model where there is NO motion - just a resolution function!

import sys
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')
from simpleQENS.fitting import leastSquares
import lmfit
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
from qef.models.deltadirac import DeltaDiracModel
from qef.models.tabulatedmodel import TabulatedModel


def generate_model_and_params_static(res_data, spectrum_index=0, init_vals=None):
    """Produce an LMFIT model with only resolution and related set of fitting parameters.
    """

    sp = '' if spectrum_index is None else '{}_'.format(spectrum_index)  # prefix if spectrum_index passed

    # Resolution
    resolution1Dy = res_data['I'][0]  # Q, E
    resolution1Dx = res_data['E']  # Q, E

    # Model components
    reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_'+sp)  # you can vary r_centre and r_amplitude
    background = LinearModel(prefix='b_'+sp)  # b_slope, b_intercept

    # Putting it all together
    mymodel = reso + background
    parameters = mymodel.make_params()  # model parameters are a separate entity.

    # Some initial sensible values
    default_vals = {'r_'+sp+'amplitude': 0.9}
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
    #parameters['b_'+sp+'slope'].set(vary=False)
    #parameters['b_'+sp+'intercept'].set(vary=False)

    return mymodel, parameters


def make_global_model(res, Qvalues, init_params):
    """ Put all individual spectrum model into one big model and set of parameters.
    """
    n_spectra = len(Qvalues)
    l_model = list()
    g_params = lmfit.Parameters()

    for i in range(0, n_spectra):
        # model and parameters for one of the spectra
        m, ps = generate_model_and_params_static(res, spectrum_index=i, init_vals=init_params)
        ps['r_{}_amplitude'.format(i)].set(value=init_params['I'])
        # l2 is the rotational and translational lorentzian convolved,
        # which gives a lorentzian with fwhm = fwhm_rot + fwhm_trans
        l_model.append(m)
        for p in ps.values():
            g_params.add(p)

    return l_model, g_params


def get_fit(data, resolution, init_params=None, minim_method='leastsq'):
    """ Fit the data to the translational - rotational model.
    Parameters
    ----------
    data: dict
        dictionary containing the data, with keys 'I', 'E', 'Q', 'dI'.
    resolution: dict
        dictionary containing the resolution data, same keys as data.
    init_params: dict
        dictionary containing one or more of the initial values you might want to set
        Can define any of the other parameters in here, such as 'I_0_c' etc.
    minim_method: string
        minimisation method (see https://lmfit.github.io/lmfit-py/fitting.html for options)

    Returns
    -------
    MinimizerResult, globalModel, minimizer
    """

    default_params = {'I': 1.0,
                      }

    # set custom parameters if given
    if init_params is not None:
        for param in init_params.keys():
            default_params[param] = init_params[param]

    l_model, g_params = make_global_model(resolution, data['Q'], default_params)

    global_fit, minimizer = leastSquares.minim(g_params, data, l_model, method=minim_method)

    return global_fit, l_model, minimizer
