# functions to do an initial least-squares fit of the data

import numpy as np
from scipy.special import spherical_jn
import lmfit
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
from qef.models.deltadirac import DeltaDiracModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.operators.convolve import Convolve


def generate_model_and_params(res_data, spectrum_index=0, init_vals=None):
    """Produce an LMFIT model with one Lorentzian and related set of fitting parameters"""

    sp = '' if spectrum_index is None else '{}_'.format(spectrum_index)  # prefix if spectrum_index passed

    # Resolution

    # see if resolution data has different spectra, otherwise take first spectrum
    try:
        resolution1Dy = res_data['I'][spectrum_index]
    except:
        resolution1Dy = res_data['I'][0]

    resolution1Dx = res_data['E']  # Q, E

    # Model components
    intensity = ConstantModel(prefix='I_'+sp)  # I_amplitude
    elastic = DeltaDiracModel(prefix='e_'+sp)  # e_amplitude, e_center
    inelastic = LorentzianModel(prefix='l_'+sp)  # l_amplitude, l_center, l_sigma (also l_fwhm, l_height)
    reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_'+sp)  # you can vary r_center and r_amplitude
    background = LinearModel(prefix='b_'+sp)  # b_slope, b_intercept

    # Putting it all together
    model = intensity * Convolve(reso, elastic + inelastic) + background
    parameters = model.make_params()  # model parameters are a separate entity.

    # Ties and constraints
    parameters['e_'+sp+'amplitude'].set(min=0.05, max=1.0)
    parameters['l_'+sp+'sigma'].set(min=0.00001)
    parameters['r_'+sp+'amplitude'].set(min=0.0001)
    # allowing the HWHM to get closer to zero than this makes the EISF and QISF too correlated

    parameters['l_'+sp+'center'].set(expr='e_'+sp+'center')  # centers tied
    parameters['l_'+sp+'amplitude'].set(expr='1 - e_'+sp+'amplitude')

    # Some initial sensible values
    if init_vals is None:
        init_vals = {'I_'+sp+'c': 1.0, 'e_'+sp+'amplitude': 0.9, 'l_'+sp+'sigma': 0.0245,
                     'b_'+sp+'slope': 0, 'b_'+sp+'intercept': 0, 'r_'+sp+'center': 0}
    for p, v in init_vals.items():
        parameters[p].set(value=v)

    parameters['r_'+sp+'amplitude'].set(vary=False)
    #  parameters['b_'+sp+'slope'].set(vary=False)
    #  parameters['b_'+sp+'intercept'].set(vary=False)

    return model, parameters


def generate_model_and_params2(res_data, spectrum_index=0, init_vals=None):
    """Produce an LMFIT model with two Lorentzians and related set of fitting parameters"""

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
    parameters['l_'+sp+'amplitude'].set(min=0.00001, max=1.0)
    parameters['l2_'+sp+'amplitude'].set(min=0.00001, max=1.0)
    parameters['l_'+sp+'sigma'].set(min=0.00001)
    parameters['l2_'+sp+'sigma'].set(min=0.00001)
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


# put all individual models of the spectra into one global parameters.
def make_global_model(res, n_spectra, n_L):
    """ Put all individual spectrum model into one big model and set of parameters.
    """
    l_model = list()
    g_params = lmfit.Parameters()
    for i in range(0, n_spectra):
        # model and parameters for one of the spectra
        if n_L == 2:
            m, ps = generate_model_and_params2(res, spectrum_index=i)
        else:
            m, ps = generate_model_and_params(res, spectrum_index=i)
        l_model.append(m)
        for p in ps.values():
            g_params.add(p)
    return l_model, g_params


def make_separate_models(res, n_spectra, n_L, fwhm=None):
    """ Put all spectra models in a list, and do not make a global parameters object.
    """
    l_model = list()
    params_list = list()
    for i in range(0, n_spectra):
        # model and parameters for one of the spectra
        if n_L == 2:
            m, ps = generate_model_and_params2(res, spectrum_index=i)
        else:
            m, ps = generate_model_and_params(res, spectrum_index=i)
            if fwhm is not None:
                ps['l_{}_sigma'.format(i)].set(value=fwhm*0.5, vary=False)
        l_model.append(m)
        params_list.append(ps)
    return l_model, params_list


def make_global_params(n_spectra, g_params, n_L, fixfwhm, fwhm=None, fwhm2=None, constraintFrac=None):
    """ Create global parameters that tie spectrum parameters together.
    Currently fixes the value of the Lorentzian width over Q: this is true for rotational models, but should be changed for diffusion models.

    Params
    ------
    n_spectra : int
        number of Q-spectra
    g_params : lmfit.Parameters
        global parameters created in make_global_model
    fwhm : float
        initial value of global fwhm

    Returns
    -------
    lmfit.Parameters
        completed global parameters
    """

    if fixfwhm:

        # Introduce global parameter fwhm.
        g_params.add('fwhm', value=fwhm, min=0.0)
        if n_L == 2:
            g_params.add('fwhm2', value=fwhm2, min=0.0)
            if constraintFrac is not None:
                g_params['fwhm2'].set(value=fwhm*2)
                g_params['fwhm2'].set(expr='{}*fwhm'.format(constraintFrac))

        # Tie each lorentzian l_i_sigma to the global fwhm (sigma = fwhm/2)
        for i in range(n_spectra):
            g_params['l_{}_sigma'.format(i)].set(expr='0.50000*fwhm')
            if n_L == 2:
                g_params['l2_{}_sigma'.format(i)].set(expr='0.50000*fwhm2')

    # Fix the background over the different spectra:
    # Introduce global parameters b_slope and b_intercept
    #g_params.add('b_slope', value=0)
    #g_params.add('b_intercept', value=0)

    #Tie the background of each spectrum to these values
    #for i in range(n_spectra):
    #    g_params['b_{}_slope'.format(i)].set(expr='b_slope')
    #    g_params['b_{}_intercept'.format(i)].set(expr='b_intercept')

    return g_params


def make_global_params_tetra(n_spectra, g_params, fixfwhm, fixEISF, data, alpha=None, tau=None):
    """ Model from Andersson paper: 2 Lorentzians with their amplitudes tied.
    We can fix fwhm over Q. Make sure you print out Chi-squared values
    because otherwise how do we know this constant-f model works?"""

    if fixfwhm:
        g_params.add('alpha', value=1.5, min=0.0)
        g_params.add('tau', value=100.0, min=0.0)

    if alpha is not None:
        g_params['alpha'].set(value=alpha, vary=False)
        g_params['tau'].set(value=tau, vary=False)

    for i in range(n_spectra):
        # amplitude
        if fixEISF:
            eisf = 0.5*(1+3*spherical_jn(0, data['Q'][i] * 1.74))
            g_params['e_{}_amplitude'.format(i)].set(value=eisf)
        g_params['l2_{}_amplitude'.format(i)].set(expr='2.0*l_{}_amplitude'.format(i))
        # fwhm
        g_params.add('alpha_{}'.format(i), value=1.0)
        g_params.add('tau_{}'.format(i), value=26.0)
        if fixfwhm:
            g_params['alpha_{}'.format(i)].set(expr='alpha')
            g_params['tau_{}'.format(i)].set(expr='tau')
        g_params['l_{}_sigma'.format(i)].set(expr='8.0/(alpha_{}*tau_{})'.format(i, i))
        g_params['l2_{}_sigma'.format(i)].set(expr='(2.0+6.0*alpha_{})/(alpha_{}*tau_{})'.format(i, i, i))

    return g_params


def residuals(params, data, l_models, real_residu=False):
    n_spectra = len(l_models)
    r"""Difference between model and experiment, weighted by the experimental error

    Parameters
    ----------
    params : lmfit.Parameters
        Parameters for the global model

    Returns
    -------
    numpy.ndarray
        1D array of residuals for the global model
    """
    l_residuals = list()
    for i in range(n_spectra):
        E = data['E']  # fitting range of energies
        I = data['I'][i]  # associated experimental intensities
        e = data['dI'][i]  # associated experimental errors
        if e.any() < 0.0:
            print(e)
        model_evaluation = l_models[i].eval(x=E, params=params)
        if real_residu:
            #l_residuals.append((np.sqrt((model_evaluation - I)**2)))
            l_residuals.append(I - model_evaluation)
        else:
            l_residuals.append(np.sqrt((model_evaluation - I)**2) / e)
    if real_residu:
        return l_residuals
    else:
        return np.concatenate(l_residuals)


def residual_sp(params, data, l_model, sp, real_residual=False):
    E = data['E']  # fitting range of energies
    I = data['I'][sp]  # associated experimental intensities
    e = data['dI'][sp]  # associated experimental errors
    if e.any() < 0.0:
        print(e)
    model_evaluation = l_model.eval(x=E, params=params)
    if real_residual:
        return np.sqrt((model_evaluation - I) ** 2)
    else:
        return np.sqrt((model_evaluation - I) ** 2) / e


def minim(parameters, data, l_model, method='leastsq'):
    """ Minimize the residuals function and return the optimal parameters + statistics
    """
    minimizer = lmfit.Minimizer(residuals, parameters, fcn_args=(data, l_model), nan_policy='omit')
    g_fit = minimizer.minimize(method=method)
    return g_fit, minimizer


def residuals_globaltemp(params, data, temps, l_models, real_residu=False):
    n_spectra = len(l_models)
    r"""Difference between model and experiment, weighted by the experimental error

    Parameters
    ----------
    params : lmfit.Parameters
        Parameters for the global model

    Returns
    -------
    numpy.ndarray
        1D array of residuals for the global model
    """
    all_residuals = []
    for tt, temp in enumerate(temps):  # loop over temperatures
        temp_models = l_models[tt]  # collect spectrum models for each temperature
        all_residuals.append(residuals(params, data[tt], temp_models, real_residu=real_residu))  # residual for temperature

    if real_residu:
        return all_residuals
    else:
        return np.concatenate(all_residuals)


def minim_globaltemp(parameters, data, temps, l_model, method='leastsq'):
    """ Minimize the residuals function and return the optimal parameters + statistics
    """
    minimizer = lmfit.Minimizer(residuals_globaltemp, parameters, fcn_args=(data, temps, l_model), nan_policy='omit')
    g_fit = minimizer.minimize(method=method)
    return g_fit, minimizer


def get_initial_fit(data, resolution, n_spectra, n_L, fixfwhm=True, fwhm1=None, fwhm2=None, init_fwhm_fix=False,
                    constraintFrac=None, bmax=None, minim_method='leastsq', init_params=None):
    """ Get fit for single or two-lorentzian model, with option to specify some initial conditions and to fix the fwhm
    of the first lorentzian globally or not.
    To do: clean up this function, especially the initial conditions as they are double now. and add argument
    descriptions. """

    default_params = {'I': 1.0, 'fwhm': 0.02, 'fwhm2': 0.04}
    # replace defaults by arguments
    if init_params is not None:
        for init_param in init_params.keys():
            default_params[init_param] = init_params[init_param]

    if fixfwhm:
        l_model, g_params = make_global_model(resolution, n_spectra, n_L)
        g_params = make_global_params(n_spectra, g_params, n_L, fixfwhm, fwhm1, fwhm2, constraintFrac=constraintFrac)
        if bmax is not None:
            for i in range(n_spectra):
                g_params['b_{}_intercept'.format(i)].set(max=bmax)

        # set initial parameter values
        g_params['fwhm'].set(value=default_params['fwhm'])
        if n_L == 2:
            g_params['fwhm2'].set(value=default_params['fwhm2'])
        for i in range(n_spectra):
            g_params['I_{}_c'.format(i)].set(value=default_params['I'])

        global_fit, minimizer = minim(g_params, data, l_model, method=minim_method)
        return global_fit, l_model, minimizer
    else:
        results = []
        minimizers = []
        if init_fwhm_fix:
            # option to fix the fwhm
            l_models, params_list = make_separate_models(resolution, n_spectra, n_L, fwhm=fwhm1)
        else:
            l_models, params_list = make_separate_models(resolution, n_spectra, n_L)
        for sp in range(n_spectra):
            if bmax is not None:
                params_list[sp]['b_{}_intercept'.format(sp)].set(max=bmax)

            # set initial parameter values
            params_list[sp]['l_{}_sigma'.format(sp)].set(value=0.5*default_params['fwhm'])
            params_list[sp]['I_{}_c'.format(sp)].set(value=default_params['I'])

            minimizer = lmfit.Minimizer(residual_sp, params_list[sp],
                                        fcn_args=(data, l_models[sp], sp), nan_policy='omit')
            sp_fit = minimizer.minimize(method=minim_method)
            results.append(sp_fit)
            minimizers.append(minimizer)
        return results, l_models, minimizers


def fit_tetramodel(data, resolution, n_spectra, fixEISF, fixfwhm=True, alpha=None, tau=None):
    l_model, g_params = make_global_model(resolution, n_spectra, 2)
    g_params = make_global_params_tetra(n_spectra, g_params, fixfwhm, fixEISF, data, alpha, tau)
    global_fit, minimizer = minim(g_params, data, l_model)
    return global_fit, l_model, minimizer
