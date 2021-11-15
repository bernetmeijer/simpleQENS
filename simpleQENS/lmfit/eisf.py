# fit the eisf

import lmfit
from scipy.special import spherical_jn
import matplotlib.pyplot as plt
import numpy as np


def EISF(x, frac, radius, multiple):
    """

    Parameters
    ----------
    x : np.array
        q-values
    frac : float in [0, 1]
        fraction of rotating molecules
    radius : float
        hopping distance in angstrom
    multiple : float in [0, 1]
        multiple scattering correction factor (1=no multiple scattering, 0=only multiple scattering)

    Returns
    -------
    np.ndarray
        eisf

    """
    return 1 - frac + (frac * multiple) * 1. / 2 * (1 + spherical_jn(0, x * radius))


def make_eisf():
    model_EISF = lmfit.Model(EISF)
    eisf_params = model_EISF.make_params()

    # set some initial parameters
    eisf_params['frac'].set(min=0, max=1.0)
    eisf_params['radius'].set(min=0.1)
    eisf_params['multiple'].set(min=0.0, max=1.0)
    return model_EISF, eisf_params


def EISF_tetra(x, frac, radius, multiple):
    return 1 - frac + (frac * multiple) * 1. / 4 * (1 + 3 * spherical_jn(0, x * radius))


def make_eisf_tetra():
    model_EISF = lmfit.Model(EISF_tetra)
    eisf_params = model_EISF.make_params()

    # set some initial parameters
    eisf_params['frac'].set(min=0, max=1)
    # I
    eisf_params['radius'].set(min=0.1)
    eisf_params['multiple'].set(min=0.0, max=1.0)
    return model_EISF, eisf_params


def EISF_2site(x, frac, radius, multiple):
    return 1 - frac + (frac * multiple) * 1. / 2 * (1 + spherical_jn(0, x * radius))  # same as C2/C3 EISF!


def make_eisf_2site():
    model_EISF = lmfit.Model(EISF_2site)
    eisf_params = model_EISF.make_params()

    # set some initial parameters
    eisf_params['frac'].set(min=0, max=1)
    # I
    eisf_params['radius'].set(min=0.1)
    eisf_params['multiple'].set(min=0.0, max=1.0)
    return model_EISF, eisf_params


def EISF_mirror(x, frac, radius, multiple):
    tot_model = 0
    radius_set = [0.15476240000000013, 0.09523840000000008, 0.4992227471314684, 0.4992227471314685,
                  0.5714303999999998, 0.5238112000000003, 0.819265802668322, 0.8192658026683219]
    for r in radius_set:
        tot_model += EISF_2site(x, frac, r*radius, multiple)
    return tot_model/len(radius_set)


def make_eisf_mirror():
    model_EISF = lmfit.Model(EISF_mirror)
    eisf_params = model_EISF.make_params()

    # set some initial parameters
    eisf_params['frac'].set(min=0, max=1)
    # I
    eisf_params['radius'].set(value=1.0, vary=False)
    eisf_params['multiple'].set(min=0.0, max=1.0)
    return model_EISF, eisf_params


def isotropic(x, frac, radius, multiple):
    return 1-frac + frac * multiple * (spherical_jn(0, x*radius))**2


def make_isotropic():
    model_EISF = lmfit.Model(isotropic)
    eisf_params = model_EISF.make_params()

    # set some initial parameters
    eisf_params['frac'].set(min=0, max=1)
    # I
    eisf_params['radius'].set(min=0.1)
    eisf_params['multiple'].set(min=0.0, max=1.0)
    return model_EISF, eisf_params


def EISF_n(x, frac, radius, multiple, n):
    theoretical = 0.25*(4-n*(1-spherical_jn(0, x*radius)))
    return 1-frac+frac*multiple*theoretical


def make_eisf_n():
    model_EISF = lmfit.Model(EISF_n)
    eisf_params = model_EISF.make_params()

    # set some initial parameters
    eisf_params['frac'].set(min=0, max=1)
    # I
    eisf_params['radius'].set(min=0.1)
    eisf_params['multiple'].set(min=0.0, max=1.0)
    eisf_params['n'].set(value=2.5)
    return model_EISF, eisf_params


# where the eisf data are
eisf_folder = './fit_results/individual_nomaps/fitparams/'

# where you want to save it
save_folder = './fit_results/individual_nomaps/eisf/'


def fit_eisf(qs, eisfs, errs, models, params):
    """

    Parameters
    ----------
    qs : np.ndarray
        q values
    eisfs : np.ndarray
        eisf values
    models : list(model)
        models are make_eisf, make_eisf_tetra, make_eisf_2site
    params : list(list)
        for each model, parameters: name, radius, radius_fix, m_fix, f_fix

    Returns
    -------

    """
    fig, ax = plt.subplots()
    results = []

    q_plot = np.arange(0.3, 4.0, 0.01)

    # plot data
    ax.scatter(qs, eisfs, label='data')

    # fit and plot each model
    for ii, model in enumerate(models):
        if model == make_eisf_n:
            name, radius_fix, f_fix, m_fix, radius, f, m, n, vary_n = params[ii]
        else:
            name, radius_fix, f_fix, m_fix, radius, f, m = params[ii]
        my_eisf_mod, my_eisf_params = model()
        my_eisf_params['radius'].set(value=radius, vary=radius_fix)
        my_eisf_params['multiple'].set(value=m, vary=m_fix)
        my_eisf_params['frac'].set(value=f, vary=f_fix)
        if model == make_eisf_n:
            my_eisf_params['n'].set(value=n, vary=vary_n)


        eisf_result = my_eisf_mod.fit(eisfs, x=qs, params=my_eisf_params,
                                      weights=np.nan_to_num(1./errs, posinf=1e4),
                                      method='leastsq')
        results.append(eisf_result)
        # plot result
        eisf_model_result = my_eisf_mod.eval(x=q_plot, params=eisf_result.params)
        plt.plot(q_plot, eisf_model_result, label=name)

    plt.legend()
    plt.ylabel('EISF')
    plt.xlabel('Q')
    plt.ylim([0.0, 1.0])

    return results, fig, ax


def just_fit(qs, eisfs, errs, models, params):
    results = []
    evals = []
    q_plot = np.arange(0.0, 4.0, 0.01)

    # fit and plot each model
    for ii, model in enumerate(models):
        if model == make_eisf_n:
            name, radius_fix, f_fix, m_fix, radius, f, m, n, vary_n = params[ii]
        else:
            name, radius_fix, f_fix, m_fix, radius, f, m = params[ii]
        my_eisf_mod, my_eisf_params = model()
        my_eisf_params['radius'].set(value=radius, vary=radius_fix)
        my_eisf_params['multiple'].set(value=m, vary=m_fix)
        my_eisf_params['frac'].set(value=f, vary=f_fix)
        if model == make_eisf_n:
            my_eisf_params['n'].set(value=n, vary=vary_n)

        eisf_result = my_eisf_mod.fit(eisfs, x=qs, params=my_eisf_params,
                                      weights=np.nan_to_num(1./errs, posinf=1e4),
                                      method='leastsq')
        results.append(eisf_result)
        # plot result
        eisf_model_result = my_eisf_mod.eval(x=q_plot, params=eisf_result.params)
        evals.append(eisf_model_result)

    return results, evals
