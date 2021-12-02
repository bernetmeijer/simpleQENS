# plot residual, given a fitted model and the data

from fitting import leastSquares
from lmfit.models import LorentzianModel, ConstantModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.operators.convolve import Convolve
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')


def lorentzian(x, ampl, fwhm, centre, I, resolution):
    # make model
    intensity = ConstantModel(prefix='I_')
    inel = LorentzianModel(prefix='l_')
    resolution1Dy = resolution['I'][0]
    resolution1Dx = resolution['E']  # Q, E
    reso = TabulatedModel(resolution1Dx, resolution1Dy, prefix='r_')
    combi = intensity * Convolve(reso, inel)

    para = combi.make_params()
    para['l_amplitude'].set(value=ampl)
    para['l_sigma'].set(value=(0.5 * fwhm))
    para['l_center'].set(value=centre)
    para['I_c'].set(value=I)

    # convolve with resolution
    y = combi.eval(x=x, params=para)

    return y


def background(x, offset, slope):
    y = x * slope + offset
    return y


def plot_fit(data, resolution, fit, spec, l_model, fixfwhm=True):
    """

    Parameters
    ----------
    data : dict
        data dictionary
    resolution : dict
        resolution dictionary
    fit
    spec
    l_model

    Returns
    -------

    """
    spec_model = l_model[spec]
    # spec_fit.plot(data_kws=dict(color='black', marker='o', markersize=1, markerfacecolor='none'),
    # fit_kws=dict(color='red', linewidth=4))

    # simple plot: data, model and residual
    fig1, ax1 = plt.subplots()
    # plot data
    ax1.plot(data['E'], data['I'][spec], label='data')
    # plot model
    best_model = spec_model.eval(x=data['E'], params=fit.params)
    ax1.plot(data['E'], best_model, 'r', label='best fit')
    # plot residu
    if fixfwhm:
        residu_lijst = leastSquares.residuals(fit.params, data, l_model, real_residu=True)[spec]
    else:
        residu_lijst = leastSquares.residual_sp(fit.params, data, spec_model, spec, real_residual=True)
    ax1.plot(data['E'], residu_lijst, label='residual')

    plt.legend()
    plt.xlabel('E (meV)')
    plt.ylabel('I')

    # advanced plot: plot all contributions
    fig2, ax2 = plt.subplots()
    # plot data
    ax2.scatter(data['E'], data['I'][spec], color='b', label='data')
    # plots total model
    ax2.plot(data['E'], best_model, color='orange', label='total model')
    # plot the contrubutions
    inel1 = lorentzian(data['E'], ampl=fit.params['l_' + np.str(spec) + '_amplitude'].value,
                       fwhm=fit.params['l_' + np.str(spec) + '_fwhm'].value,
                       centre=fit.params['l_' + np.str(spec) + '_center'].value,
                       I=fit.params['I_' + np.str(spec) + '_c'], resolution=resolution)
    ax2.plot(data['E'], inel1, color='purple', label='inelastic 1')
    # see if there's a second Lorentzian contribution
    try:
        inel2 = lorentzian(data['E'], ampl=fit.params['l2_' + np.str(spec) + '_amplitude'].value,
                       fwhm=fit.params['l2_' + np.str(spec) + '_fwhm'].value,
                       centre=fit.params['l2_0_center'].value, I=fit.params['I_' + np.str(spec) + '_c'],
                       resolution=resolution)
        ax2.plot(data['E'], inel2, color='pink', label='inelastic 2')
    except:
        inel2 = 0
    bg = background(data['E'], offset=fit.params['b_' + np.str(spec) + '_intercept'].value,
                    slope=fit.params['b_' + np.str(spec) + '_slope'].value)
    ax2.plot(data['E'], bg, color='red', label='background')

    ax2.plot(resolution['E'], resolution['I'][0], color='green', label='resolution')
    plt.legend()
    plt.yscale('log')
    # plt.ylim(top=10)
    plt.text(-0.47, 7, 'Chi-squared = {}'.format(fit.redchi))

    return fig1, ax1, fig2, ax2
