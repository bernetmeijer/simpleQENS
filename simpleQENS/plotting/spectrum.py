# plot residual, given a fitted model and the data
import sys
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')

from simpleQENS.fitting import leastSquares
from simpleQENS.materials.quinPF6 import transModel
from simpleQENS.materials.quinPF6 import transAndRotModel, transAndRotModel_2L, staticModel
from lmfit.models import LorentzianModel, ConstantModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.operators.convolve import Convolve
import matplotlib.pyplot as plt
import numpy as np


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


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def my_residual(data, l_model, params, spec):
    E = data['E']  # fitting range of energies
    I = data['I'][spec]  # associated experimental intensities
    e = data['dI'][spec]  # associated experimental errors
    if e.any() < 0.0:
        print(e)
    model_evaluation = l_model.eval(x=E, params=params)
    return I - model_evaluation


def plot_fit(data, resolution, params, spec, l_model, plot_bg=False, plot_data=True):
    """

    Parameters
    ----------
    data: dict
        dictionary with the data (keys 'E', 'I', 'Q', 'dI')
    resolution: dict
        dictionary with the resolution data (keys 'E', 'I', 'Q', 'dI')
    params: lmfit Parameters object
        the parameters of the best fit. (Get it from result.params or through
        helpers.savingLmfit.load_minimizerResult() and the key 'params'
    spec: int
        spectrum index to plot
    l_model: lmfit Model object
        generated by one of the model generators (see plot_allspectra() in this script)
    plot_bg: bool
        True if you want to plot the linear background in the contributions plot
    delta: bool
        False if you do NOT want to plot a delta function (in the case of any translational model!)
    Returns
    -------
    fig1, ax1, fig2, ax2
        first figure is the simple plot of data, model, residual
        second figure is the contributions plot

    """

    # some settings
    lw = 5
    lw2 = 3
    plt.rc('font', size=30)
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 4.0
    plt.rcParams['ytick.major.width'] = 4.0
    plt.rcParams['xtick.minor.width'] = 2.0
    plt.rcParams['ytick.minor.width'] = 2.0
    plt.rcParams['xtick.major.size'] = 12.0
    plt.rcParams['ytick.major.size'] = 12.0
    plt.rcParams['xtick.minor.size'] = 6.0
    plt.rcParams['ytick.minor.size'] = 6.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [15, 10]

    spec_model = l_model[spec]

    # paramaters: add I_sp_c to parameters if it is not there
    try:
        I = params['I_{}_c'.format(spec)]
    except:
        params.add('I_{}_c'.format(spec), value=1.0)

    # simple plot: data, model and residual
    fig1, ax1 = plt.subplots()
    # plot data
    if plot_data:
        ax1.scatter(data['E'], data['I'][spec], color='darkorange', label='data', facecolor='None', s=400, lw=2,
                    zorder=3)
    # plot model
    best_model = spec_model.eval(x=data['E'], params=params)
    ax1.plot(data['E'], best_model, color='navy', label='total model', lw=lw, zorder=4)
    # plot residual
    residu_lijst = my_residual(data, spec_model, params, spec)
    ax1.plot(data['E'], residu_lijst, label='residual', linestyle='-', lw=lw, color='gold', zorder=7)

    # calculate chi-squared
    chi2 = np.sum(np.array(residu_lijst)**2)

    # ticks
    ax1.tick_params(which='major', pad=15,
                    bottom=True, top=True,
                    left=True, right=True)
    ax1.tick_params(which='minor',
                    bottom=True, top=True,
                    left=True, right=True)
    ax1.tick_params(which="major", direction='in')
    ax1.tick_params(which='minor', direction='in')
    ax1.minorticks_on()

    plt.legend(frameon=False)
    ax1.text(0, 0.8, 'Chi-2 = {}'.format(np.round(chi2, 3)), transform=ax1.transAxes)
    ax1.set_xlabel('E (meV)', fontsize=18)
    ax1.set_ylabel('I (arb. units)', fontsize=18)

    # advanced plot: plot all contributions
    fig2, ax2 = plt.subplots()
    # plot data
    if plot_data:
        ax2.scatter(data['E'], data['I'][spec], color='darkorange', label='data', facecolor='None', s=400, lw=2,
                    zorder=3)

    # plots total model
    ax2.plot(data['E'], best_model, color='navy', label='total model', lw=lw, zorder=4)

    # resolution
    intensity = params['I_{}_c'.format(spec)]
    try:
        resoy = intensity * params['e_{}_amplitude'.format(spec)] * resolution['I'][spec]
    except:
        resoy = intensity * resolution['I'][spec]
    # set maximum to the data maximum
    resoy = resoy * max(data['I'][spec])/max(resoy)
    #colorres = adjust_lightness('skyblue', amount=1.27)
    colorres = 'grey'
    # for nice darker legend colour:
    #ax2.plot(resolution['E'], resoy,
    #         color='mediumslateblue', linestyle='--', label='resolution', lw=lw2, zorder=9, alpha=0.7)
    # actual colour:
    #ax2.plot(resolution['E'], resoy,
    #         color=colorres, linestyle='--', lw=lw2, zorder=0)

    # plot the contrubutions
    lor_center = 0.0  # just fix it at zero

    try:  # 1st lorentzian
        inel1 = lorentzian(data['E'], ampl=params['l_' + str(spec) + '_amplitude'].value,
                           fwhm=params['l_' + str(spec) + '_fwhm'].value,
                           centre=lor_center,
                           I=params['I_' + str(spec) + '_c'], resolution=resolution)
        ax2.plot(data['E'], inel1, color='mediumslateblue', label='Lorentzian 1', linestyle='-.', lw=lw2, zorder=6)
        colorlor = adjust_lightness('mediumslateblue', amount=1.23)
        #colorlor = 'lightblue'
        ax2.fill_between(data['E'], np.zeros(inel1.shape), inel1, color=colorlor, zorder=2)
        #ax2.fill_between(resolution['E'], inel1, resoy, color='lightblue', alpha=0.5, zorder=2)

        try:  # 2nd lorentzian
            inel2 = lorentzian(data['E'], ampl=params['l2_' + str(spec) + '_amplitude'].value,
                               fwhm=params['l2_' + str(spec) + '_fwhm'].value,
                               centre=lor_center, I=params['I_' + np.str(spec) + '_c'],
                               resolution=resolution)
            ax2.plot(data['E'], inel2, color='darkorchid', label='Lorentzian 2', linestyle='-.', lw=lw2, zorder=7)
            ax2.fill_between(data['E'], inel1, inel2, color=adjust_lightness('darkorchid', 1.3), zorder=1)

            try:  # 3rd lorentzian
                inel3 = lorentzian(data['E'], ampl=params['l3_' + str(spec) + '_amplitude'].value,
                               fwhm=params['l3_' + str(spec) + '_fwhm'].value,
                               centre=lor_center, I=params['I_' + np.str(spec) + '_c'],
                               resolution=resolution)
                ax2.plot(data['E'], inel3, color='red', label='Lorentzian 3', linestyle='-.', lw=lw2, zorder=8)
            except:
                print('found 2 lorentzians')
        except:
            print('found 1 Lorentzian')
    except:
        print('found no lorentzians')

    if plot_bg:
        bg = background(data['E'], offset=params['b_' + np.str(spec) + '_intercept'].value,
                        slope=params['b_' + np.str(spec) + '_slope'].value)
        ax2.plot(data['E'], bg, color='red', label='background', lw=lw2, zorder=9)

    # ticks
    ax2.tick_params(which='major', pad=15,
                   bottom=True, top=True,
                   left=True, right=True)
    ax2.tick_params(which='minor',
                   bottom=True, top=True,
                   left=True, right=True)
    ax2.tick_params(which="major", direction='in')
    ax2.tick_params(which='minor', direction='in')
    ax2.minorticks_on()

    ax2.set_xlabel('E (meV)')
    ax2.set_ylabel('I (arb. units)')
    ax2.set_ylim(bottom=min(inel1), top=(max(data['I'][spec])*7))
    plt.legend(frameon=False, loc='upper right', fontsize=25)
    plt.title('Q {} = {} A-1'.format(spec, data['Q'][spec]))
    # plt.yscale('log')
    #ax2.text(0, 0.8, 'Chi-2 = {}'.format(np.round(chi2, 3)), transform=ax2.transAxes)

    return fig1, ax1, fig2, ax2


def plot_allspectra(data, resolution, result, modelname, plot_bg=False, logscale=False, plot_data=True, xlim=[-0.5, 0.5]):
    """
    Parameters
    ----------
    data: dict
        dictionary with the data (keys 'E', 'I', 'Q', 'dI')
    resolution: dict
        dictionary with the resolution data (keys 'E', 'I', 'Q', 'dI')
    result: dict
        dictionary containing results - must contain the parameters under the key 'params'
        (see helpers.savingLmfit for how to make this dictionary)
        NB: in the case of modelname=='1Lfreefwhm', this should be a list of result objects, one for each spectrum.
    modelname: string
        name of the model used in the fit. Choose between '1Lfreefwhm', '1Lfixfwhm', '1Ltransfwhm', 'transRot'.
    Returns
    -------
    list of elements [fig1, ax1, fig2, ax2], one for each spectrum.
        The first figure in this set is the simple plot (data, fit, residual)
        and the second figure is the contributions plot.
    """

    # collect the right l_model
    n_spectra = len(data['I'])
    if modelname == '1Lfreefwhm':
        l_model, params_list = leastSquares.make_separate_models(resolution, n_spectra, 1)
        # loop over spectra (results are now in different objects, so result is a list of result)
        all_figures = []
        for sp in range(n_spectra):
            fig1, ax1, fig2, ax2 = plot_fit(data, resolution, result[sp]['params'], sp, l_model, plot_bg=plot_bg, plot_data=plot_data)
            all_figures.append([fig1, ax1, fig2, ax2])
        return all_figures

    elif modelname == '1Lfixfwhm':
        l_model, g_params = leastSquares.make_global_model(resolution, n_spectra, 1)
        delta = True

    elif modelname == '2Lfixfwhm':
        l_model, g_params = leastSquares.make_global_model(resolution, n_spectra, 2)
        delta = True

    elif '1Ltransfwhm' in modelname:
        default_params = {'fwhm_trans_a': 0.2, 'fwhm_trans_l': 1.5, 'I': 1.0}
        l_model, g_params = transModel.make_global_model(resolution, data['Q'], default_params)
        delta = False

    elif modelname in ['transRot', 'transRot_fwhmFixed', 'transRot_l6.4', 'transRot_l6.07', 'transRot_l6.395',
                       'transRot_rotFixed_l6.395']:
        default_params = {'fwhm_rot': 0.1, 'fwhm_trans_a': 0.2, 'fwhm_trans_l': 1.5, 'I': 1.0}
        l_model, g_params = transAndRotModel.make_global_model(resolution, data['Q'], default_params)
        delta = False

    elif 'transRot_2L' in modelname:
        default_params = {'fwhm_rot': 0.1, 'fwhm_rot2': 0.01, 'fwhm_trans_a': 0.2, 'fwhm_trans_l': 1.5, 'I': 1.0}
        l_model, g_params = transAndRotModel_2L.make_global_model(resolution, data['Q'], default_params)
        delta = False

    elif modelname == 'static':
        default_params = {'I': 1.0}
        l_model, g_params = staticModel.make_global_model(resolution, data['Q'], default_params)
        delta = True

    else:
        print('{} is not a valid model name!'.format(modelname))
        return

    # loop over spectra
    all_figures = []
    for sp in range(len(data['I'])):
        fig1, ax1, fig2, ax2 = plot_fit(data, resolution, result['params'], sp, l_model, plot_bg=plot_bg, plot_data=plot_data)
        if logscale:
            #ax1.set_yscale('log')
            ax2.set_yscale('log')
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        all_figures.append([fig1, ax1, fig2, ax2])

    return all_figures
