# python script that generates plots from all the results that we have
# NB: make sure the loading of the data is exactly the same as you did when fitting the model
# so the size of q groups and delete_spectra should be exactly the same!

import os
import sys
sys.path.append('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Projects/SimpleQens/')

from simpleQENS.plotting import spectrum
from simpleQENS.helpers import savingLmfit
from simpleQENS.loaders import nexusLoader

import matplotlib.pyplot as plt
import pandas as pd

resultdir = '/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Materials/QuinPF6/qens/analysis/results_Apoc'
savedir = '/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Materials/QuinPF6/qens/analysis/fitplots_Apoc'

#temps = [320, 350, 400]
temps = [240, 260, 280]
temps = [280]
models = ['1Lfreefwhm', '1Lfixfwhm', '2Lfixfwhm', '1Ltransfwhm', 'transRot', 'transRot_2L', 'transRot_rotFixed', 'static']
models = ['1Lfreefwhm', '1Lfixfwhm', '2Lfixfwhm', '1Ltransfwhm', 'transRot', 'static']
#models = ['1Lfreefwhm', '1Lfixfwhm', '1Ltransfwhm', 'transRot', 'transRot_rotFixed']
models = ['transRot_l6.07']
#models=[]

noAlu = True

""" Saving info """
save = True
save_sp = [3]


""" Functions """

def load_data(temp):
    if temp > 290:
        datadir = '/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Materials/QuinPF6/qens/data/HT_reduced'
    else:
        datadir = '/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Materials/QuinPF6/qens/data/LT_reduced'

    with open('/Users/Bernet/OneDrive - Queen Mary, University of London/PhD/Materials/QuinPF6/qens/data/log.txt', 'r') as handle:
        df = pd.read_csv(handle, sep='\t', header=0)
    datasetnum = df.loc[(df['T']==temp) & (df['notes']=='sample')]['run'].iloc[0]
    si = df.loc[(df['T']==temp) & (df['notes']=='sample')]['si'].iloc[0]
    cellnum = df.loc[(df['notes']=='cell') & (df['si']==si)]['run'].iloc[0]
    resonum = df.loc[(df['notes']=='resolution') & (df['si']==si)]['run'].iloc[0]

    datafile = 'BASIS_{}_divided_sqw.nxs'.format(datasetnum)
    cellfile = 'BASIS_{}_divided_sqw.nxs'.format(cellnum)
    resofile = 'BASIS_{}_divided_sqw.nxs'.format(resonum)
    pg = 'si_{}'.format(si)

    detectorf = os.path.join(datadir, 'qvalues_si{}.txt'.format(si))
    cellpath = os.path.join(datadir, cellfile)

    if temp < 300:
        groupQ = 2
        delete_sp = [5]
    else:
        groupQ = 3  # keep this one!
        if noAlu:
            delete_sp = [0, 9, 10, 21, 22, 23, 24, 25, 26, 27, 33]  # if you want to delete aluminium Bragg
        else:
            delete_sp = [0, 9, 10, 23, 33] # 24

    data_raw = nexusLoader.load_Nxs_Det(datafile, datadir, detectorf, groupQ=groupQ,
                                        delete_spectra=delete_sp, pg=pg)
    data_subtracted = nexusLoader.load_Nxs_Det(datafile, datadir, detectorf, groupQ=groupQ,
                                               delete_spectra=delete_sp, pg=pg,
                                              empty_cell=cellpath, subtract_factor=1.0)
    reso = nexusLoader.load_Nxs_Det(resofile, datadir, detectorf, groupQ=groupQ, delete_spectra=delete_sp, pg=pg)

    return data_subtracted, reso


for temp in temps:
    data, reso = load_data(temp)

    if temp > 290 and noAlu:
        alu_id = 'noAlu_'
    else:
        alu_id = ''
    for modelname in models:

        # load the result
        if modelname == '1Lfreefwhm':
            # then we need to collect multiple spectra
            result = []
            for sp in range(len(data['Q'])):
                filepath = os.path.join(resultdir, '{}K_{}result_{}_sp{}.json'.format(temp, alu_id, modelname, sp))
                resultsp = savingLmfit.load_minimizerResult(filepath)
                result.append(resultsp)
        else:
            filepath = os.path.join(resultdir, '{}K_{}result_{}.json'.format(temp, alu_id, modelname))
            result = savingLmfit.load_minimizerResult(filepath)

        if temp < 300:
            logscale=True
            xlim = [-0.12, 0.12]
        else:
            logscale=True
            xlim = [-0.66, 0.66]
        figures_list = spectrum.plot_allspectra(data, reso, result, modelname, plot_bg=True, logscale=logscale, plot_data=True, xlim=xlim)
        plt.show()
        if save:
            for sp in range(len(data['Q'])):
                if sp in save_sp:
                    figures_list[sp][1].set_xlim(xlim)
                    figures_list[sp][0].savefig(os.path.join(savedir, '{}K_{}{}_residual_sp{}.png'.format(temp, alu_id, modelname, sp)))
                    plt.close(figures_list[sp][0])
                    figures_list[sp][3].set_xlim(xlim)
                    figures_list[sp][2].savefig(os.path.join(savedir, '{}K_{}{}_contributions_sp{}.png'.format(temp, alu_id, modelname, sp)))
                    plt.close(figures_list[sp][2])
