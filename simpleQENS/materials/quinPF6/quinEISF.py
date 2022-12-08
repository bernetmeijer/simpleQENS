# three different EISFs for octahedral jump model of Quin

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


def j0(Q, r, r_idx):
    """ Returns spherical bessel function of Q*r, with r from the list
    of distances using r_idx.
    Parameters
    ----------
    Q: float
        momentum transfer
    r: list
        list of hopping distances, one of r1 or r2
    r_idx: int
        index of the rotation"""
    return spherical_jn(0, Q*r[r_idx])


def amplitude(Q, r):
    """ Calculate amplitude of the 5 (in)elastic contributions at a certain Q-value.
    Parameters
    ----------
    Q: float
        momentum transfer
    r: list
        list of hopping distances, one of r1 or r2
    """

    A = 0
    for nu in range(1, 5):
        A += j0(Q, r, nu-1)

    B = 0
    for nu in range(5, 8):
        B += j0(Q, r, nu-1)

    C = 0
    for nu in range(8, 14):
        C += j0(Q, r, nu - 1)

    D = 0
    for nu in range(14, 17):
        D += j0(Q, r, nu - 1)

    ampl1 = 1.0 / 24 * (1 + 2 * A + B + C + 2 * D)
    ampl2 = 1.0 / 24 * (1 + 2 * A + B - C - 2 * D)
    ampl3 = 1.0 / 24 * (4 - 4 * A + 4 * B)
    ampl4 = 1.0 / 24 * (9 - 3 * B + 3 * C - 6 * D)
    ampl5 = 1.0 / 24 * (9 - 3 * B - 3 * C + 6 * D)

    return np.array([ampl1, ampl2, ampl3, ampl4, ampl5])


def EISF_octa(Q, jump_distances, modelname='C4C2', f=1.0, m=1.0):
    """ Make the EISF for a given model in the octahedral point group
    for the Qvalues Q and list of 16 jump distances for each H-atom
    Allowed modelnames are ['C4C2', 'onlyC2', 'C3C2'].
    See jupyter notebook or Bee p. 225 for descriptions of these models.
    (models A, C, B respectively.)
    """

    all_atom_eisfs = []
    for atom_jumps in jump_distances:  # loop over H atoms
        amplitudes = amplitude(Q, atom_jumps)
        if modelname == 'C4C2':
            eisf = amplitudes[0]

        elif modelname == 'onlyC2':
            eisf = amplitudes[0] + amplitudes[1] + amplitudes[2]

        elif modelname == 'C3C2':
            eisf = amplitudes[0] + amplitudes[1]

        else:
            print('{} is not a valid model!'.format(modelname))
            return

        all_atom_eisfs.append(eisf)

    total_eisf = np.average(all_atom_eisfs, axis=0)

    return 1-f + f*m*total_eisf


