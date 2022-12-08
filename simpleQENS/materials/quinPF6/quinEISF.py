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


def EISF_octa(Q, jump_distances, modelname='octa C4C2', f=1.0, m=1.0):
    """ Make the EISF for a given model in the octahedral point group
    for the Qvalues Q and list of 16 jump distances for each H-atom
    Allowed modelnames are ['octa C4C2', 'octa only C2', 'octa C3C2'].
    See jupyter notebook or Bee p. 225 for descriptions of these models.
    (models A, C, B respectively.)
    """

    all_atom_eisfs = []
    for atom_jumps in jump_distances:  # loop over H atoms
        amplitudes = amplitude(Q, atom_jumps)
        #print('amplitudes: {}'.format(amplitudes))
        if modelname == 'octa C4C2':
            eisf = amplitudes[0]

        elif modelname == 'octa only C2':
            eisf = amplitudes[0] + amplitudes[1] + amplitudes[2]

        elif modelname == 'octa C3C2':
            eisf = amplitudes[0] + amplitudes[1]

        else:
            print('{} is not a valid model!'.format(modelname))
            return

        all_atom_eisfs.append(eisf)

    #print('all_atom_eisfs: {}'.format(all_atom_eisfs))
    total_eisf = np.average(all_atom_eisfs, axis=0)

    return 1-f + f*m*total_eisf


def EISF(Q, modelname, f=1.0, m=1.0, jump_distances=None):
    """
    optional model names:
    ['isotropic', 'D3 including C2', 'D3 only C3', 'octa C4C2', 'octa only C2', 'octa C3C2']
    if you take a octa model, you need to specify the jump distances.
    """

    if modelname == 'isotropic':
        radius = 3.23
        return 1 - f + f * m * (spherical_jn(0, Q * radius)) ** 2

    elif 'D3' in modelname:

        d_C2a1 = 3.2  # 4 H
        d_C2a2 = 4.6  # 4 H
        d_C2b = 2.714  # 4 H
        d_C2c = 4.515  # 2 H
        d_C3 = 3.5  # 12 H

        # sum over the C2 rotations (3 equivalent rotations)
        sum_C2 = 3 * 1.0 / 14 * (4 * spherical_jn(0, Q * d_C2a1) + 4 * spherical_jn(0, Q * d_C2a2) +
                                 4 * spherical_jn(0, Q * d_C2b) + 2 * spherical_jn(0, Q * d_C2c))

        # sum over the C3 rotations (2 equivalent rotations)
        sum_C3 = 2 * 1.0 / 14 * (12 * spherical_jn(0, Q * d_C3) + 2)  # the +2 comes from the 2 stationary H atoms

        A_A1 = 1.0 / 6 * (1 + sum_C3 + sum_C2)
        A_A2 = 1.0 / 6 * (1 + sum_C3 - sum_C2)
        A_E = 1.0 / 6 * (2 - sum_C3)

        if modelname == 'D3 including C2':
            eisf_C2_present = A_A1
            return 1 - f + f * m * eisf_C2_present

        elif modelname == 'D3 only C3':
            eisf_only_C3 = A_A1 + A_A2
            return 1 - f + f * m * eisf_only_C3

    elif 'octa' in modelname:
        return EISF_octa(Q, jump_distances, modelname=modelname, f=f, m=m)




