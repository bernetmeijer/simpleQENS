import h5py
import logging
import numpy as np
import os
from os.path import join as pjn


def histogram_to_point_data(values):
    r"""
    I have taken this from the qef package - give them a reference!
    Transform histogram(s) to point data
    Parameters
    ----------
    values : :class:`~numpy:numpy.ndarray`
        Array with histogram data
    Returns
    -------
    :class:`~numpy:numpy.ndarray`
        Array with point data
    """
    if values.ndim == 1:
        return (values[1:] + values[:-1]) / 2.0
    else:
        return (values[::, 1:] + values[::, :-1]) / 2.0
    
    
def load_nexus_processed(file_name):
    r"""
    I have taken this from the qef package - give them a reference!
    (I have modified it a bit though)
    Load data from a Mantid Nexus processed file
    Parameters
    ----------
    file_name : str
        Path to file
    Returns
    -------
    dict
        keys are q(momentum transfer), x(energy or time), y(intensities), and
        errors(e)
    """
    with h5py.File(file_name) as f:
        data = f['mantid_workspace_1']
        w = data['workspace']
        x = w['axis1'][:]  # energy or time values
        y = w['values'][:]  # intensities
        e = w['errors'][:]  # undeterminacies in the intensities
        # Transform to point data
        if len(x) == 1 + len(y[0]):
            x = histogram_to_point_data(x)
        # Obtain the momentum transfer values
        q = w['axis2'][:]
        if w['axis2'].attrs['units'] != 'MomentumTransfer':
            logging.warning('Units of vertical axis is not MomentumTransfer')
        # Transform Q values to point data
        if len(q) == 1 + len(y):
            q = histogram_to_point_data(q)
        return dict(q=q, x=x, y=y, e=e)


def load_nexus(file_name):
    r"""
    I have taken this from the qef package - give them a reference!
    Parameters
    ----------
    file_name : str
        Absolute path to file
    Returns
    -------
    """
    data = None
    # Validate extension
    _, extension = os.path.splitext(file_name)
    if extension != '.nxs':
        raise IOError('File extension is not .nxs')
    # Validate content
    with h5py.File(file_name) as f:
        if 'mantid_workspace_1' in f.keys():
            data = load_nexus_processed(file_name)
        else:
            raise IOError('No reader found for this HDF5 file')
    return data


def group_spectra(groupQ, data_dict):
    """

    Parameters
    ----------
    groupQ : int
        number of spectra in each group
    data_dict : dict
        dictionary of data
    Returns
    -------
    dictionary of data
    """
    n = 0
    temp_Q = []
    temp_I = []
    temp_dI = []

    new_Q = []
    new_I = []
    new_dI = []
    for qq, Q in enumerate(data_dict['Q']):
        temp_Q.append(Q)
        temp_I.append(data_dict['I'][qq])
        temp_dI.append((data_dict['dI'][qq])**2)
        n += 1
        if n == groupQ:
            new_Q.append(np.mean(np.array(temp_Q)))
            new_I.append(np.mean(np.array(temp_I), axis=0))
            new_dI.append(np.sqrt(np.mean(np.array(temp_dI), axis=0)))

            # reset
            n = 0
            temp_Q = []
            temp_I = []
            temp_dI = []

    if len(temp_Q) != 0:
        new_Q.append(np.mean(np.array(temp_Q)))
        new_I.append(np.mean(np.array(temp_I), axis=0))
        new_dI.append(np.sqrt(np.mean(np.array(temp_dI), axis=0)))

    data_dict['Q'] = np.array(new_Q)
    data_dict['I'] = np.array(new_I)
    data_dict['dI'] = np.array(new_dI)

    return data_dict


def group_energy(groupE, data_dict):
    """

    Parameters
    ----------
    groupE : int
        number of energy points in each group
    data_dict : dict
        dictionary of data

    Returns
    -------
    dictionary of data
    """
    n = 0
    temp_E = 0
    temp_I = np.zeros((len(data_dict['I'])))
    temp_dI = np.zeros((len(data_dict['I'])))

    new_E = []
    new_I = []
    new_dI = []
    for ii, E in enumerate(data_dict['E']):
        temp_E += E
        temp_I += data_dict['I'][:, ii]
        temp_dI += (data_dict['dI'][:, ii])**2
        n += 1
        if n == groupE:
            new_E.append(temp_E / groupE)
            new_I.append(temp_I / groupE)
            new_dI.append(np.sqrt(temp_dI / groupE))
            # reset
            n = 0
            temp_E = 0
            temp_I = np.zeros((len(data_dict['I'])))
            temp_dI = np.zeros((len(data_dict['I'])))
    if temp_E != 0:
        new_E.append(temp_E / n)
        new_I.append(temp_I / n)
        new_dI.append(np.sqrt(temp_dI / n))

    # transpose I and dI
    new_I = np.array(new_I).T
    new_dI = np.array(new_dI).T

    data_dict['E'] = np.array(new_E)
    data_dict['I'] = new_I
    data_dict['dI'] = new_dI

    return data_dict


def load_Nxs_Det(data_fn, data_dir, detector_fn, groupE=1, groupQ=1, delete_spectra=[], usecols=0, pg='002'):
    """Reads a 2D Mantid data set exported to ASCII, returning the result as a dict of numpy arrays.
    
    Parameters
    ------
    data_fn: string
        input data file name
    data_dir: string
        path to the folder where the data file name is
    detector_fn: string
        path to input detector
        We assume that the detector file name has columns
        Index, Spectrum No, Detector ID(s), R, Theta, Q, Phi, Monitor
    groupE: int
        optionally group E-values together in groups of size groupE
        (this might make the MCMC smapling easier and faster)
    groupQ: int
        optionally group Q-spectra together in groups of size groupQ
        (this can also be done in Mantid with a maps file, but this function
        makes it easier to play with Q-resolution and find the optimum value of 
        groupQ to get the best statistics.)
    delete_spectra : list(int)
        Indices of spectra to delete.

    Returns
    -------
    dictionary
        'E': energy values
        'Q': momentum values
        'I': intensities, one array for each Q-spectrum
        'dI': errors in intensities, one array for each Q-spectrum
    """
    
    data = load_nexus(pjn(data_dir, data_fn))

    detectors = np.loadtxt(detector_fn, usecols=usecols)
    n_spectra = detectors.shape
    
    # Find indexes of dat['x'] with values in (e_min, e_max). 'mask' is the array with the energy-limits adjusted
    if pg == '002':
        e_min = -0.5
        e_max = 0.5
    elif pg == '004':
        e_min = -1.0
        e_max = 1.0
    mask = np.intersect1d(np.where(data['x'] > e_min), np.where(data['x'] < e_max))

    # Drop data outside the fitting range
    fr = dict()  # fitting range. Use in place of 'dat'
    fr['x'] = data['x'][mask]
    fr['y'] = np.asarray([y[mask] for y in data['y']])
    fr['e'] = np.asarray([e[mask] for e in data['e']])

    data_dict = {
        'E': fr['x'],
        'Q': detectors,
        'I': fr['y'],
        'dI': fr['e']
    }

    # group E-values
    if groupE != 1:
        data_dict = group_energy(groupE, data_dict)

    # delete Q-spectra
    data_dict['Q'] = np.delete(data_dict['Q'], delete_spectra, axis=0)
    data_dict['I'] = np.delete(data_dict['I'], delete_spectra, axis=0)
    data_dict['dI'] = np.delete(data_dict['dI'], delete_spectra, axis=0)

    # group Q-spectra
    if groupQ != 1:
        data_dict = group_spectra(groupQ, data_dict)

    return data_dict
