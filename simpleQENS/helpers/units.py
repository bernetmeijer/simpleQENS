
def meV_to_ps(E):
    return 10**(-1) * 6.582119569/E


def fwhm_to_tau_h(fwhm):
    """ fwhm in meV, tau in ps
    tau = 2 h / fwhm"""
    return 2*4.135667696 / fwhm


def fwhm_to_tau(fwhm):
    """ fwhm in meV, tau in ps.
    tau = 2 hbar / fwhm"""
    return 2*0.6582 / fwhm
