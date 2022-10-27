# save the params output of lmfit to a dictionary

import json


def toJson(fit, filename):
    """ Create large dictionary, with entries the parameters. Each parameter entry is itself a dictionary
    with the entries value and stderr.
    Parameters
    ----------
    params: lmfit.fit object
    filename: basestring
        where to save .json file"""

    params = fit.params

    mydict = {}
    for p in params:
        name = p
        value = params[p].value
        stderr = params[p].stderr
        mydict[p] = {'value': value,
                     'stderr:': stderr}

    # also save redchi
    mydict['redchi'] = {'value': fit.redchi}

    with open(filename, 'w') as f:
        json.dump(mydict, f)
    return
