# save the params output of lmfit to a dictionary

import json
from lmfit import Parameters


def params_to_json(fit, filename):
    """ Create large dictionary, with entries the parameters. Each parameter entry is itself a dictionary
    with the entries value and stderr.
    Parameters
    ----------
    fit: lmfit.fit object
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


def save_minimizerResult(result, filepath):
    resultdict = {}

    # save best-fit params
    params = result.params.dumps()

    # save statistics
    statresults = [result.covar.tolist(), result.bic, result.redchi, result.chisqr, result.aic, result.var_names]
    for ii, stat in enumerate(['covar', 'bic', 'redchi', 'chisqr', 'aic', 'var_names']):
        resultdict[stat] = statresults[ii]

    resultdict['params'] = params
    with open(filepath, 'w') as handle:
        json.dump(resultdict, handle)


def load_minimizerResult(filepath):
    with open(filepath, 'r') as handle:
        resultdict = json.load(handle)
    params = Parameters().loads(resultdict['params'])
    resultdict['params'] = params
    return resultdict
