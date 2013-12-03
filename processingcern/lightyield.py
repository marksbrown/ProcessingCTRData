from __future__ import print_function, division
import os
from scipy import stats
import scikits.bootstrap as btp
import pandas as pds
from numpy import std, mean, floor


def normdist(xdata, loc, scale, amp, noise):
    '''
    Shifted Normal distribution
    '''
    return amp * stats.norm(loc, scale).pdf(xdata) + noise


def chisquaretest(ydata, y, yerr, reducenum):
    return 1 / reducenum * sum((ydata - y) ** 2 / yerr ** 2)


def GetDatLocations(rootloc, verbose=False):
    for dirname, dirnames, filenames in os.walk(rootloc):
            for filename in filenames:
                if filename.find("dat") < 0:
                    continue
                fileloc = dirname + "/" + filename
                if verbose > 0:
                    print("file to be outputted is", fileloc)

                yield fileloc


def ScikitsBootstrap(fdf):
    CILower, CIUpper = btp.ci(fdf.counts, std)
    scaleerr = (CIUpper - std(fdf.counts)) / 1.96

    CILower, CIUpper = btp.ci(fdf.counts, mean)
    locerr = (CIUpper - mean(fdf.counts)) / 1.96
    amperr = 0  # currently ignored
    return (locerr, scaleerr, amperr)


def LoadData(fileloc, verbose=False):

    defaultparam = {
        'crystal': 'None',
        'grease': 1.0,
        'dB': 0.0,
        'source': 1.0,
        'orientation': 'top',
        'outer': "unwrapped"}
    if verbose > 0:
        print(fileloc)
    SkipRows = []
    Keys = []
    Vals = []
    with open(fileloc) as f:
        for i, line in enumerate(f):
            if line[0] == '#':
                SkipRows.append(i)

                key, val = line[2:-2].split(' ')

                Keys.append(key)
                Vals.append(val)

    def TypeCast(val):
        try:
            return float(val)
        except ValueError:
            return str(val)

    param = {key: TypeCast(val) for key, val in zip(Keys, Vals)}
    defaultparam.update(param)
    if verbose > 0:
        print(param)
    df = pds.read_csv(fileloc, sep='\t', escapechar='\n', skiprows=SkipRows)
    df.columns = ["channelnum", "counts", "C"]
    del(df["C"])

    return defaultparam, df

from uncertainties import ufloat


def CalculatePhe(location, locationerr, dB=15, verbose=0):
    '''
    Calculates absolute number of photoelectrons
    '''
    qe = 1.602E-19  # electron charge
    # Egamma = 0.662 #per MeV
    qch = 160E-15  # C
    # with gain correction factor due to broken dynode
    G = 1.58E7 * (3390 / 4650)
    NPhe = lambda loc: ((qch * loc * 10 ** (dB / 10)) / (qe * G))
    Val = NPhe(ufloat(location, locationerr))
    return floor(Val.n), floor(Val.s)


def LightOutput(location, locationerr, dB=15):
    '''
    Absolute Number of photons
    '''
    a, b = CalculatePhe(location, locationerr, dB)
    QE = 0.22
    return floor(a * QE), floor(b * QE)
