from __future__ import print_function, division
from numpy import sqrt, ones, array, shape, ptp, var, linspace
from . import processingcern as pc
from uncertainties import ufloat, umath
import scipy.optimize as opt
from matplotlib.pyplot import subplots, show

tidystring = lambda astr: '{:1.2f}'.format(astr).replace('+/-', '$\pm$')


def CalculatePTP(df):
    ptpchange = []
    for key, grp in df.groupby(['SampleB', 'configuration']):
        astr = [ufloat(a, b) for a, b in zip(grp.CTR, grp.CTRError)]
        ptpchange.append(tidystring(ptp(astr)))

    return ptpchange


def CalculateSTD(df):
    StandardDeviation = []
    for key, grp in df.groupby(['SampleB', 'configuration']):
        astr = [ufloat(a, b) for a, b in zip(grp.CTR, grp.CTRError)]
        StandardDeviation.append(tidystring(umath.sqrt(var(astr))))

    return StandardDeviation


def GenerateLaTeXTable(df, sortby=['configuration', 'SampleB', 'DOI'], 
                        sortdata=True, index=False, **kwargs):

    Cols = kwargs.get(
        'cols', ["configuration", "length", "SampleB", "energyresolution",
         "ggevents", "loc", "ctr", "chisquared"])
    
    ff = kwargs.get('ff', lambda x: '%10.2f' % x)
    pm = "$\pm$"
    RoundTo = 1
    if sortdata:
        df = df.sort(column=sortby)

    if any([aval == 'ctr' for aval in Cols]):
        df["ctr"] = [str(round(a, RoundTo)) + pm + str(round(b, RoundTo))
                     for a, b in zip(df.CTR, df.CTRError)]

    if any([aval == 'loc' for aval in Cols]):
        df['loc'] = [str(round(a, RoundTo)) + pm + str(round(b, RoundTo))
                     for a, b in zip(df.location, df.locationerr)]

    df['ggevents'] = [str(int(val)) + pm + str(int(sqrt(val)))
                      for val in df.numofsamples]
    df["energyresolution"] = [str(round(a, RoundTo + 1)) + pm + str(round(b, RoundTo + 1))
                              for a, b in zip(df.ERright, df.ERrighterr)]
    df["energyresolutionLeft"] = [str(round(a, RoundTo + 1)) + pm + str(round(b, RoundTo + 1))
                                  for a, b in zip(df.ERleft, df.ERlefterr)]

    #Cols = ["configuration","DOI","CTR"]
    return(df.to_latex(float_format=ff, index=index, columns=Cols))


def getchi(grp, dist='nofit', verbose=0, plot=False, fetchparam=False):

    chi = lambda FD: pc.chisquaretest(
        array(grp['CTR']),
        FD,
        yerr=grp['CTRError'],
        reducenum=len(grp['CTR']) + P - 1)  # N+P-1

    if dist == 'nofit':
        P = 1
        line = lambda xdata, c: ones(shape(xdata)) * c
        c, cerr = opt.curve_fit(line, array(grp['DOI']), array(grp['CTR']), sigma=grp['CTRError'])

        param = (ufloat(c, cerr))
        if verbose > 0:
            print("intercept", ufloat(c, cerr))

        fitdist = line(array(grp['DOI']), c)

        if plot:
            print(chi(fitdist))
            fig, ax = subplots()
            ax.grid()
            ax.errorbar(grp['DOI'], grp['CTR'], yerr=grp['CTRError'], fmt='.')

            X = linspace(0, 30)
            Y = line(X, c)
            ax.plot(X, Y, 'k-')

            ax.set_xlim(-2, 32)
            ax.set_ylim(160, 300)
            show()

    elif dist == 'linear':
        P = 2
        linear = lambda xdata, m, c: m * xdata + c
        (m, c), err = opt.curve_fit(
            linear, array(grp['DOI']), array(grp['CTR']), sigma=grp['CTRError'])
        merr, cerr = err.diagonal()

        param = (ufloat(m, merr), ufloat(c, cerr))
        if verbose > 0:
            print("gradient", ufloat(m, merr))
            print("intercept", ufloat(c, cerr))

        fitdist = linear(array(grp['DOI']), m, c)

        if plot:
            print(chi(fitdist))
            fig, ax = subplots()
            ax.grid()
            ax.errorbar(grp['DOI'], grp['CTR'], yerr=grp['CTRError'], fmt='.')

            X = linspace(0, 30)
            Y = linear(X, m, c)
            ax.plot(X, Y, 'k-')

            ax.set_xlim(-2, 32)
            ax.set_ylim(160, 300)
            show()
    else:
        raise KeyError("Unknown distribution!")

    if fetchparam:
        return param, chi(fitdist)
    else:
        return chi(fitdist)
