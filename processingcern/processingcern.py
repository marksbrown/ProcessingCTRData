from __future__ import print_function, division
import os
from pandas import read_csv, DataFrame
import pandas as pds
from numpy import histogram, sqrt, linspace, mean, random, array, floor, ptp, std
from scipy.optimize import curve_fit
from scipy import stats
from uncertainties import ufloat
import uncertainties.umath as uncmath
from matplotlib.pyplot import figure, show
import statsmodels.api as sm
import scikits.bootstrap as btp
from . import peakdetect as pkd

# ImageSaveLocation = os.getcwd()+'/images' ##Final Location
ImageSaveLocation = '/home/mbrown/Desktop/tmpimages'  # Temporary Location
Extensions = ['png', 'pdf', 'svg']
Seperator = ';'

FileNumDict = {1: "Maximum Left SiPM Signal",
               2: "Maximum Right SiPM Signal",
               3: "Delay between Left and right NINO",
               4: "Width of Left NINO Signal",
               5: "Width of Right NINO Signal",
               7: "Edges of Left NINO Signal",
               8: "Edges of Right NINO Signal"}


def MatchFile(fn, keywords):
    '''
    returns true if all keywords are present in filename
    '''
    Flag = True
    for kw in keywords:
        if fn.find(kw) < 0:
            Flag = False
    return Flag


def Fetchfile(rootloc, keyword="", cropnum=2, skipfirst=True, verbose=0):
    '''
    New fetch file which will group files in 7 and return everything within this directory!
    '''
    FullLocation = []
    Files = []
    RunNames = []

    # http://stackoverflow.com/questions/120656/directory-listing-in-python
    for dirname, dirnames, filenames in os.walk(rootloc):
        for filename in filenames:
            # skips over guff
            if filename.startswith('input') or filename.startswith('output'):
                continue
            if filename[-3:] == "txt":
                RunNames.append(filename[cropnum:-4])
                Files.append(filename)
                FullLocation.append(os.path.join(dirname, filename))

        # Advanced usage:
        # editing the 'dirnames' list will stop os.walk() from recursing into
        # there.
        if '.git' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.git')

    FullDetails = []
    for i, afile in enumerate(set(RunNames)):

        # ignores transitional files
        if skipfirst and afile.find("00000") >= 0:
            print("skipping", afile)
            continue
        if afile.find(keyword) < 0:
            continue

        if verbose > 0:
            print(i, ":", afile)

        fs = filter(lambda filename: filename.find(afile) > 0, Files)
        fls = filter(lambda filename: filename.find(afile) > 0, FullLocation)
        # grouped files
        fs = {int(fn[1]): fullloc for fn, fullloc in zip(fs, fls)}

        if verbose > 0:
            for j in range(1, 6) + [7, 8]:
                print(i, ":", afile, fs[j])

        FullDetails.append([afile, fs])

    return zip(*FullDetails)


def CombineFiles(rootloc, splitby='_0', verbose=0):
    '''
    Combines files generated by oscilloscope with matching rootnames
    Ignores the first file as this is transitional with temperature
    '''
    UniqueNames, Files = Fetchfile(rootloc, verbose=0)  # all files, grouped

    SameName = set([val.split(splitby)[0]
                   for val in UniqueNames])  # all unique files

    for rootname in SameName:
        if verbose > 0:
            print(rootname)
        fFiles = [afile for afile in Files if afile[3].find(rootname) >= 0]
        for j in range(1, 6) + [7, 8]:
            for i, filename in enumerate(fFiles):
                if filename[j].find('_00000') >= 0:  # skips transition file
                    continue

                df = pds.read_csv(
                    filename[j],
                    skiprows=4,
                    sep=";",
                    index_col=0)
                df.index = [str(val) + "-" + str(i) for val in df.index]
                if i == 0:
                    fdf = df
                else:
                    fdf = fdf.append(df)

                fileloc, fn = os.path.split(filename[3])

            newloc = rootloc + "-Combined/" + rootname[4:]
            if not os.path.exists(newloc):
                os.makedirs(newloc)
            fdf.to_csv(
                newloc + "/F" + str(j) + "Run_" + rootname[4:] + ".txt",
                sep=";",
                header="Time;Ampl")


def WhenWasTheFileCreated(rootloc, verbose=0):
    '''
    Returns a dataframe containing the creation and modification
    dates of the files within the directory
    '''
    filedata = Fetchfile(rootloc, verbose=verbose)

    TimeData = [{"uniquename": un,
                 "mtime": os.path.getmtime(
                     fn[3]),
                 "ctime":os.path.getctime(fn[3])} for un,
                fn in zip(*filedata)]
    df = DataFrame(TimeData)
    df.ctime *= 1e9
    df.ctime = df.ctime.apply(pds.Timestamp)
    df.mtime *= 1e9
    df.mtime = df.mtime.apply(pds.Timestamp)

    return df


def normdist(xdata, loc, scale, amp):
    '''
    Shifted Normal distribution
    '''
    return amp * stats.norm(loc, scale).pdf(xdata)


def normdistwithnoise(xdata, loc, scale, amp, noise):
    return amp * stats.norm(loc, scale).pdf(xdata) + noise


def normfit(xdata, ydata, yerr=None, ScaleGuess=0.05,
            PeakGuess=None, failedfitmax=5, verbose=0):
    '''
    Shifted Normal distribution fit using stats.curve_fit()
    (Least squared fit)

    Will attempt to fit multiple times before giving up
    '''

    FailedToFitCounter = 0

    if len(xdata) == 0:
        if verbose > 0:
            print("No data has been passed to function")
        return (None, None), None

    if PeakGuess is None:
        x0 = [xdata[ydata.argmax()], ScaleGuess, 1]  # initial parameter guess
    else:
        x0 = [PeakGuess, ScaleGuess, 1]  # initial parameter guess

    while True:
        if verbose > 0:
            print("beginning run", FailedToFitCounter)
        if FailedToFitCounter >= failedfitmax:
            if verbose > 0:
                print("All fits failed")
            return (None, None), None

        try:
            param, err = curve_fit(normdist, xdata, ydata, sigma=yerr, p0=x0)
            if not isinstance(err, float):  # success!
                break
        except RuntimeError:
            return (None, None), None

        #MeanIndex = ydata.searchsorted(mean(ydata) * random.uniform(-1, 1))
        mleloc, mlescale = stats.norm.fit(ydata)
        try:
            x0 = [ydata[xdata.searchsorted(mleloc)], ScaleGuess, max(ydata)]
        except IndexError:
            if verbose > 0:
                fig = figure()
                ax = fig.add_subplot(111)
                ax.step(xdata, ydata, 'k-')
                ax.grid()
                show()
                assert 1 == 2
                print("outside range!")
            x0 = [ydata[array(xdata).searchsorted(random.uniform(0.4, 0.7))],
                  ScaleGuess * random.uniform(0.5, 5), 1]

        FailedToFitCounter += 1

    p1, p2, p3 = param
    chival = chisquaretest(
        normdist(xdata,
                 p1,
                 p2,
                 p3),
        ydata,
        sqrt(ydata),
        len(xdata) + len(param) - 1)
    if verbose > 0:
        PrintValues(param, err.diagonal())
        print("Chisquare test value is ", chival)
    return (param, err), chival


def chisquaretest(ydata, y, yerr, reducenum):
    return 1 / reducenum * sum((ydata - y) ** 2 / yerr ** 2)
    # return 1/reducenum*sum([(o-e)**2/yerr**2 for o,e in zip(ydata,y) if o>0])

# def chisquaretest(ydata,y,reducenum):
#    return 1/reducenum*sum([(o-e)**2/o for o,e in zip(ydata,y) if o>0])


def PrintValues(param, err, axis=None, roundto=4, size=10, offset=0):
    '''
    Prints fitted parameters from normfit in a pretty way (skips Amplitude)
    values will be placed into top/left corner of chosen subplot
    '''
    ParamNames = {0: "Location", 1: "Scale", 2: "Amplitude"}
    PorM = u"\u00B1"  # plus or minus symbol

    for i, (p, e) in enumerate(zip(param[:-1], err[:-1])):
        CurrentString = ":  " + \
            str(round(p, roundto)) + PorM + str(round(e, roundto))
        if axis is not None:
            axis.text(
                0.6,
                0.9 - (i + 2 * offset) * 0.08,
                ParamNames[i],
                horizontalalignment='left',
                verticalalignment='center',
                transform=axis.transAxes,
                size=size)
            axis.text(
                0.7,
                0.9 - (i + 2 * offset) * 0.08,
                CurrentString,
                horizontalalignment='left',
                verticalalignment='center',
                transform=axis.transAxes,
                size=size)
        else:
            print(ParamNames[i], CurrentString)


def FindFirstPhePeak(File, axis=None, SkipRows=4, verbose=0):
    '''
    Loads F7 and returns the indices corresponding to the first peak only
    '''
    df = read_csv(File, skiprows=SkipRows, sep=Seperator, index_col=0)
    fdf = df[(df.Ampl > 1.5) & (df.Ampl < 2.5)]  # selects 2 only

    if len(fdf) == 0:
        return []
    if axis is not None:
        X, Y = zip(*stats.itemfreq(df.Ampl))
        X = array(X, dtype=float)
        Y = array(Y, dtype=float)
        Condition = (X > 1.5) & (X < 2.5)
        axis.bar(X, Y, alpha=0.5, color='g')
        axis.bar(X[Condition], Y[Condition], color='r')
        axis.grid()

    return list(fdf.index)


def FindPhotoPeakEvents(File, binrange=(0.1, 1), fitrange=(0.3, 0.9), Bins=200,
                        leftsigma=3, rightsigma=5, MinSamples=100, axis=None, SkipRows=4, verbose=0):
        '''
        Finds peak in data and returns indices corresponding to
        left/right sigma from centroid of Normal distribution
        if matplotlib axis is passed then it will plot the data
        '''

        df = read_csv(File, skiprows=SkipRows, index_col=0, sep=Seperator)

        if verbose > 0:
            print("Length of data is", len(df))

        if len(df) < MinSamples:
            if verbose > 0:
                print("Not enough data", len(df))

        freq, edges = histogram(df.Ampl, bins=Bins, range=binrange)

        edges = 0.5 * (edges[1:] + edges[:-1])
        edges = edges[freq > 0]  # drops empty bins
        freq = freq[freq > 0]

        # Passing reduced datarange so the fitting algorithm is less confused
        #BinWidth = ptp(binrange) / Bins

        edgemin, edgemax = fitrange
        # Peak is probably in this range
        Condition = (edges > edgemin) & (edges < edgemax)

        if verbose > 0:
            print("Number of data points to fit to is", len(edges[Condition]))

        (param, err), chival = normfit(edges[Condition], freq[
            Condition], yerr=sqrt(freq[Condition]), ScaleGuess=0.05, verbose=verbose)

        if chival is None:
            return None, None, None

        if verbose > 0:
            print("Chi squared value is", chival)
        xmin, xmax = binrange

        p1, p2, p3 = param
        MinValue = p1 - leftsigma * p2
        MaxValue = p1 + rightsigma * p2

        if axis is not None:
            MaxVal = max(freq) * 1.1
            axis.fill_between(fitrange, 0, MaxVal, color='0.95')
            axis.plot(edges, freq, 'k-', linestyle='steps')
            X = linspace(MinValue, MaxValue, 500)
            Y = normdist(X, p1, p2, p3)
            axis.plot(X, Y, color='r')
            axis.grid()
            axis.set_ylabel("Frequency")
            axis.set_xlabel("Energy")
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(0, MaxVal)
            PrintValues(param, err.diagonal(), axis)

        fdf = df[(df.Ampl > MinValue) & (df.Ampl < MaxValue)]
        return list(fdf.index), p1, chival


def LocatePhotoPeaks(
    afile, binrange=(0.1, 1), factor=8, MinValue=100, Step=0.05, leftsigma=2, rightsigma=2,
        SkipRows=4, axis=None, verbose=0):
    '''
    afile : filename (either 1 or 2)
    binrange : events outside this range are ignored
    factor : assuming we're binning to a power of 2
    MinValue : minimum number of amplitude values to bother fitting to
    Step : crop-range for fitting algorithm about peak positions
    leftsigma,rightsigma : range of data to accept from either side of photopeak
    SkipRows : cropped guff out of data files
    GenerateImages : Should we generate images...yes
    axis : plot on this axis
    '''

    Bins = floor(ptp(binrange) * 2 ** factor)
    GenerateImages = axis is not None

    xmin, xmax = binrange
    df = pds.read_csv(afile, skiprows=SkipRows, sep=";", index_col=0)
    df = df[(df.Ampl > xmin) & (df.Ampl < xmax)]

    if verbose > 0:
        print("Length of Data:", len(df))

    if len(df.Ampl) < MinValue:
        if verbose > 0:
            print("insufficient data!")
        return None, None

    freq, edges = histogram(df.Ampl, bins=Bins, range=binrange)

    edges = 0.5 * (edges[1:] + edges[:-1])
    edges = edges[freq > 0]  # drops empty bins
    freq = freq[freq > 0]

    if GenerateImages:
        axis.step(edges, freq, 'k-', where='mid')
        axis.grid()
        axis.set_xlabel("Energy")
        axis.set_ylabel("Frequency")

    MaxPeaks, MinPeaks = pkd.peakdetect(freq, edges, lookahead=10)
    maxpeak = 0  # resets for each series of peaks (biggest wins!)
    secondarypeakloc = xmin
    secondpeakfound = False
    for x, y in MaxPeaks:
        Condition = (edges > x - Step) & (edges < x + Step)
        ydata = freq[Condition]
        xdata = edges[Condition]

        try:
            peakindex = ydata.searchsorted(mean(ydata))
            guessloc = xdata[peakindex]  # peak location
        except IndexError:
            guessloc = xdata[len(xdata) / 2 - 1]  # middle of random data

        try:
            param, err = curve_fit(
                normdist, xdata, ydata, p0=[guessloc, 0.05, 1])
        except RuntimeError:
            if verbose > 0:
                print("fit failed")
            continue
        # This is over complicated bollocks
        #(16thSept)Nice

        #(param,err),chival = normfit(xdata,ydata,yerr=sqrt(ydata),PeakGuess=guessloc,ScaleGuess=0.05,verbose=verbose)

        if err is not float:
            fdf = df[(df.Ampl > x - Step) & (df.Ampl < x + Step)]
            p1, p2, p3 = param
            errors = ScikitsBootstrap(fdf, loc=p1, scale=p2, verbose=verbose)
            p1err, p2err, p3err = errors
            Y = [normdist(p1, p1, p2, p3)]  # max value
            if verbose > 0:
                print("Peak Height is", max(Y), "whereas maxpeak is", maxpeak)

            if max(Y) > maxpeak:  # for multiple peaks, we choose the highest
                maxpeak = max(Y)
                peakparam = param
                peakerrors = errors
            elif p1 > secondarypeakloc:
                secondarypeakloc = p1
                secondpeakparam = param
                secondpeakerrors = errors
                secondpeakfound = True

    try:
        p1, p2, p3 = peakparam
        p1err, p2err, p3err = peakerrors
        photopeakindices = list(
            df[(df.Ampl > p1 - leftsigma * p2) & (df.Ampl < p1 + rightsigma * p2)].index)
    except NameError:
        print("photopeak fit failed")
        return None, None

    if verbose > 0:
        print("photopeak location :", p1, "+/-", p1err)

    if GenerateImages:
        #Label = ""
        X = linspace(p1 - p2 * leftsigma, p1 + p2 * rightsigma, 1000)
        Y = normdist(X, p1, p2, p3)
        axis.plot(X, Y, '-')
        PrintValues(peakparam, peakerrors, axis, size=10)
        axis.set_xlim(xmin, xmax)
    if secondpeakfound:
        s1, s2, s3 = secondpeakparam
        s1err, s2err, s3err = secondpeakerrors
        secondpeakindices = list(
            df[(df.Ampl > s1 - leftsigma * s2) & (df.Ampl < s1 + rightsigma * s2)].index)

        if verbose > 0:
            print("Secondary photopeak location :", s1, "+/-", s1err)

        if GenerateImages:
            X = linspace(s1 - s2 * leftsigma, s1 + s2 * rightsigma, 1000)
            Y = normdist(X, s1, s2, s3)
            axis.plot(X, Y, '-')
            PrintValues(
                secondpeakparam,
                secondpeakerrors,
                axis,
                offset=1,
                size=10)
            axis.set_xlim(xmin, xmax)
    else:
        secondpeakparam = [0, 0, 0]
        secondpeakerrors = [0, 0, 0]
        secondpeakindices = []

    return (
        (photopeakindices, peakparam, peakerrors), (
            secondpeakindices, secondpeakparam, secondpeakparam)
    )


def DelayPeakFitting(filenames, uniquename, **kwargs):
    '''
    Fits Gaussian distribution to delay distribution after removing Left,Right Energy SiPM and multiple edges

    filenames : Dict of filenames generated by FetchFile
    uniquename : unique name describing this set of parameters

    Full kwargs arguments given in file
    workingon ('doi') : experiment keyword
    SkipRows (4) : When files aren't combined we have a series of parameters left
    GenerateImages (False) : Plots and saves figures to Global variable ImageSaveLocation
    ImageKey (""): additional text string to add when saving figures
    timerange (-1000,1000): time range for plotting over
    MinSamples (100) : Minimum number of datapoints to bother fitting to
    errortype ('scikits'): lsq,parametric bootstrap or empirical bootstrap - what kind of error should we calculate?
    dt (25) : bin width of delay histogram
    SelectIndices : Determines whether we should select actively from secondary photopeak
    leftpherange : Range to search for photopeak in left scintillator detector energy spectrum
    rightpherange : Range to search for photopeak in Right scintillator detector energy spectrum
    verbose : verbosity variable (lots of potential printing WARNING!)
    '''

    # Default parameters (updated as found)
    workingon = kwargs.get('workingon', 'doi')
    SkipRows = kwargs.get('SkipRows', 4)
    GenerateImages = kwargs.get('GenerateImages', False)
    ImageKey = kwargs.get('ImageKey', "")
    timerange = kwargs.get('timerange', (-1000, 1000))
    MinSamples = kwargs.get('MinSamples', 100)
    errortype = kwargs.get('errortype', 'scikits')
    dt = kwargs.get('dt', 25)
    SelectIndices = kwargs.get('SelectIndices', 0)
    LeftPheRange = kwargs.get('leftpherange', (0.4, 0.8))
    RightPheRange = kwargs.get('rightpherange', (0.2, 0.8))
    verbose = kwargs.get('verbose', 0)

    A, B = uniquename.split('vs')

    A = A.split('_')[-1]  # Left Sample
    B = B.split('_')[0]  # Right Sample

    SampleNames = A + " vs " + B
    if verbose > 0:
        print(SampleNames, ":", uniquename)

    ModificationTime = os.path.getmtime(filenames[3])
    CreationTime = os.path.getctime(filenames[3])

    if workingon == "2396":  # a bit hacky but it's a one off case comparison
        crystallength = 20
        B = workingon
    else:
        try:
            # Position B refers to the scintillator under interest always
            crystallength = int(B[:-1])
        except ValueError:
            crystallength = int(B.split(workingon)[0])
            if crystallength == 24044:  # known sample name
                crystallength = 20
            elif crystallength == 2396:
                crystallength == 20

    if GenerateImages:
        fig = figure(figsize=(12, 12))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.set_title("Left Maxima")
        ax2.set_title("Right Maxima")
        ax3.set_title("Left Edges")
        ax4.set_title("Right Edges")
    else:
        ax1 = None
        ax2 = None
        ax3 = None
        ax4 = None

    if verbose > 1:
        for j in range(1, 6) + [7, 8]:
            print(j, ":", filenames[j])

    LeftFirstPeak, LeftSecondpeak = LocatePhotoPeaks(
        filenames[1], binrange=LeftPheRange, SkipRows=SkipRows, axis=ax1, verbose=verbose)
    if LeftFirstPeak is None:
        if GenerateImages:
            fig.clear()  # scrubs plot
        if verbose > 0:
            print("Left photopeak won't fit")
        return 1

    IndicesOne, LeftPeakParam, LeftPeakError = LeftFirstPeak
    # ignoring any secondary peaks in reference photodetector

    if verbose > 0:
        p1, p2, p3 = LeftPeakParam
        p1err, p2err, p3err = LeftPeakError
        print("Left Peak Position", p1, "+/-", p1err)

    RightFirstPeak, RightSecondpeak = LocatePhotoPeaks(
        filenames[2], binrange=RightPheRange, SkipRows=SkipRows, axis=ax2, verbose=verbose)
    if RightFirstPeak is None:
        if GenerateImages:
            fig.clear()  # scrubs plot
        if verbose > 0:
            print("Right photopeak won't fit")
        return 1

    IndicesTwo, RightPeakParam, RightPeakError = RightFirstPeak
    if verbose > 0:
        p1, p2, p3 = LeftPeakParam
        p1err, p2err, p3err = LeftPeakError
        print("Right Peak Position", p1, "+/-", p1err)

    IndicesTwoSecond, RightSecondPeakParam, RightSecondPeakError = RightSecondpeak

    if verbose > 0:
        print("Calculating Edges")

    IndicesThree = FindFirstPhePeak(
        filenames[7],
        axis=ax3,
        SkipRows=SkipRows,
        verbose=verbose)
    IndicesFour = FindFirstPhePeak(
        filenames[8],
        axis=ax4,
        SkipRows=SkipRows,
        verbose=verbose)

    if GenerateImages:
        fig.tight_layout()
        for Ext in Extensions:
            fig.savefig(
                ImageSaveLocation +
                '/' +
                Ext +
                '/' +
                SampleNames +
                '_' +
                uniquename +
                '_' +
                'IndexData' +
                '.' +
                Ext)
        show()

    if SelectIndices == 0:
        Indices = list(set(IndicesOne) & set(
            IndicesTwo) & set(IndicesThree) & set(IndicesFour))
    elif SelectIndices == 1:
        Indices = list(set(IndicesOne) & set(
            IndicesTwoSecond) & set(IndicesThree) & set(IndicesFour))
    elif SelectIndices == 2:
        Indices = list(set(IndicesOne) & (set(IndicesTwo) | set(IndicesTwoSecond))
                       & set(IndicesThree) & set(IndicesFour))
    else:
        print("Only three choices available matey jim!")
        return 1

    if verbose > 0:
        print("Length of Indices is", len(Indices))

    df = read_csv(filenames[3], skiprows=SkipRows, index_col=0, sep=Seperator)
    df.Ampl *= 1e12  # time in ps
    fdf = df[df.index.isin(Indices)]  # selects matching data only

    if len(fdf) < MinSamples:
        if verbose > 0:
            print("Insufficient number of samples", len(fdf))
        return 1

    Frequency, Values = histogram(
        fdf.Ampl, range=timerange, bins=ptp(timerange) / dt)
    Values = 0.5 * (Values[1:] + Values[:-1])

    Values = Values[Frequency > 0]
    Frequency = Frequency[Frequency > 0]

    if verbose > 0:
        print("Number of samples is", len(fdf))

    xmin, xmax = timerange
    (param, err), chival = normfit(Values, Frequency, yerr=sqrt(Frequency),
                                   ScaleGuess=100, PeakGuess=100, failedfitmax=100, verbose=verbose)  # fit to CTR peak

    try:
        p1, p2, p3 = param
    except TypeError:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.step(Values, Frequency, 'k-')
        show()
        assert 1 == 2

    if errortype == 'lsq':  # error generated by curve_fit()
        locerr, scaleerr, amperr = err.diagonal()
    elif errortype == 'parametric':
        locerr, scaleerr = ParametricBootstrap(p1, p2, len(fdf.Ampl))
        amperr = 0
    elif errortype == 'empirical':
        (param, err), chival = EmpiricalBootstrap(fdf.Ampl, p2,
                                                  filenames[3], GenerateImages=GenerateImages, ImageKey=ImageKey, verbose=verbose)
        locerr, scaleerr, amperr = param
    elif errortype == 'scikits':
            #(fdf,loc=0,sigma=100,leftsigma=2,rightsigma=2,verbose=0)
        locerr, scaleerr, amperr = ScikitsBootstrap(
            fdf, loc=p1, scale=p2, verbose=verbose)
        #fRawData = fdf.Ampl[abs(fdf.Ampl) < 500]
        #CILower,CIUpper = btp.ci(fRawData,std)
        #scaleerr = (CIUpper-std(fRawData))/1.96

        #CILower,CIUpper = btp.ci(fRawData,mean)
        #locerr = (CIUpper-mean(fRawData))/1.96
        # amperr = p3 #currently ignored
    else:
        print("Unknown error type!")
        assert 1 == 2, "Error!"

    if GenerateImages:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.errorbar(Values, Frequency, yerr=sqrt(Frequency), fmt='r.')
        X = linspace(xmin, xmax, 1000)
        Y = normdist(X, p1, p2, p3)
        ax.set_title("SiPM Relative Delay")
        ax.plot(X, Y, color='0.7')
        ax.grid()
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Delay (ps)")
        ax.set_ylabel("Frequency")

        # prints errors from selected method
        PrintValues(param, [locerr, scaleerr, amperr], ax)
        fig.tight_layout()

        for Ext in Extensions:
            fig.savefig(
                ImageSaveLocation +
                '/' +
                Ext +
                '/' +
                SampleNames +
                '_' +
                uniquename +
                '_' +
                'CTR' +
                '.' +
                Ext)
        show()

    DataDict = {
        "uniquename": uniquename, "location": p1, "locationerr": locerr, "scale":
        p2, "scaleerr": scaleerr, "mtime": ModificationTime, "ctime": CreationTime,
        "amplitude": p3, "amplitudeerr": amperr, "chisquared": chival,
        "numofsamples": len(fdf.Ampl), "length": crystallength,
        "SampleA": A, "SampleB": B,
        "LPloc": LeftPeakParam[0], "LPscale": LeftPeakParam[1],
        "LPlocerr": LeftPeakError[0], "LPscaleerr": LeftPeakError[1],
        "RPloc": RightPeakParam[0], "RPscale": RightPeakParam[1],
        "RPlocerr": RightPeakError[0], "RPscaleerr": RightPeakError[1],
        "RSPloc": RightSecondPeakParam[
            0], "RSPscale": RightSecondPeakParam[1],
        "RSPlocerr": RightSecondPeakError[
            0], "RSPscaleerr": RightSecondPeakError[1],
    }
    return DataDict


def FindDelayData(filenames, uniquename, outputdir, verbose=0):
    '''
    Fits Gaussian distribution to delay distribution after removing Left,Right Energy SiPM and multiple edges

    filenames : Dict of filenames generated by FetchFile
    '''
    ##ParsedFilename = uniquename.split('_')
    Leftfitregion = (0.5, 0.7)
    Rightfitregion = (0.4, 0.6)
    if verbose > 0:
        print("--", uniquename, "--")

    IndicesOne, LeftPhotopeakLoc, FitValLeft = FindPhotoPeakEvents(
        filenames[1], fitrange=Leftfitregion, verbose=verbose)
    IndicesTwo, RightPhotopeakLoc, FitValRight = FindPhotoPeakEvents(
        filenames[2], fitrange=Rightfitregion, verbose=verbose)

    if (FitValLeft is None) or (FitValRight is None):
        print("Fit Failed")
        return None

    IndicesThree = FindFirstPhePeak(filenames[7], verbose=verbose)
    IndicesFour = FindFirstPhePeak(filenames[8], verbose=verbose)

    Indices = list(set(IndicesOne) & set(IndicesTwo)
                   & set(IndicesThree) & set(IndicesFour))

    df = read_csv(filenames[3], skiprows=SkipRows, sep=Seperator, index_col=0)
    df.Ampl *= 1e12  # time in ps
    fdf = df[df.index.isin(Indices)]  # selects matching data only

    rootloc, fn = os.path.split(filenames[3])
    ctrloc = outputdir + '/ctrdata'
    if not os.path.exists(ctrloc):
        os.makedirs(ctrloc)

    fdf.to_csv(ctrloc + '/' + uniquename + '.csv')  # saves CTR data
    return None


def FitToDelayData(
        DelayValues, timerange=1000, GenerateImages=True, verbose=0):
    '''
    Loads cropped data and fits Gaussian
    Calculates error using bootstrap
    '''

    freq, binedges = histogram(
        DelayValues, bins=2 * timerange / 25 + 1, range=(-timerange, timerange))
    binedges = 0.5 * (binedges[1:] + binedges[:-1])

    binedges = binedges[freq > 0]
    freq = freq[freq > 0]

    (param, err), chival = normfit(binedges, freq, yerr=sqrt(freq),
                                   ScaleGuess=100, verbose=verbose)  # fit to CTR peak
    p1, p2, p3 = param

    DelayValues = array(DelayValues)
    fRawData = DelayValues[abs(DelayValues) < 500]
    CILower, CIUpper = btp.ci(fRawData, std)
    scaleerr = (CIUpper - std(fRawData)) / 1.96

    CILower, CIUpper = btp.ci(fRawData, mean)
    locerr = (CIUpper - mean(fRawData)) / 1.96
    ##amperr = p3  # currently ignored

    p1err, p2err, p3err = err.diagonal()

    return param, (locerr, scaleerr, p3err)


def ScikitsBootstrap(fdf, loc=0, scale=100,
                     leftsigma=5, rightsigma=5, minsamples=100, verbose=1):
    '''
    parameters from fit of Gaussian are used to clip total range of data
    from this a BCA bootstrap of the error in the loc and scale are found
    by the MLE estimates (std and mean respectively) --> This will ONLY
    work if the data given IS Gaussian
    '''

#    fRawData = fdf.Ampl[abs(fdf.Ampl) < 1000]
    fRawData = fdf.Ampl[
        (fdf.Ampl > loc - leftsigma * scale) & (fdf.Ampl < loc + rightsigma * scale)]
    if verbose > 0:
        print("number of samples", len(fRawData))
    if len(fRawData) < minsamples:
        if verbose > 0:
            print("insufficient data")
        return (1e12, 1e12, 1e12)
    CILower, CIUpper = btp.ci(fRawData, std)
    scaleerr = (CIUpper - std(fRawData)) / 1.96

    CILower, CIUpper = btp.ci(fRawData, mean)
    locerr = (CIUpper - mean(fRawData)) / 1.96
    amperr = 0  # currently ignored
    return (locerr, scaleerr, amperr)


def ParametricBootstrap(loc, scale, N, Runs=500, verbose=0):
    dist = stats.norm(loc=loc, scale=scale)

    RandomSamples = dist.rvs(N * Runs)
    estloc, estscale = zip(*[stats.norm.fit(nsam)
                           for nsam in RandomSamples[::Runs]])
    return mean(estloc), mean(estscale)


def RandomSample(xdata, cdf, NSamples, verbose=0):
    '''
    calculates a random sample for a given cdf
    '''
    R = random.uniform(size=NSamples)
    limit = cdf[-1]  # max probability to look upto

    if verbose > 0:
        # maximum probability cdf can be used for
        print(xdata[cdf.searchsorted(limit)])

    try:
        Sample = [xdata[cdf.searchsorted(r, side='left')]
                  for r in R if r < limit]
    except IndexError:
        print(R)

    return array(Sample)


def EmpiricalBootstrap(rawdata, fitscale, filename, NRuns=500, timerange=(
        -500, 500), dt=25, GenerateImages=True, ImageKey="", FetchData=False, verbose=0):
    '''
    Empirical Bootstrap using ECDF
    '''

    ScaleRange = (fitscale - 20, fitscale + 20)
    smin, smax = ScaleRange
    X = linspace(smin, smax, 1000)

    ecdf = sm.distributions.ECDF(rawdata)  # step 1, find ECDF
    ParameterValues = []
    for _ in range(NRuns):
        # step 2, generate new sample
        RndSample = RandomSample(ecdf.x, ecdf.y, NSamples=len(rawdata))

        Values, Frequency = zip(*stats.itemfreq(RndSample))
        Frequency = array(Frequency, dtype=float)

        (param, err), chival = normfit(Values, Frequency, yerr=sqrt(Frequency),
                                       ScaleGuess=100, verbose=verbose)  # step 3, fit to CTR peak

        p1, p2, p3 = param

        ParameterValues.append(param)

    if FetchData:
        return zip(*ParameterValues)

    LocValues, ScaleValues, AmpValues = zip(*ParameterValues)
    Freq, BinEdges = histogram(ScaleValues, bins=41, range=ScaleRange)
    BinEdges = 0.5 * (BinEdges[1:] + BinEdges[:-1])
    # step 3, fit to CTR peak
    (param, err), chival = normfit(BinEdges,
                                   Freq, ScaleGuess=100, verbose=verbose)
    p1, p2, p3 = param

    if GenerateImages:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.set_title("Empirical Bootstrap")
        ax.step(BinEdges, Freq)

        Y = normdist(X, p1, p2, p3)
        ax.plot(X, Y, color='0.7')
        ax.set_xlim(ScaleRange)
        ax.grid()

        PrintValues(param, err.diagonal(), ax)

        fig.tight_layout()

        ParsedFilename = filename[:-4].split('_')
        FileType, SampleNames, KeyWords, Key = ParsedFilename

        for Ext in Extensions:
            fig.savefig(
                ImageSaveLocation +
                '/' +
                Ext +
                '/' +
                SampleNames +
                '_' +
                Key +
                '_' +
                ImageKey +
                '_' +
                'empiricalbootstrap' +
                '.' +
                Ext)
        show()

    if verbose > 0:
        print("Empirical Bootstrap gives", p2)

    return (param, err), chival

# ErrorType='scikits',Combined=False,workingon='reference',,,SelectIndices=0):


def ProcessFiles(fileloc, **kwargs):
    '''
    Processes CTR data files from standard and DOI CTR measurements

    skipfirst (True) : ignores 00000 events - typically a short run to
    wait for temperature to stabilise (but not always!)
    '''

    workingon = kwargs.get("workingon", "DOI")
    ErrorType = kwargs.get("ErrorType", "scikits")
    splitby = kwargs.get('splitby', '_0')
    skipfirst = kwargs.get('skipfirst', True)
    Combined = kwargs.get('Combined', False)
    verbose = kwargs.get('verbose', 0)

    if Combined: #literally combine all files (bar skipfirst)
        CombineFiles(fileloc, splitby=splitby, verbose=verbose)
        fileloc += '-Combined'
        kwargs = dict(kwargs.items() + [('SkipRows', 0)])
    else:
        kwargs = dict(kwargs.items() + [('SkipRows', 4)])

    GeneratedData = []
    # BORING :P
    UniqueNames, Files = Fetchfile(fileloc, skipfirst=skipfirst, verbose=0)
    GeneratedData += [DelayPeakFitting(fs, un, **kwargs)
                      for fs, un in zip(Files, UniqueNames)]
    show()

    # gets rid of failed events
    GeneratedData = [gi for gi in GeneratedData if isinstance(gi, dict)]

    try:
        df = pds.DataFrame(GeneratedData)
        df.to_csv(fileloc + '/' + workingon + '-' + ErrorType + '.csv')
        print("Complete!")
    except AttributeError:
        print("failed to create dataframe")
        print(GeneratedData)


def FetchDataFrame(rootloc, workingon,
                   Combined=False, ErrorType='scikits', verbose=0):
    '''
    Retrieves dataframe matching conditions
    '''
    if Combined:
        filename = rootloc + '-Combined' + \
            '/' + workingon + '-' + ErrorType + '.csv'
    else:
        filename = rootloc + '/' + workingon + '-' + ErrorType + '.csv'

    if verbose > 0:
        print("Attempting to retrieve :", filename)
    try:
        return pds.read_csv(filename)
    except IOError:
        print("File", filename, "does not exist")
        if verbose > 0:
            print(
                "Check folder for any .csv files. Have you run ProcessFiles yet?")
        return 1


def GenerateCTR(df, reference=ufloat(42, 2), refflag=True, verbose=0):
    '''
    Calculates time resolution and error from scale parameter
    '''
#    TotalSigma = [ufloat(grp.scale, grp.scaleerr)
#                  for key, grp in df.groupby('uniquename')]
    
    

    if refflag==True:
        def afunc(srs):
                if srs["scale"]>reference.nominal_value:
                    return ufloat(srs["scale"], srs["scaleerr"])
                else:
                    return reference.nominal_value
                  
        TotalSigma = array(df.apply(afunc,axis=1))

        if verbose > 0:
            print("Subtracting",reference,"in quadrature!")
                        
        TimeResolution = [uncmath.sqrt(val**2 - reference**2) for val in TotalSigma]
    
    else:
        def afunc(srs):
            return ufloat(srs["scale"], srs["scaleerr"])
            
        TotalSigma = array(df.apply(afunc,axis=1))

        if verbose > 0:
            print("identical scintillator detectors")
        TimeResolution = [val / uncmath.sqrt(2) for val in TotalSigma]

    if verbose > 0:
        for i, val in enumerate(TimeResolution):
            print(i, ":", val, "ps")

        ##print("\nmean value :", mean(TimeResolution), "ps")
    
    return zip(*[[val.n, val.s] for val in TimeResolution])
    #return zip(*[[val.nominal_value, val.std_dev] for val in TimeResolution])


def CalculateDOI(df, minposition=34, whichkey='KeyWords', verbose=0):
    def TidyKeyWord(srs):  # generates DOI
        if isinstance(srs, float):
            return srs
        rmmm = srs[:-2]
        return float(rmmm)

    return minposition - df[whichkey].apply(TidyKeyWord)


def PrintLatexDOI(
        singlesampledf, RoundLvl=1, withsamplename=False, verbose=0):

    singlesampledf["Time Resolution (ps)"] = [str(
        round(grp["TimeResolution"],
              RoundLvl)) + "$\pm$" + str(round(grp["TimeResolutionError"],
                                               RoundLvl)) for key,
        grp in singlesampledf.groupby("DOI")]
    singlesampledf["CTR (ps)"] = [str(round(grp["CTR"], RoundLvl - 1)) + "$\pm$" + str(round(grp["CTRError"], RoundLvl - 1))
                                  for key, grp in singlesampledf.groupby("DOI")]
    singlesampledf = singlesampledf.sort("DOI")

    if withsamplename:
        print(
            singlesampledf.to_latex(
                index=False,
                columns=[
                    "DOI",
                    "SampleB",
                    "Time Resolution (ps)",
                    "CTR (ps)"]))
    else:
        print(
            singlesampledf.to_latex(
                index=False,
                columns=[
                    "DOI",
                    "Time Resolution (ps)",
                    "CTR (ps)"]))



def savefigureadir(name, loc, fig, ext='png', plos=False):
    '''
    Saves figure to location 
    loc/<name>/<name>.<ext> unless plos is true
    '''

    if plos:
        aname = name + ".tiff"
        if not os.path.exists(loc):
            os.makedirs(loc)

        saveloc = os.path.join(loc, aname)
    
    else:    
        extloc = os.path.join(loc, name)
        if not os.path.exists(extloc):
            os.makedirs(extloc)
    
        aname = name + '.' + ext
        saveloc = os.path.join(extloc, aname)
    
    fig.savefig(saveloc)

def savefigure(name, loc, fig, Ext=['pdf', 'eps', 'png', 'svg']):
    '''
    Saves figure to location given by rootloc/<ext>/<name>.<ext>
    '''

    for ext in Ext:
        extloc = os.path.join(loc, ext)
        if not os.path.exists(extloc):
            os.makedirs(extloc)
        
        aname = name + '.' + ext
        saveloc = os.path.join(extloc, aname)
        fig.savefig(saveloc)

