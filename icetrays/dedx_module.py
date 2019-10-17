from ROOT import TH1F
from ROOT import TGraphErrors
import numpy as np
import math
from icecube import icetray, dataclasses, simclasses, dataio, phys_services 


def dEdx_fit(frame, losses='newPS_SplineMPETruncatedEnergy_SPICEMie_BINS_Muon'):
    if losses not in frame.keys():
        return True
    dEdxVector = frame[losses]
    if 'Millipede' in losses:
        new_dEdxVector = [p.energy for p in dEdxVector if p.energy > 50.] # for millipede only use >50pe losses
        dEdxVector = new_dEdxVector

    if len(dEdxVector) > 2:
        bin = np.linspace(0.0, len(dEdxVector)-1, num=len(dEdxVector))
        bin2 = np.asarray(bin)
        dEdxVector2 = np.asarray(dEdxVector)
        hist = TH1F("hist","Header ; x-axis ; y-axis ",len(bin2),0,len(bin2))
        hist.Sumw2()
        peak_energy_loss = 0
        for i in range(len(bin2)):
            hist.Fill(i, dEdxVector2[i])
            if dEdxVector2[i] > peak_energy_loss:
                peak_energy_loss = dEdxVector2[i]
                peak = i
        median_energy_loss = np.median(dEdxVector2)
        PeakOverMedian = peak_energy_loss / median_energy_loss
        bins = len(bin2)
        func = hist.Fit("pol1", "NQS")
        chi2 = func.Chi2()
        slope = func.Value(1)
        NDF = len(bin2) - 2 #ndf is number of data points - number of fit parameters

        '''
        if PeakOverMedian > 8 and chi2 > 1.5:
            global cnt_peakovermedian
            cnt_peakovermedian+=1
            print "----------------------------------------------------------"
            print dEdxVector
            print "Bins: " + str(bins)
            print "peak energy loss: " + str(peak_energy_loss)
            print "median energy loss: " + str(median_energy_loss)
            print "PeakOverMedian: " + str(PeakOverMedian)
            print "chi2: " + str(chi2)
        '''

        # EXPERIMENTAL NEW CHI2
        for i in range(len(dEdxVector2)):
            dEdxVector2[i] = math.log(dEdxVector2[i])
        yerrs = np.linspace(0.22, 0.22, num=len(dEdxVector2))

        xvals = np.linspace(60, 60+(120*(len(dEdxVector2)-1)), num=len(dEdxVector2))
        xwidths = np.linspace(120, 120, num=len(dEdxVector2))

        gr = TGraphErrors(len(dEdxVector2),xvals,dEdxVector2,xwidths,yerrs)
        func2 = gr.Fit("pol1", "NQS")

        NewChi2 = func2.Chi2()
        ndf = func2.Ndf()

        NewChi2 = NewChi2 / ndf

       # print "----------------------------------------------------------"
       # print "dEdxVector: " 
       # print dEdxVector2 
       # print "yerrs: "
       # print yerrs       
       # print "xvals: "
       # print xvals
       # print "xerrors: " 
       # print xwidths
       # print "New Chi2: " + str(NewChi2)
       # print "Chi2: " + str(func2.Chi2())
       # print "NDF: " + str(ndf)

        frame['newPSCollection_'+losses] = dataclasses.I3MapStringDouble()
        frame['newPSCollection_'+losses]['NewChi2'] = NewChi2
        # ==================================

        frame['newPSCollection_'+losses]['chi2'] = chi2
        frame['newPSCollection_'+losses]['slope'] = slope
        frame['newPSCollection_'+losses]['NDF'] = NDF
        frame['newPSCollection_'+losses]['chi2_red'] = chi2 / NDF
        frame['newPSCollection_'+losses]['PeakOverMedian'] = PeakOverMedian
        frame['newPSCollection_'+losses]['bins'] = bins
        frame['newPSCollection_'+losses]['peak_energy_loss'] = peak_energy_loss
        return True
    else:
        return True


