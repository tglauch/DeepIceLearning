import numpy as np
import scipy
import dashi as d

def energyResolutionGaussY(predicted, target, weights, Ebins):
    #https://arxiv.org/pdf/1311.4767.pdf
    e_reco_bins = Ebins
    e_true_bins = Ebins
    h_pred = d.factory.hist2d((predicted, target), bins=(e_reco_bins, e_true_bins), weights=weights)
    ffunc = lambda x, loc, scale, norm : norm*scipy.stats.norm.pdf(x,loc,scale)
    stds_per_energy = []
    means_per_energy = []
    for i in xrange(h_pred.bincontent.shape[1]):
        h_pred = d.factory.hist2d((predicted, target), 
                                  bins=(e_reco_bins,e_true_bins),
                                  weights=np.ones(len(target)))
        h_slice = h_pred[:,i]
        h_pred.bincontent = h_pred.bincontent * h_slice.bincontent[:,np.newaxis]
        hs = h_pred.bincontent.sum(axis=0)
        if hs.sum() != 0:
            hs = hs / hs.sum()
        nan_mask = ~np.isnan(hs)
        ppar, pcov = scipy.optimize.curve_fit(ffunc,
                                              h_pred.bincenters[1][nan_mask],
                                              hs[nan_mask],
                                              p0=[h_slice.bincenters[i],0.2,1])
        stds_per_energy.append(ppar[1])
        means_per_energy.append(ppar[0])
    return h_pred.bincenters[1], means_per_energy, stds_per_energy
