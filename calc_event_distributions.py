import numpy as np
import os
from fancy_plot import *

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpath",
        help="main config file, user-specific",
        type=str)
    args = parser.parse_args()
    return args
args = parseArguments().__dict__
bpath = args['bpath']

def calc_probabilities1(flist, etype_dict, add_str = '',
                        logE_bins = np.linspace(1.47,2.8,20),
                        cos_zen_bins = np.linspace(-1, 1, 10)):
    X,Y = np.meshgrid(logE_bins, cos_zen_bins)
    hist_dict = dict()
    for typ in etype_dict.keys():
        print typ
        hist_dict[typ] = []
        for f in flist:
            print f
            idata = h5py.File(f, 'r')
            classi = ifile['reco_vals']['classification']
            mask = np.array([True if classi[i] in etype_dict[typ] else False for i in range(len(classi))])
            H = np.histogram2d(np.log10(ifile['reco_vals']['ic_hitdoms'])[mask],
                               np.cos(ifile['reco_vals']['mc_prim_zen'][mask]),
                               bins=[logE_bins, cos_zen_bins])
            hist_dict[typ].append(H[0])
            idata.close()
        H_temp= np.sum(hist_dict[typ], axis=0)
        print('Minimum Value: {}'.format(np.min(H_temp)))
        print('Number of Zeros: {}'.format(len(np.where(H_temp.flatten()==0)[0])))
        H_temp[H_temp==0] = 1
        hist_dict[typ] = H_temp
        fig, ax = newfig(0.9)
        cbar = ax.pcolormesh(X,Y,hist_dict[typ].T)
        plt.colorbar(cbar)
        fig.savefig(os.path.join(bpath, 'plots', '{}_{}_tot.png'.format(typ, add_str)), dpi=300)
    H_tot = np.min(np.array([hist_dict[typ] for typ in etype_dict.keys()]), axis=0)
    print('Final H')
    print np.sum(H_tot)
    fig, ax = newfig(0.9)
    cbar = ax.pcolormesh(X,Y,H_tot.T)
    plt.colorbar(cbar)
    fig.show()
    # calc ratios
    for typ in etype_dict.keys():
        hist_dict[typ] = 1.*H_tot/hist_dict[typ]
        fig, ax = newfig(0.9)
        cbar = ax.pcolormesh(X,Y,hist_dict[typ].T)
        plt.colorbar(cbar)
        fig.savefig(os.path.join(bpath, 'plots', '{}_{}_ratio.png'.format(typ, add_str)), dpi=300)
        odict = {'logE_bins' : logE_bins,
                 'cos_zen_bins' : cos_zen_bins,
                 'H' : hist_dict[typ]}
        np.save(os.path.join('./pick_probs/', '{}_n_hit_doms{}.npy'.format(typ, add_str)), odict)
    
    return
            
   
os.makedirs(os.path.join(bpath, 'plots'))
os.makedirs('./pick_probs/') 
iflist = [os.path.join(bpath,i) for i in os.listdir(bpath) if '.h5' in i]
print('Using {}'.files(len(ifilist)))
etype_dict = {'starting': [3],
              'through' : [2,22],
              'cascade' : [1],
              'passing': [0,11]
}
bins_hits = np.linspace(1.6,2.8,20)
bins_cos_zen = np.linspace(-1, 1, 10)
calc_probabilities1(flist, etype_dict, logE_bins=bins_hits, cos_zen_bins=bins_cos_zen)

etype_dict = {'starting': [3],
 #             'through' : [2,22],
              'cascade' : [1],
              'passing': [0,11],
              'stopping': [4,23]
}
bins_hits = np.linspace(1.1,1.6,10)
bins_cos_zen = np.linspace(-1, 1, 10)
calc_probabilities1(flist, etype_dict, logE_bins=bins_hits, cos_zen_bins=bins_cos_zen, 
                    add_str='few_hits')
