
#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3/icetray-start
#METAPROJECT /data/user/tglauch/Software/combo/build
from icecube import dataio, dataclasses, icetray
from icecube import NewNuFlux
from os.path import join
import os
import numpy as np
import sys
sys.path.append('/data/user/mkronmueller/code/DeepIceLearning/lib')
from reco_quantities import testing_event, tau_decay_length, classify


def atm_flux(I3MCWeightDict):
    flux = NewNuFlux.makeFlux('honda2006')
    flux.knee_reweighting_model = "gaisserH4a_elbert"
    ptype = I3MCWeightDict["PrimaryNeutrinoType"]
    energy = I3MCWeightDict["PrimaryNeutrinoEnergy"]
    cos_theta = np.cos(I3MCWeightDict["PrimaryNeutrinoZenith"])
    type_weight = 0.5
    nevts = I3MCWeightDict["NEvents"]
    oneweight = I3MCWeightDict["OneWeight"]
    return flux.getFlux(ptype, energy, cos_theta) * oneweight / (type_weight * nevts)

def select_stream(phy_frame):
   if phy_f['I3EventHeader'].sub_event_stream != 'NullSplit':
       return False
   else:
       return True 
 

geo_file = '/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2016.57531_V0.i3.gz'
folder = '/data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2' 
s_fols = os.listdir(folder)
N1 = 0
N2 = 0
N3 = 0
N4 = 0
N5 = 0
N6 = 0
powerlaw = lambda energy: 0.91*1e-18*(energy/1e5)**(-2.19)
i = 0
f_ev = 0 
for s_fol in s_fols:
    files = os.listdir(join(folder, s_fol))
    for file in files:
        i+=1
        f_path = join(folder, s_fol, file)
        print('Read File: {}'.format(f_path))
        f = dataio.I3File(f_path)
        while f.more():
            phy_f = f.pop_physics()
	    sel = select_stream(phy_f)
            if sel == False:
                continue
	    tval = testing_event(phy_f, geo_file)
	    c_var = classify(phy_f, geo_file)
	    tdl = tau_decay_length(phy_f, geo_file)
	    if c_var!=5 or tdl<10:
                continue
	    print(phy_f['I3EventHeader'].event_id)
	    if tval==-1:
		print('continue...')
                continue
            w_dict = phy_f['I3MCWeightDict']
            primary_E = w_dict["PrimaryNeutrinoEnergy"]
            r1 = powerlaw(primary_E) * w_dict["OneWeight"] / (w_dict['NEvents'])
            r2 = atm_flux(w_dict)  
	    if primary_E>(50.*1e3):
                N3 += r1
	        N4 += r2
            if primary_E>(200.*1e3):
                N5 += r1
                N6 += r2
            f_ev += 1
            N1 += r1
            N2 += r2
        print('Astro Rate {}/year'.format(N1 / (i) * np.pi * 1e7 ))
        print('Atmo Rate {}/year'.format(N2 / (i) * np.pi * 1e7 ))
        print('Astro Rate {}/year (>50 TeV)'.format(N3 / (i) * np.pi * 1e7 ))
        print('Atmo Rate {}/year (>50 TeV)'.format(N4 / (i) * np.pi * 1e7 ))
        print('Astro Rate {}/year (>200 TeV)'.format(N5 / (i) * np.pi * 1e7 ))
        print('Atmo Rate {}/year (>200 TeV)'.format(N6 / (i) * np.pi * 1e7 ))
        print('Events/File {}'.format(1.*f_ev/i))    
