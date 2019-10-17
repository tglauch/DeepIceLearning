#!/usr/bin/env python
import math
import numpy as np
import os
import sys
from I3Tray import *
from glob import glob
from optparse import OptionParser

from icecube import dataio, tableio, rootwriter, portia
from icecube import icetray, dataclasses, simclasses, dataio, phys_services
from icecube import DomTools, linefit
from icecube import gulliver,paraboloid,lilliput

from te_segment import Truncated
from dedx_module import dEdx_fit

tray = I3Tray()

infiles = ["/data/exp/IceCube/2014/filtered/level2/VerifiedGCD/Level2_IC86.2014_data_Run00124707_0508_0_90_GCD.i3.gz"]
infiles += ["/data/ana/Diffuse/AachenUpgoingTracks/exp/2014_pass2/paraboloid_from_finallevel/FinalLevel_NuMu_IC86-2014_9.i3.zst"]

tray.AddModule("I3Reader","reader",FilenameList = infiles)

tray.AddService( "I3PhotonicsServiceFactory", "PhotonicsServiceMu_SpiceMie",
    PhotonicsTopLevelDirectory="/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/",
    DriverFileDirectory="/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/driverfiles",
    PhotonicsLevel2DriverFile="mu_photorec.list",
    PhotonicsTableSelection=2,
    ServiceName="PhotonicsServiceMu_SpiceMie")

tray.AddSegment(Truncated,
    Pulses="TWSRTHVInIcePulsesIC",
    Seed="SplineMPE",
    Suffix="",
    PhotonicsService="PhotonicsServiceMu_SpiceMie",
    Model="_SPICEMie")

tray.AddModule(dEdx_fit, 'dEdx_fit1', losses='newPS_SplineMPETruncatedEnergy_SPICEMie_BINS_dEdxVector')
tray.AddModule(dEdx_fit, 'dEdx_fit2', losses='SplineMPE_MillipedeHighEnergyMIE')

tray.AddModule("I3Writer", "EventWriter",
                   FileName="/data/user/hmniederhausen/point_sources/stochasticity/test.i3.bz2",
                   Streams=[icetray.I3Frame.TrayInfo,
                            icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics,
                            icetray.I3Frame.Stream('S')],
                   DropOrphanStreams=[icetray.I3Frame.DAQ],
                   )

tray.AddModule('TrashCan','can')
tray.Execute(20)
tray.Finish()
