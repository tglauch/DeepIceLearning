#!/usr/bin/env python
import math
import numpy as np

from I3Tray import *

from icecube import icetray, dataclasses, simclasses, dataio, phys_services
from icecube import DomTools, linefit
from icecube import gulliver,paraboloid,lilliput

load("libtruncated_energy")

# Truncated energy
@icetray.traysegment
def Truncated(tray, Name, Pulses="", Seed="", Suffix="", If=lambda f: True,
    PhotonicsService="", Model=""):

    TruncatedName = "newPS_"+Seed+"TruncatedEnergy"+Suffix+Model # ! base result Name to put into frame
    tray.AddModule("I3TruncatedEnergy",
        RecoPulsesName = Pulses, # ! Name of Pulses
        RecoParticleName = Seed,
        ResultParticleName = TruncatedName, # ! Name of result Particle
        I3PhotonicsServiceName = PhotonicsService,  # ! Name of photonics service to use
        UseRDE = True, # ! Correct for HQE DOMs !!! MUST BE TRUE USUALLY, BUT GIVES A BUG IN TRUNCATED FOR IC86-1 SIMULATIONS 
        If = If )
