#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
#METAPROJECT /data/user/tglauch/Software/combo/build
# coding: utf-8

"""This file is part of DeepIceLearning
DeepIceLearning is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from icecube import dataclasses, dataio, icetray
from icecube.icetray import I3Units


def calc_depositedE(physics_frame):
    I3Tree = physics_frame['I3MCTree']
    losses = 0
    for p in I3Tree:
        if not p.is_cascade: continue
        if not p.location_type == I3Particle.InIce: continue 
        if p.shape == p.Dark: continue 
        if p.type in [p.Hadrons, p.PiPlus, p.PiMinus, p.NuclInt]:
            if p.energy < 1*I3Units.GeV:
                losses += 0.8*p.energy
            else:
                energyScalingFactor = 1.0 + ((p.energy/I3Units.GeV/0.399)**-0.130)*(0.467 - 1)
                losses += energyScalingFactor*p.energy
        else:
            losses += p.energy 
    return losses


def classificationTag(physics_frame):
    energy = calc_depsitedE(physics_frame)
    I3Tree = physics_frame['I3MCTree']
    if abs(I3Tree[0].pdg_encoding) == 12: # primary particle is a electron neutrino
        classificationTag = 1
    elif abs(I3Tree[0].pdg_encoding) == 14: # primary particle is a muon neutrino
        classificationTag = 2
    elif abs(I3Tree[0].pdg_encoding) == 16: # primary particle is a tau neutrino
        listChildren = I3Tree.children(I3Tree[1])
        if not listChildren: # without this, the function collapses
            if energy > 10**6: # more than 1 PeV, due to energy in GeV
                classificationTag = 3
            else:
                classificationTag = 1
        else:
            for i in listChildren:
                if abs(i.pdg_encoding) == 13:
                     classificationTag = 2
                else:
                    if energy > 10**6: # more than 1 PeV, due to energy in GeV
                        classificationTag = 3
                    else:
                        classificationTag = 1
    else:
        print "Error: primary particle wasnt a neutrino"
    # classificationTag = 1 means cascade
    # classificationTag = 2 means track
    # classificationTag = 3 means double bang
    return classificationTag


def starting(physics_frame):
    gcdfile = "/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2012.56063_V0.i3.gz"
    N = 10
    neutrino = physics_frame['I3MCTree'][0]
    surface = MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=-N)
    intersections = surface.GetIntersection(neutrino.pos + neutrino.length*neutrino.dir, neutrino.dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = True
    else:
        starting = False
    return starting








