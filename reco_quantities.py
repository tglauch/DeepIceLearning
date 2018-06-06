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
import icecube.MuonGun
import numpy as np


def calc_depositedE(physics_frame):
    I3Tree = physics_frame['I3MCTree']
    losses = 0
    for p in I3Tree:
        if not p.is_cascade: continue
        if not p.location_type == dataclasses.I3Particle.InIce: continue 
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

def calc_hitDOMs(physics_frame):
    hitDOMS = 0
    pulses = physics_frame["InIcePulses"]
    # apply the pulsemask --> make it an actual mapping of omkeys to pulses
    pulses = pulses.apply(physics_frame)
    for key, pulses in pulses:
        hitDOMs += 1   
    return hitDOMs

def classificationTag(physics_frame):
    energy = calc_depositedE(physics_frame)
    ParticelList = [12, 14, 16]
    I3Tree = physics_frame['I3MCTree']
    primary_list = I3Tree.get_primaries()
    if len(primary_list) == 1:
        neutrino = I3Tree[0]
    else:
        for p in primary_list:
            pdg = p.pdg_encoding
            if abs(pdg) in ParticelList:
                neutrino = p
    if abs(neutrino.pdg_encoding) == 12: # primary particle is a electron neutrino
        classificationTag = 1
    elif abs(neutrino.pdg_encoding) == 14: # primary particle is a muon neutrino
        classificationTag = 2
    elif abs(neutrino.pdg_encoding) == 16: # primary particle is a tau neutrino
        listChildren = I3Tree.children(I3Tree.first_child(neutrino))
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
    N = 0
    ParticelList = [12, 14, 16]
    I3Tree = physics_frame['I3MCTree']
    primary_list = I3Tree.get_primaries()
    if len(primary_list) == 1:
        neutrino = I3Tree[0]
    else:
        for p in primary_list:
            pdg = p.pdg_encoding
            if abs(pdg) in ParticelList:
                neutrino = p
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=-N)
    intersections = surface.intersection(neutrino.pos + neutrino.length*neutrino.dir, neutrino.dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = 0 # starting event
    else:
        starting = 1 # through-going or stopping event
    return starting

def up_or_down(physics_frame):
    zenith = physics_frame["LineFit"].dir.zenith
    if zenith > 1.5*np.pi or zenith < 0.5*np.pi:
        up_or_down = 1 # down-going
    else:
        up_or_down = 0 # up-going    
    return up_or_down


def coincidenceLabel(physics_frame):
    primary_list = physics_frame["I3MCTree"].get_primaries() 
    if len(primary_list) > 1:
        coincidence = 1
    else:
        coincidence = 0
    return coincidence




