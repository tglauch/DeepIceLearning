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

def calc_depositedE(physics_frame):
    I3Tree = physics_frame['I3MCTree']
    truncated_energy = 0
    for i in I3Tree:
        interaction_type = str(i.type)
        if interaction_type in ['DeltaE','PairProd','Brems','EMinus']:
            truncated_energy += i.energy
    return truncated_energy


def classificationTag(physics_frame):
    I3Tree = physics_frame['I3MCTree']
    if abs(I3Tree[0].pdg_encoding) == 12: # primary particle is a electron neutrino
        classificationTag = 1
    elif abs(I3Tree[0].pdg_encoding) == 14: # primary particle is a muon neutrino
        classificationTag = 2
    elif abs(I3Tree[0].pdg_encoding) == 16: # primary particle is a tau neutrino
        listChildren = I3Tree.children(I3Tree[1])
        if not listChildren: # without this, the function collapses
            classificationTag = 3
        else:
            for i in listChildren:
                if abs(i.pdg_encoding) == 13:
                     classificationTag = 2
                else:
                    classificationTag = 3
    else:
        print "Error: primary particle wasnt a neutrino"
    # classificationTag = 1 means cascade
    # classificationTag = 2 means track
    # classificationTag = 3 means double bang
    return classificationTag
