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
    hitDOMs = 0
    pulses = physics_frame["InIceDSTPulses"]
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


def starting(p_frame, gcdfile):
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame, gcdfile)
    primary_list = I3Tree.get_primaries()
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=25)
    intersections = surface.intersection(neutrino.pos + neutrino.length * neutrino.dir, neutrino.dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = 0  # starting event
    else:
        starting = 1  # through-going or stopping event
    return starting


def up_or_down(physics_frame):
    zenith = physics_frame["LineFit"].dir.zenith
    if zenith > 1.5 * np.pi or zenith < 0.5 * np.pi:
        up_or_down = 1  # down-going
    else:
        up_or_down = 0  # up-going
    return up_or_down


def coincidenceLabel(physics_frame):
    primary_list = physics_frame["I3MCTree"].get_primaries()
    if len(primary_list) > 1:
        coincidence = 1
    else:
        coincidence = 0
    return coincidence


def tau_decay_length(p_frame):
    I3Tree = p_frame['I3MCTree']    
    neutrino = get_the_right_particle(p_frame, gcdfile)    
    if abs(neutrino.pdg_encoding) == 16:
        return I3Tree.children(neutrino)[0].length
    else:
        return -1


# calculates if the particle is in or near the detector
# if this is the case it further states weather the event is starting, stopping or through-going
def has_signature(p, gcdfile):
    surface = MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=25)
    intersections = surface.intersection(p.pos, p.dir)
    if p.is_neutrino:
        return -1
    if not np.isfinite(intersections.first):
        return -1
    if p.is_cascade:
        if intersections.first <= 0 and intersections.second > 0:
            return 0 # starting event
        else:
            return -1 # no hits
    elif p.is_track:
        if intersections.first <= 0 and intersections.second > 0:
            return 0 # starting event
        elif intersections.first > 0 and intersections.second > 0:
            if p.length <= intersections.first:
                return -1 # no hit
            elif p.length > intersections.second:
                return 1 #through-going event
            else:
                return 2 #stopping event
        else:
            return -1


def get_the_right_particle(p_frame, gcdfile):
    nu_pdg = [12, 14, 16, -12, -14, -16]
    I3Tree = p_frame['I3MCTree']
    #find first neutrino as seed for find_particle 
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    p_list  = find_particle(p, I3Tree, gcdfile)
    if len(p_list) == 0 or len(p_list)>1:
        return -1
    else:
        return p_list[0]


def testing_event(p_frame, gcdfile):
    nu_pdg = [12, 14, 16, -12, -14, -16]
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame, gcdfile)
    if neutrino == -1:
        return -1
    else:
        children = I3Tree.children(neutrino)
        p_types = [np.abs(child.pdg_encoding) for child in children]
        p_strings = [child.type_string for child in children]

        if not np.any([p_type in nu_pdg for p_type in p_types]) \
            and not ((11 in p_types) or (13 in p_types) or (15 in p_types)):
            return -1  # kick the event
        else:
            return 0  # everything fine

# returns a list of neutrinos, that children interact with the detector, determines after the level, where one is found
def find_particle(p, I3Tree, gcdfile):
    t_list = []
    nu_pdg = [12, 14, 16, -12, -14, -16]
    children = I3Tree.children(p)
    IC_hit = np.any([(has_signature(tp, gcdfile)!=-1) for tp in children])
    if IC_hit:
        if not p.pdg_encoding in nu_pdg:
            return [I3Tree.parent(p)]
        else:
            return [p]
    elif len(children)>0:
        for child in children:
            t_list = np.concatenate([t_list, find_particle(child, I3Tree, gcdfile)])
        return t_list
    else:
        return []



# Generation of the Classification Label	
def classify(p_frame, gcdfile):
    nu_pdg = [12, 14, 16, -12, -14, -16]
    I3Tree = p_frame['I3MCTree']
    #for p in I3Tree.get_primaries():
    #    if p.pdg_encoding in nu_pdg:
    #        break
    #p_list  = find_particle(p, I3Tree)
    #if len(p_list) == 0 or len(p_list)>1:
    #    return -1
    neutrino = get_the_right_particle(p_frame, gcdfile)
    children = I3Tree.children(neutrino)
    p_types = [np.abs(child.pdg_encoding) for child in children]
    p_strings = [child.type_string for child in children]
    
    #if not np.any([p_type in nu_pdg for p_type in p_types]) \
    #   and not ((11 in p_types) or (13 in p_types) or (15 in p_types)):
    #    return -1
    
    if np.any([p_type in nu_pdg for p_type in p_types]):
        return 0 # is NC event
    else:
        if (11 in p_types):
            return 1 # Cascade 
        elif (13 in p_types):
            mu_ind = p_types.index(13)
            had_ind = p_strings.index('Hadrons')
            if has_signature(children[had_ind], gcdfile) == 0:
                return 3 #Starting Track
            elif has_signature(children[mu_ind], gcdfile) == 1:
                return 2 # Through Going Track
            elif has_signature(children[mu_ind], gcdfile) == 2:
                return 4 ## Stopping Track
        elif (15 in p_types):
            tau_ind = p_types.index(15)
            had_ind = p_strings.index('Hadrons')
            tau_child = I3Tree.children(children[tau_ind])[-1]
            if np.abs(tau_child.pdg_encoding) == 13:
                if has_signature(children[had_ind], gcdfile) == 0:
                    return 3 # Starting Track
                elif has_signature(tau_child, gcdfile) == 1:
                    return 2 # Through Going Track
                elif has_signature(tau_child, gcdfile) == 2:
                    return 4 # Stopping Track
            
            else:
                if children[tau_ind].length < 5:
                    return 1 # Cascade
                if has_signature(children[had_ind], gcdfile) == 0 and has_signature(tau_child, gcdfile) == 0: 
                    return 5 # Double Bang
                elif has_signature(children[had_ind], gcdfile) == 0 and has_signature(tau_child, gcdfile) == -1:
                    return 3 # Starting Track
                elif has_signature(children[had_ind], gcdfile) == -1  and has_signature(tau_child, gcdfile) == 0:
                    return 6 # Stopping Tau

nomenclature = {-1: 'no Hit',
               0: 'NC',
               1: 'Cascade',
               2: 'Through-Going Track',
               3: 'Starting Track',
               4: 'Stopping Track',
               5: 'Double Bang',
               6: 'Stopping Tau'}

def time_of_percentage(charges, times, percentage):
    charges = charges.tolist()
    cut = np.sum(charges)/(100./percentage)
    sum=0
    for i in charges:
        sum = sum + i
        if sum > cut:
            tim = times[charges.index(i)]
            break
    return tim

