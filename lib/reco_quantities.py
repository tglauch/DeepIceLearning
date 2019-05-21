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

from icecube import dataclasses, dataio, icetray, MuonGun
from icecube.icetray import I3Units
import icecube.MuonGun
import numpy as np
from icecube.weighting.weighting import from_simprod
from icecube.icetray import I3Units

gcdfile_path = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2013.56429_V0.i3.gz'
surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile_path, padding=0)

nu_pdg = [12, 14, 16, -12, -14, -16]

weight_info = {
     '11029': {'nfiles': 3190,'nevents': 200000},
     '11069': {'nfiles': 3920,'nevents': 5000},
     '11070': {'nfiles': 997,'nevents': 400}, }

def calc_gen_ow(frame, gcdfile):
    soft = from_simprod(11029)
    hard_lowE = from_simprod(11069)
    hard_highE = from_simprod(11070)
    generator = 3190 * soft + 3920 * hard_highE + 997 * hard_lowE
    unit = I3Units.cm2/I3Units.m2
    gen_w = generator(frame['MCPrimary1'].energy, frame['I3MCWeightDict']['PrimaryNeutrinoType'], np.cos(frame['MCPrimary1'].dir.zenith))
    pint = frame['I3MCWeightDict']['TotalWeight']
    ow = pint/gen_w/unit
    return ow

def calc_depositedE(frame):
    I3Tree = frame['I3MCTree']
    losses = 0
    for p in I3Tree:
        if not p.is_cascade: continue
        if not (has_signature(p, gcdfile_path) == 0): continue 
        if p.shape == p.Dark: continue 
        if p.type in [p.Hadrons, p.PiPlus, p.PiMinus, p.NuclInt]:
            if p.energy < 1*I3Units.GeV:
                losses += 0.8*p.energy
            else:
                energyScalingFactor = 1.0 + ((p.energy/I3Units.GeV/0.399)**-0.130)*(0.467 - 1)
                losses += energyScalingFactor*p.energy
        else:
            losses += p.energy
    print(losses)
    frame.Put("depE", dataclasses.I3Double(losses)) 
    return 

def calc_hitDOMs(physics_frame, gcdfile, which=""):
    IC_hitDOMs = 0
    DC_hitDOMs = 0
    DC = [79, 80, 81, 82, 83, 84, 85, 86]
    pulses = physics_frame["InIceDSTPulses"]
    # apply the pulsemask --> make it an actual mapping of omkeys to pulses
    pulses = pulses.apply(physics_frame)
    for key, pulses in pulses:
        if key[0] in DC:
            DC_hitDOMs +=1
        else:
            IC_hitDOMs +=1
    if which == "IC":
        return IC_hitDOMs
    if which == "DC":
        return DC_hitDOMs

def starting(p_frame, gcdfile):
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame, gcdfile)
    primary_list = I3Tree.get_primaries()
    intersections = surface.intersection(neutrino.pos + neutrino.length * neutrino.dir, neutrino.dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = 0  # starting event
    else:
        starting = 1  # through-going or stopping event
    return starting


def coincidenceLabel_poly(physics_frame, gcdfile):
    poly = physics_frame['PolyplopiaCount']
    #print "Poly {}".format(poly)
    if poly == icetray.I3Int(0):
        coincidence = 0
    #elif poly == icetray.I3Int(0):
    #    coincidence = 0
    else:
        coincidence = 1
    return coincidence

def coincidenceLabel_primary(physics_frame, gcdfile):
    primary_list = physics_frame["I3MCTree"].get_primaries()
    if len(primary_list) > 1:
        coincidence = 1
    else:
        coincidence = 0
    return coincidence


def tau_decay_length(p_frame, gcdfile):
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame, gcdfile)
    if abs(neutrino.pdg_encoding) == 16:
        return I3Tree.children(neutrino)[0].length
    else:
        return -1


# calculates if the particle is in or near the detector
# if this is the case it further states weather the event is starting,
# stopping or through-going

def has_signature(p, gcdfile):
    intersections = surface.intersection(p.pos, p.dir)
    if p.is_neutrino:
        return -1
    if not np.isfinite(intersections.first):
        return -1
    if p.is_cascade:
        if intersections.first <= 0 and intersections.second > 0:
            return 0  # starting event
        else:
            return -1  # no hits
    elif p.is_track:
        if intersections.first <= 0 and intersections.second > 0:
            return 0  # starting event
        elif intersections.first > 0 and intersections.second > 0:
            if p.length <= intersections.first:
                return -1  # no hit
            elif p.length > intersections.second:
                return 1  # through-going event
            else:
                return 2  # stopping event
        else:
            return -1


def get_the_right_particle(p_frame, gcdfile):
    I3Tree = p_frame['I3MCTree']
    # find first neutrino as seed for find_particle
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    p_list = find_particle(p, I3Tree, gcdfile)
    if len(p_list) == 0 or len(p_list) > 1:
        return -1
    else:
        return p_list[0]


def find_particle(p, I3Tree, gcdfile):
# returns a list of neutrinos, that children interact with the detector,
# determines after the level, where one is found
    t_list = []
    children = I3Tree.children(p)
    #print "Len Children {}".format(len(children))
    if len(children) > 3:
        return []
    IC_hit = np.any([(has_signature(tp, gcdfile) != -1) for tp in children])
    if IC_hit:
        if p.pdg_encoding not in nu_pdg:
            return [I3Tree.parent(p)]
        else:
            return [p]
    elif (len(children) > 0) or (p.type_string == 'Hadrons'):
        for child in children:
            add_to_list = find_particle(child, I3Tree, gcdfile)
            t_list = np.concatenate([t_list, add_to_list])
        return t_list
    else:
        return []


# Generation of the Classification Label


def classify(p_frame):
    gcdfile = gcdfile_path
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame, gcdfile)
    if isinstance(neutrino, int):
        p_frame.Put("classification", dataclasses.I3Double(-1)) # Not Energy Loss in Detector
        return
    children = I3Tree.children(neutrino)
    p_types = [np.abs(child.pdg_encoding) for child in children]
    p_strings = [child.type_string for child in children]
    p_frame.Put("visible_nu", neutrino)

    if p_frame['I3MCWeightDict']['InteractionType'] == 3 and (len(p_types) == 1 and p_strings[0] == 'Hadrons'):
        pclass = 7  # Glashow Cascade
    elif np.any([p_type in nu_pdg for p_type in p_types]) and not (p_frame['I3MCWeightDict']['InteractionType'] == 3):
        pclass = 0  # is NC event
    else:
        if (11 in p_types):
            pclass = 1  # Cascade
        elif (13 in p_types):
            mu_ind = p_types.index(13)
            p_frame.Put("visible_track", children[mu_ind])
            if 'Hadrons' not in p_strings:
                if has_signature(children[mu_ind], gcdfile) == 0:
                    pclass = 8  # Glashow Track
            if has_signature(children[mu_ind], gcdfile) == 0:
                pclass = 3  # Starting Track
            if has_signature(children[mu_ind], gcdfile) == 1:
                pclass = 2  # Through Going Track
            if has_signature(children[mu_ind], gcdfile) == 2:
                pclass = 4  # Stopping Track
        elif (15 in p_types):
            tau_ind = p_types.index(15)
            p_frame.Put("visible_track", children[tau_ind])
            # consider to use the interactiontype here...
            if 'Hadrons' not in p_strings:
                pclass =  9  # Glashow Tau
            had_ind = p_strings.index('Hadrons')
            try:
                tau_child = I3Tree.children(children[tau_ind])[-1]
            except:
                tau_child = None
            if tau_child:
                if np.abs(tau_child.pdg_encoding) == 13:
                    if has_signature(tau_child, gcdfile) == 0:
                        pclass = 3  # Starting Track
                    if has_signature(tau_child, gcdfile) == 1:
                        pclass = 2  # Through Going Track
                    if has_signature(tau_child, gcdfile) == 2:
                        pclass = 4  # Stopping Track
                else:
                    if has_signature(children[tau_ind], gcdfile) == 0 and has_signature(tau_child, gcdfile) == 0:
                        pclass = 5  # Double Bang
                    if has_signature(children[tau_ind], gcdfile) == 0 and has_signature(tau_child, gcdfile) == -1:
                        pclass = 3  # Starting Track
                    if has_signature(children[tau_ind], gcdfile) == -1 and has_signature(tau_child, gcdfile) == 0:
                        pclass = 6  # Stopping Tau
            else: # Tau Decay Length to large, so no childs are simulated
                if has_signature(children[tau_ind], gcdfile) == 0:
                    pclass = 3 # Starting Track
                if has_signature(children[tau_ind], gcdfile) == 1:
                    pclass = 2  # Through Going Track
                if has_signature(children[tau_ind], gcdfile) == 2:
                    pclass = 4  # Stopping Track
        else:
            print('Bad Error!')
    print pclass
    p_frame.Put("classification", dataclasses.I3Double(pclass))
    return

def get_inelasticity(p_frame, gcdfile):
    I3Tree = p_frame['I3MCTree']
    interaction_type = p_frame['I3MCWeightDict']['InteractionType']
    if interaction_type!= 1:
        return 0
    else:
        neutrino = get_the_right_particle(p_frame, gcdfile)
        children = I3Tree.children(neutrino)
        for child in children:
            if child.type_string == "Hadrons":
                return 1.0*child.energy/neutrino.energy 

def millipede_rel_highest_loss(frame, gcdfile):
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy > 0.]
    if len(e_losses) == 0:
        return 0
    return np.max(e_losses) / np.sum(e_losses)


def millipede_n_losses(frame, gcdfile):
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy > 0.]
    return len(e_losses)


def millipede_std(frame, gcdfile):
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy>0.]
    if len(e_losses) == 0:
        return 0
    return np.std(e_losses)/np.mean(e_losses)


def millipede_max_loss(frame, gcdfile):
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy>0.]
    if len(e_losses) == 0:
        return 0
    return np.amax(e_losses)

def get_most_E_muon_info(frame):

    e0_list = []
    edep_list = []
    particle_list = []
    for track in icecube.MuonGun.Track.harvest(frame['I3MCTree'], frame['MMCTrackList']):
        # Find distance to entrance and exit from sampling volume
        intersections = surface.intersection(track.pos, track.dir)
        # Get the corresponding energies
        e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
        e0_list.append(e0)
        edep_list.append((e0-e1))
        particle_list.append(track)
        # Accumulate
    edep_list = np.array(edep_list)
    inds = np.argsort(edep_list)[::-1]
    e0_list = np.array(e0_list)[inds]
    particle_list = np.array(particle_list)[inds]
    if len(particle_list) == 0:
        print('no clear muon')
        return False
    frame.Put("Reconstructed_Muon", particle_list[0])
    frame.Put("mu_E_on_entry", dataclasses.I3Double(e0_list[0]))
    frame.Put("mu_E_deposited", dataclasses.I3Double(edep_list[inds][0]))
    return

def set_signature(frame):
    if "visible_nu" not in frame.keys():
        frame.Put("signature", dataclasses.I3Double(-1))
        return
    if "visible_track" in frame.keys():
        p = frame["visible_track"]
        intersections = surface.intersection(p.pos, p.dir)
    else:
        p = frame["visible_nu"]
        intersections = surface.intersection(p.pos+p.length*p.dir, p.dir)
    if not np.isfinite(intersections.first):
        val = -1
    elif p.is_neutrino:
        if intersections.first <= 0 and intersections.second > 0:
            val = 0  # starting event
        else:
            val =  -1  # no hits
    elif p.is_track:
        if intersections.first <= 0 and intersections.second > 0:
            val =  0  # starting event
        elif intersections.first > 0 and intersections.second > 0:
            if p.length <= intersections.first:
                val = -1  # no hit
            elif p.length > intersections.second:
                val = 1  # through-going event
            else:
                val = 2  # stopping event
        else:
            val = -1
    frame.Put("signature", dataclasses.I3Double(val))
    return


def muon_track_length(frame, key="Reconstructed_Muon"):
    p = frame[key]
    intersections = surface.intersection(p.pos, p.dir)
    if frame['signature'].value == 0:
        val = intersections.second # Starting Track
    elif frame['signature'].value == 1:
        val = intersections.second-intersections.first # Through Going Track 
    elif frame['signature'].value == 2:
        val = p.length-intersections.first # Stopping Track
    else:
        print('mmh {}'.format(frame['signature'].value))
        val = -1
    frame.Put("track_length", dataclasses.I3Double(val))
    return
