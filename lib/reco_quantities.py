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


nu_pdg = [12, 14, 16, -12, -14, -16]



def is_data(frame):
    if ('I3MCWeightDict' in frame) or ('CorsikaWeightMap' in frame) or ('MCPrimary' in frame) or ('I3MCTree' in frame):
        return False
    else:
        return True

def calc_depositedE(frame, gcdfile=None, surface=None, key='I3MCTree'):
    if is_data(frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    if isinstance(frame[key], icecube.dataclasses.I3Particle):
        I3Tree = [frame[key]]
    else:
        I3Tree = frame[key]
    losses = 0
    for p in I3Tree:
        if not p.is_cascade: continue
        if not (has_signature(p, surface) == 0): continue 
        if p.shape == p.Dark: continue 
        if p.type in [p.Hadrons, p.PiPlus, p.PiMinus, p.NuclInt]:
            if p.energy < 1*I3Units.GeV:
                losses += 0.8*p.energy
            else:
                energyScalingFactor = 1.0 + ((p.energy/I3Units.GeV/0.399)**-0.130)*(0.467 - 1)
                losses += energyScalingFactor*p.energy
        else:
            losses += p.energy
    #print('deposited energy {}'.format(losses))
    frame.Put("depE", dataclasses.I3Double(losses)) 
    return 

def calc_depositedE_single_p(p, I3Tree, surface):
    losses = 0
    for p in I3Tree.get_daughters(p):
        if not p.is_cascade: continue
        if not (has_signature(p, surface) == 0): continue
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

def calc_hitDOMs(frame, pulsemap="InIceDSTPulses"):
    IC_hitDOMs = 0
    DC_hitDOMs = 0
    DC = [79, 80, 81, 82, 83, 84, 85, 86]
    pulses = frame[pulsemap]
    # apply the pulsemask --> make it an actual mapping of omkeys to pulses
    try:
        pulses = pulses.apply(frame)
    except Exception as ex:
        print('skip')
        print ex
        return False
    for key, pulses in pulses:
        if key[0] in DC:
            DC_hitDOMs +=1
        else:
            IC_hitDOMs +=1
    frame.Put("IC_hit_doms", dataclasses.I3Double(IC_hitDOMs))
    frame.Put("DC_hit_doms", dataclasses.I3Double(DC_hitDOMs))
    return

def starting(p_frame, gcd=None, surface=None):
    if is_data(p_frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame)
    primary_list = I3Tree.get_primaries()
    intersections = surface.intersection(neutrino.pos + neutrino.length * neutrino.dir, neutrino.dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = 0  # starting event
    else:
        starting = 1  # through-going or stopping event
    return starting


def coincidenceLabel_poly(p_frame):
    if is_data(p_frame):
        return True
    if 'I3MCWeightDict' not in p_frame.keys():
        return
    poly = p_frame['PolyplopiaCount']
    #print('Poly? {}'.format(poly.value))
    if poly == icetray.I3Int(0):
        coincidence = 0
    else:
        coincidence = 1
    p_frame.Put("multiplicity", icetray.I3Int(coincidence))    
    return

def coincidenceLabel_primary(physics_frame):
    if is_data(physics_frame):
        return True
    primary_list = physics_frame["I3MCTree"].get_primaries()
    if len(primary_list) > 1:
        coincidence = 1
    else:
        coincidence = 0
    return coincidence


def tau_decay_length(p_frame):
    if is_data(p_frame):
        return True
    I3Tree = p_frame['I3MCTree']
    neutrino = get_the_right_particle(p_frame)
    if abs(neutrino.pdg_encoding) == 16:
        return I3Tree.children(neutrino)[0].length
    else:
        return -1


# calculates if the particle is in or near the detector
# if this is the case it further states weather the event is starting,
# stopping or through-going

def has_signature(p, surface):
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

def get_primary_nu(p_frame):
    if is_data(p_frame):
        return True
    if 'I3MCWeightDict' not in p_frame.keys():
        return True
    I3Tree = p_frame['I3MCTree']
    # find first neutrino as seed for find_particle
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    p_frame.Put("primary_nu", p)
    return


def get_the_right_particle(p_frame, gcdfile=None, surface=None):
    if is_data(p_frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    I3Tree = p_frame['I3MCTree']
    # find first neutrino as seed for find_particle
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    p_list = find_particle(p, I3Tree, surface)
    if len(p_list) == 0 or len(p_list) > 1:
        return -1
    else:
        return p_list[0]


def find_particle(p, I3Tree, surface):
# returns a list of neutrinos, that children interact with the detector,
# determines after the level, where one is found
    if is_data(frame):
        return True
    t_list = []
    children = I3Tree.children(p)
    if len(children) > 3:
        return []
    IC_hit = np.any([((has_signature(tp, surface) != -1) & np.isfinite(tp.length)) for tp in children])
    if IC_hit:
        if p.pdg_encoding not in nu_pdg:
            return [I3Tree.parent(p)]
        else:
            return [p]
    elif len(children) < 3:
        for child in children:
            add_to_list = find_particle(child, I3Tree, surface)
            t_list.extend(add_to_list)
        return t_list
    else:
        return []


def find_all_neutrinos(p_frame):
    if is_data(p_frame):
        return True
    I3Tree = p_frame['I3MCTree']
    # find first neutrino as seed for find_particle
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    all_nu = [i for i in crawl_neutrinos(p, I3Tree, plist=[]) if len(i) > 0]
    return all_nu[-1][0]


def crawl_neutrinos(p, I3Tree, level=0, plist = []):
    if len(plist) < level+1:
        plist.append([])
    if (p.is_neutrino) & np.isfinite(p.length):
        plist[level].append(p) 
    children = I3Tree.children(p)
    if len(children) < 10:
        for child in children:
            crawl_neutrinos(child, I3Tree, level=level+1, plist=plist)
    return plist

 
# Generation of the Classification Label
def classify(p_frame, gcdfile=None, surface=None):
    if is_data(p_frame):
        return True
    pclass = 101 # only for security
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(p_frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    I3Tree = p_frame['I3MCTree']
    neutrino = find_all_neutrinos(p_frame)
    children = I3Tree.children(neutrino)
    p_types = [np.abs(child.pdg_encoding) for child in children]
    p_strings = [child.type_string for child in children]
    p_frame.Put("visible_nu", neutrino)
    IC_hit = np.any([((has_signature(tp, surface) != -1) & np.isfinite(tp.length)) for tp in children])
    if p_frame['I3MCWeightDict']['InteractionType'] == 3 and (len(p_types) == 1 and p_strings[0] == 'Hadrons'):
        pclass = 7  # Glashow Cascade
    else:
        if (11 in p_types) or (p_frame['I3MCWeightDict']['InteractionType'] == 2):
            if IC_hit:
                pclass = 1  # Cascade
            else:
                pclass = 0 # Uncontainced Cascade
        elif (13 in p_types):
            mu_ind = p_types.index(13)
            p_frame.Put("visible_track", children[mu_ind])
            if not IC_hit:
                pclass = 11 # Passing Track
            elif p_frame['I3MCWeightDict']['InteractionType'] == 3:
                if has_signature(children[mu_ind], surface) == 0:
                    pclass = 8  # Glashow Track
            elif has_signature(children[mu_ind], surface) == 0:
                pclass = 3  # Starting Track
            elif has_signature(children[mu_ind], surface) == 1:
                pclass = 2  # Through Going Track
            elif has_signature(children[mu_ind], surface) == 2:
                pclass = 4  # Stopping Track
        elif (15 in p_types):
            tau_ind = p_types.index(15)
            p_frame.Put("visible_track", children[tau_ind])
            if not IC_hit:
                pclass = 12 # uncontained tau something...
            else:
                # consider to use the interactiontype here...
                if p_frame['I3MCWeightDict']['InteractionType'] == 3:
                    pclass =  9  # Glashow Tau
                else:
                    had_ind = p_strings.index('Hadrons')
                    try:
                        tau_child = I3Tree.children(children[tau_ind])[-1]
                    except:
                        tau_child = None
                    if tau_child:
                        if np.abs(tau_child.pdg_encoding) == 13:
                            if has_signature(tau_child, surface) == 0:
                                pclass = 3  # Starting Track
                            if has_signature(tau_child, surface) == 1:
                                pclass = 2  # Through Going Track
                            if has_signature(tau_child, surface) == 2:
                                pclass = 4  # Stopping Track
                        else:
                            if has_signature(children[tau_ind], surface) == 0 and has_signature(tau_child, surface) == 0:
                                pclass = 5  # Double Bang
                            if has_signature(children[tau_ind], surface) == 0 and has_signature(tau_child, surface) == -1:
                                pclass = 3  # Starting Track
                            if has_signature(children[tau_ind], surface) == 2 and has_signature(tau_child, surface) == 0:
                                pclass = 6  # Stopping Tau
                            if has_signature(children[tau_ind], surface) == 1:
                                pclass = 2  # Through Going Track
                    else: # Tau Decay Length to large, so no childs are simulated
                        if has_signature(children[tau_ind], surface) == 0:
                            pclass = 3 # Starting Track
                        if has_signature(children[tau_ind], surface) == 1:
                            pclass = 2  # Through Going Track
                        if has_signature(children[tau_ind], surface) == 2:
                            pclass = 4  # Stopping Track
        else:
            pclass = 100 # unclassified
    #print('Classification: {}'.format(pclass))
    p_frame.Put("classification", icetray.I3Int(pclass))
    return


def classify_muongun(p_frame, gcdfile=None, surface=None, primary_key='MCPrimary'):
    if is_data(p_frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(p_frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    p = p_frame[primary_key]
    if has_signature(p, surface) == 1:
        pclass = 2  # Through Going Track
    elif has_signature(p, surface) == 2:
        pclass = 4  # Stopping Track
    else:
        pclass = 0
    p_frame.Put("classification", icetray.I3Int(pclass))
    p_frame.Put("visible_track", p)
    return

def get_inelasticity(p_frame, mctree='I3MCTree'):
    if is_data(p_frame):
        return True
    I3Tree = p_frame[mctree]
    if 'I3MCWeightDict' not  in p_frame.keys():
        return
    interaction_type = p_frame['I3MCWeightDict']['InteractionType']
    if interaction_type!= 1:
        inela = 0.
    else:
        neutrino = p_frame['visible_nu']
        children = I3Tree.children(neutrino)
        for child in children:
            if child.type_string == "Hadrons":
                inela = 1.0*child.energy/neutrino.energy  
    p_frame.Put("inelasticity", dataclasses.I3Double(inela))
    return

def millipede_rel_highest_loss(frame):
    if not 'SplineMPE_MillipedeHighEnergyMIE' in frame:
        return
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy > 0.]
    if len(e_losses) == 0:
        val = 0
    else:
        val = np.max(e_losses) / np.sum(e_losses)
    frame.Put("millipede_rel_highest_loss", dataclasses.I3Double(val))
    return


def millipede_n_losses(frame):
    if not 'SplineMPE_MillipedeHighEnergyMIE' in frame:
        return
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy > 0.]
    frame.Put("millipede_n_loss", dataclasses.I3Double(len(e_losses)))
    return


def millipede_std(frame):
    if not 'SplineMPE_MillipedeHighEnergyMIE' in frame:
        return
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy>0.]
    if len(e_losses) == 0:
        val = 0
    else:
        val = np.std(e_losses)/np.mean(e_losses)
    frame.Put("millipede_std", dataclasses.I3Double(val))
    return


def millipede_max_loss(frame):
    if not 'SplineMPE_MillipedeHighEnergyMIE' in frame:
        return
    e_losses = [i.energy for i in frame['SplineMPE_MillipedeHighEnergyMIE'] if i.energy>0.]
    if len(e_losses) == 0:
        val= 0
    else:
        val = np.amax(e_losses)
    frame.Put("millipede_max_loss", dataclasses.I3Double(val))
    return

def get_most_E_muon_info(frame, gcdfile=None, surface=None, tracklist='MMCTrackList', mctree='I3MCTree'):
    if is_data(frame):
        return True
    if tracklist not in frame:
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    e0_list = []
    edep_list = []
    particle_list = []
    if isinstance(frame[tracklist], icecube.dataclasses.I3Particle):
        tlist = [frame[tracklist]]
    else:
        tlist=icecube.MuonGun.Track.harvest(frame[mctree], frame[tracklist])
    for track in tlist:
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
    else:
        frame.Put("Reconstructed_Muon", particle_list[0])
        frame.Put("mu_E_on_entry", dataclasses.I3Double(e0_list[0]))
        frame.Put("mu_E_deposited", dataclasses.I3Double(edep_list[inds][0]))
    return

def set_signature(frame, gcdfile=None, surface=None):
    if is_data(frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    if "visible_track" in frame.keys():
        p = frame["visible_track"]
        intersections = surface.intersection(p.pos, p.dir)
    elif "visible_nu" in frame.keys():
        p = frame["visible_nu"]
        intersections = surface.intersection(p.pos+p.length*p.dir, p.dir)
    else:
        val = -1
        frame.Put("signature", icetray.I3Int(val))
        return
    if not np.isfinite(intersections.first):
        val = -1
    elif p.is_neutrino:
        if intersections.first <= 0 and intersections.second > 0:
            val = 0  # starting event
        else:
            val =  -1  # vertex outside detector
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
            val = -1 # vertex behind detector
    frame.Put("signature", icetray.I3Int(val))
    return


def track_length_in_detector(frame, gcdfile=None, surface=None,  key="visible_track"):
    if is_data(frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)    
    if not key in frame.keys():
        val = 0.
    else:
        p = frame[key]
        intersections = surface.intersection(p.pos, p.dir)
        if frame['classification'].value == 3:
            val = intersections.second # Starting Track
        elif frame['classification'].value == 2:
            val = intersections.second-intersections.first # Through Going Track 
        elif frame['classification'].value == 4:
            val = p.length-intersections.first # Stopping Track
        elif frame['classification'].value == 5:
            val = p.length
        elif (frame['classification'].value == 21) | (frame['classification'].value == 22) | (frame['classification'].value == 23):
            val = np.min([p.length-intersections.first,intersections.second-intersections.first])
        else:
            val = 0.
    frame.Put("track_length", dataclasses.I3Double(val))
    return




def first_interaction_point(frame, gcdfile=None, surface=None):
    if is_data(frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    particle= dataclasses.I3Particle()
    if (frame['signature'].value == 1) or (frame['signature'].value == 2):
        vis_lep = frame['visible_track']
        intersections = surface.intersection(vis_lep.pos, vis_lep.dir)
        particle.pos = vis_lep.pos + intersections.first * vis_lep.dir
    elif 'visible_nu' in frame.keys():
        vis_nu = frame["visible_nu"]
        particle.pos= vis_nu.pos + vis_nu.dir * vis_nu.length
    else:
        return
    print particle.pos
    frame.Put("first_interaction_pos", particle)
    return


def classify_wrapper(p_frame, surface, gcdfile=None):
    if is_data(p_frame):
        return True
    if 'I3MCWeightDict' in p_frame.keys():
        classify(p_frame, surface=surface, gcdfile=gcdfile)
        return
    else:
        classify_corsika(p_frame, surface=surface, gcdfile=gcdfile)
        return


def classify_corsika(p_frame, gcdfile=None, surface=None):
    if is_data(p_frame):
        return True
    if surface is None:
        if gcdfile is None:
            surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(p_frame['I3Geometry'])
        else:
            surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)
    mu_list = []
    I3Tree = p_frame['I3MCTree'] 
    primaries = I3Tree.get_primaries()
    for p in primaries:
        tlist = []
        find_muons(p, I3Tree, surface, plist=tlist)
        mu_list.append(tlist[-1])
    
    if len(np.concatenate(mu_list)) == 0:
        pclass = 11 # Passing Track
    elif len(mu_list)>1:
        pclass = 21 # several events
        clist = np.concatenate(mu_list)
        energies = np.array([calc_depositedE_single_p(p, I3Tree, surface) for p in clist])
        inds = np.argsort(energies)
        p_frame.Put("visible_track", clist[inds[-1]])
    
    else: 
        if len(mu_list[0]) >1:
            energies = np.array([calc_depositedE_single_p(p, I3Tree, surface) for p in mu_list[0]])
            mu_signatures = np.array([has_signature(p, surface) for p in mu_list[0]])
            if np.any(mu_signatures==1):
                pclass = 22 # Through Going Bundle 
            else:
                pclass = 23 # Stopping Muon Bundle
            inds = np.argsort(energies)
            p_frame.Put("visible_track", mu_list[0][inds[-1]])
        else:
            if has_signature(mu_list[0][0], surface) == 2:
                pclass = 4 # Stopping Track
            else:
                pclass = 2 # Through Going Track
            p_frame.Put("visible_track", mu_list[0][0])
    p_frame.Put("classification", icetray.I3Int(pclass))
    return

    
def find_muons(p, I3Tree, surface, level=0, plist = []):
    if len(plist) < level+1:
        plist.append([])
    if (np.abs(p.pdg_encoding)==13):
        if np.isfinite(p.length) & (has_signature(p, surface) != -1):
            plist[level].append(p)
            calc_depositedE_single_p(p,I3Tree, surface)
    else:
        children = I3Tree.children(p)
        for child in children:
            find_muons(child, I3Tree, surface, level=level+1, plist=plist)
    return plist


def intersection_depth(x, y, z, zenith, azimuth, radius=500., length=1000.):
    b = x * numpy.cos(azimuth) + y * numpy.sin(azimuth)
    d = b * b + radius * radius - x * x - y * y

    h = [numpy.nan, numpy.nan]
    r = [numpy.nan, numpy.nan]

    if d > 0:
        d = numpy.sqrt(d)

        # Down-track distance to the endcaps
        costh = numpy.cos(zenith)

        if costh != 0.:
            h[0] = (z - length / 2.) / costh
            h[1] = (z + length / 2.) / costh
            h = sorted(h)

        # Down-track distance to the side surfaces
        sinth = numpy.sin(zenith)

        if sinth != 0.:
            r[0] = (b - d) / sinth
            r[1] = (b + d) / sinth
            r = sorted(r)

        # Perfectly horizontal tracks never intersect the endcaps.
        if costh == 0.:
            if ((z > -length / 2.) and (z < length / 2.)):
                h = r
            else:
                h = [numpy.nan, numpy.nan]
        # Perfectly vertical tracks never intersect the sides.
        elif sinth == 0.:
            if numpy.hypot(x, y) >= radius:
                h = [numpy.nan, numpy.nan]
        # For general tracks, take the last entrace and first exit.
        else:
            if (h[0] >= r[1]) or (h[1] <= r[0]):
                h = [numpy.nan, numpy.nan]
            else:
                h[0] = max(r[0], h[0])
                h[1] = min(r[1], h[1])

    return z + numpy.cos(numpy.pi - zenith) * min(h)


def calc_depth(phy_frame, primary_key='MCPrimary'):
    if is_data(p_frame):
        return True
    primary = phy_frame[primary_key]
    vertex = (primary.dir.x, primary.dir.y, primary.dir.z)
    zentry = intersection(*vertex,
                          zenith=primary.dir.zenith,
                          azimuth=primary.dir.azimuth)
    phy_frame.Put('depth', dataclasses.I3Double(1950. - zentry))
    return True
