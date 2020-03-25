import sys
import os
from icecube import dataclasses
from icecube.weighting import weighting, get_weighted_primary
from icecube.weighting.fluxes import GaisserH4a
dil_path = '/data/user/tglauch/DeepIceLearning'
sys.path.append(dil_path)
sys.path.append(os.path.join(dil_path,'lib'))
import lib.reco_quantities as reco_q
from icecube import NewNuFlux

low_e = weighting.from_simprod(11499)
high_e = weighting.from_simprod(11057)
generator = 1000*low_e+1129*high_e
flux = GaisserH4a()

def corsika_weight(phy_frame):
    if 'I3MCWeightDict' in phy_frame:
        return
    energy = phy_frame['MCPrimary'].energy
    ptype = phy_frame['MCPrimary'].pdg_encoding
    weight = flux(energy, ptype)/generator(energy, ptype)
    print('Corsika Weight {}'.format(weight))
    phy_frame.Put("corsika_weight", dataclasses.I3Double(weight))
    return


def add_weighted_primary(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    if not 'MCPrimary' in phy_frame.keys():
        get_weighted_primary(phy_frame, MCPrimary='MCPrimary')
    return


flux_conv = NewNuFlux.makeFlux('honda2006')
flux_conv.knee_reweighting_model = "gaisserH4a_elbert"
def atmo_weight(frame):
    if reco_q.is_data(frame):
        return True
    if 'I3MCWeightDict' not in phy_frame:
        return
    conv = frame['I3MCWeightDict']['OneWeight'] * flux_conv.getFlux(frame['I3MCWeightDict']['PrimaryNeutrinoType'],
                                                                    frame['MCPrimary1'].energy,
                                                                    np.cos(frame['MCPrimary1'].dir.zenith))
    frame.Put("conv", dataclasses.I3Double(conv))
    return


def get_stream(phy_frame):
    if (phy_frame['I3EventHeader'].sub_event_stream == 'InIceSplit'):
        return True
    else:
        return False
