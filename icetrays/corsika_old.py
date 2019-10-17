from icecube import dataio, icetray
from icecube import dataclasses, paraboloid, simclasses, recclasses, spline_reco
from I3Tray import *
import sys
sys.path.append("/data/user/tglauch/DeepIceLearning/")
sys.path.append(os.path.join('/data/user/tglauch/DeepIceLearning/','lib'))
import reco_quantities as reco_q
import numpy as np
from icecube.weighting import weighting, get_weighted_primary
import argparse
import icecube.MuonGun
from i3module import DeepLearningClassifier

def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

    return True


def print_info(phy_frame):
    tg= phy_frame["Deep_Learning_Classification"]['Through_Going_Track']
    skim = phy_frame["Deep_Learning_Classification"]["Skimming"]
    if True: #(phy_frame['classification'].value == 2) & (skim > tg):
        print('run_id {} ev_id {} dep_E {:.2f} classification {}  signature {} track_length {:.2f}'.format(
              phy_frame['I3EventHeader'].run_id, phy_frame['I3EventHeader'].event_id,
              phy_frame['depE'].value, phy_frame['classification'].value,
              phy_frame['signature'].value, phy_frame['track_length'].value))
        print(phy_frame["Deep_Learning_Classification"])
    return


def add_weighted_primary(phy_frame):
    if not 'MCPrimary' in phy_frame.keys():
        get_weighted_primary(phy_frame)
    return


def get_stream(phy_frame):
    if (phy_frame['I3EventHeader'].sub_event_stream == 'InIceSplit') & (phy_frame['I3EventHeader'].sub_event_id==0):
        return True
    else:
        return False


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        help="files to be processed",
        type=str, nargs="+", required=False)
    parser.add_argument(
        "--geo",
        help='IceCube geometry',
        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(args.geo, padding=0)
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=args.files)
    tray.AddModule(get_stream, "get_stream",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(add_weighted_primary, "add_primary",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.classify_wrapper, "classifiy_corsika",
                   surface=surface,
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.set_signature, "signature",
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.first_interaction_point, "v_point",
                   surface=surface,
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_depositedE, 'depo_energy',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.track_length_in_detector, 'track_length',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_hitDOMs, 'hitDOMs',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(DeepLearningClassifier, 'dl_class')
    tray.AddModule(print_info, 'pinfo',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(cuts, 'cuts',
                   Streams=[icetray.I3Frame.Physics])

    tray.Execute()
    tray.Finish()

