from icecube import dataclasses, dataio, icetray
from I3Tray import *
import sys
sys.path.append("/Users/theoglauch/Documents/PhD/DeepIceLearning/")
sys.path.append(os.path.join('/Users/theoglauch/Documents/PhD/DeepIceLearning','lib'))
import numpy as np
import argparse
from i3module import DeepLearningClassifier


settings = {
    '[\"I3EventHeader\"].event_id':
    ("event_id", np.float64),
    '[\"I3EventHeader\"].run_id':
    ("run_id", np.float64),
    '[\"Deep_Learning_Classification\"][\"Starting_Track\"]':
    ("starting_score", np.float64),
    '[\"Deep_Learning_Classification\"][\"Cascade\"]':
    ("cascade_score", np.float64),
    '[\"Deep_Learning_Classification\"][\"Through_Going_Track\"]':
    ("tg_score", np.float64),
    '[\"Deep_Learning_Classification\"][\"Stopping_Track\"]':
    ("stopping_score", np.float64),
    '[\"Deep_Learning_Classification\"][\"Skimming\"]':
    ("skimming_score", np.float64),
    '[\"cscdSBU_MonopodFit4_final\"].energy':
    ("monopod_energy", np.float64),
    '[\"cscdSBU_LE_bdt_track\"].value':
    ("bdt_track", np.float64),
    '[\"cscdSBU_LE_bdt_cascade\"].value':
    ("bdt_cascade", np.float64),
    '[\"cscdSBU_LE_bdt_hybrid\"].value':
    ("bdt_hyrbrid", np.float64),
    }

def save_to_array(phy_frame):
    if phy_frame is None:
        print('Physics Frame is None')
        return False
    print('Run ID: {}, Event ID: {}'.format(phy_frame["I3EventHeader"].run_id,
                                            phy_frame["I3EventHeader"].event_id))
    for key in settings.keys():
        try:
            events[settings[key][0]].append(eval('phy_frame' + key))
        except Exception:
            print('{} is NaN'.format(settings[key][0]))
            events[settings[key][0]].append(numpy.nan)
    return True

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        help="files to be processed",
        type=str, nargs="+", required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    events = dict()
    for key in settings.keys():
        events[settings[key][0]] = []
    dtype = [settings[key] for key in settings.keys()]
    args = parseArguments()
    files = []
    for j in np.atleast_1d(args.files):
        if os.path.isdir(j):
            files.extend([os.path.join(j,i) for i in os.listdir(j) if '.i3' in i])
        else:
            files.append(j)
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=np.array(sorted(files)))
    tray.AddModule(DeepLearningClassifier, 'dl_class')
    tray.AddModule(save_to_array, 'write_data_to_array',
                   Streams=[icetray.I3Frame.Physics],)
    tray.Execute()
    tray.Finish()

    for key in events:
        events[key] = np.array(events[key])
    events = np.array(
        zip(*[events[f[0]] for f in dtype]), dtype=dtype)
    np.save('cascades.npy', events)
