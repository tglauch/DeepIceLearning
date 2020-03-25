import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from icecube import dataio, icetray
from I3Tray import *
import argparse
import sys
sys.path.append('/home/tglauch/i3deepice/')
from i3deepice.i3module import DeepLearningModule
from pulse_modifications import PulseModification
from icecube.hdfwriter import I3HDFTableService
from icecube.tableio import I3TableWriter
import shutil

def print_info(phy_frame, pulsemap, key="TUM_dnn_classification",):
    print('Run_ID {} Event_ID {}'.format(phy_frame['I3EventHeader'].run_id,
                                         phy_frame['I3EventHeader'].event_id))
    if 'classification_truth' in phy_frame.keys():
        print('Truth:\n{}'.format(phy_frame['classification_truth'].value))
    for p in pulsemap:
        key_all = key+'_'+p
        if key_all in phy_frame.keys():
            print('Prediction ({}) :\n{}'.format(p, phy_frame[key_all]))
        else:
            print('Key {} does not exist in frame'.format(key_all))
    print('\n')
    return

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--pulsemap", type=str, default="SplitInIcePulses", nargs='+')
    parser.add_argument(
        "--batch_size", type=int, default=32)
    parser.add_argument(
        "--ncpus", type=int, default=1)
    parser.add_argument(
        "--ngpus", type=int, default=1)
    parser.add_argument(
        "--remove_daq", action='store_true', default=False)
    parser.add_argument(
        "--model", type=str, default='classification')
    parser.add_argument(
        "--dataset_config", type=str, default='None')
    parser.add_argument(
        "--replace", action='store_true',
        default=False)
    parser.add_argument(
        "--outfile", type=str, default='default')
    parser.add_argument(
        "--muongun", action='store_true', default=False)
    parser.add_argument(
        "--gcd", type=str, default='None')    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()
    if isinstance(args.pulsemap, str):
        args.pulsemap = [args.pulsemap]
    scratch_bpath = os.getenv('_CONDOR_SCRATCH_DIR')
    args.files =  np.atleast_1d(args.files)
    files = []
    files.append(args.gcd)
    if (len(args.files)==1) & ('.txt' in args.files[0]):
         with open(args.files[0], 'r') as ifile:
            add_files= ifile.read().splitlines()
         files = np.concatenate([files, add_files])
    else:
        for j in np.atleast_1d(args.files):
            if os.path.isdir(j):
                files.extend([os.path.join(j, i)
                              for i in os.listdir(j) if '.i3' in i])
            else:
                files.append(j)
    print('Filelist: {}'.format(files))

    fname = args.files[-1].split('/')[-1]
    bpath = '/data/user/tglauch/HESE_DATA/hese_mc_new'
    args.outfile = os.path.join(scratch_bpath, 'tmp', fname.split('.')[0] + '.h5')
    dest = os.path.join(bpath, args.files[-1].split('/')[-2])
    if not os.path.exists(dest):
        os.makedirs(dest)
    args.copy_dest = os.path.join(dest,
                                  fname.split('.')[0] + '.h5')
    args.i3_out = os.path.join(dest,
                               fname.split('.')[0] + '.i3.bz2')
    #if 'MuonGun' in fname:
    args.pulsemap = ['SplitOfflinePulses']
    #args.outfile = args.files[-1].split('.')[0] + '.h5'
    print('#### Run with Args ####')
    print(args)

    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader',
                   FilenameList=files)
    modifications = [('_shift_pulses_20_ns', 'shift_pulses', {'time_shift': -20, 'first_k_pulses':1}),
                     ('_smear_20ns', 'gaussian_smear_pulse_times', {'scale': 20}),
                     ('_smear_5ns', 'gaussian_smear_pulse_times', {'scale': 5}),
                     ('_discard_10_DOMs', 'discard_k_highest_charge_doms', {'k':10}),
                     ('_discard_5_DOMs', 'discard_k_highest_charge_doms', {'k':5}),
                     ('_add_white_noise_2', 'add_white_noise', {'noise_rate_factor':2, 'time_range': [0, 20000],
                                                               'charge_range': [0.25, 1.25]}),
                     ('_add_white_noise_5', 'add_white_noise', {'noise_rate_factor':5, 'time_range': [0, 20000],
                                                               'charge_range': [0.25, 1.25]}),
                     ('_add_white_noise_10', 'add_white_noise', {'noise_rate_factor':10, 'time_range': [0, 20000],
                                                                'charge_range': [0.25, 1.25]})]
    for i, mod in enumerate(modifications):
        out_key = args.pulsemap[0] + mod[0]
        tray.AddModule(PulseModification, 'pulse_mod_{}'.format(i),
                       OutKey=out_key,
                       PulseKey=args.pulsemap[0],
                       Modification=mod[1],
                       ModificationSettings=mod[2])
        args.pulsemap.append(out_key)
    tray.AddModule(DeepLearningModule, "DeepLearningMod",
                   pulsemap=args.pulsemap,
                   batch_size=args.batch_size,
                   add_truth=False,
#                   calib_errata='CalibrationErrata',
#                   bad_dom_list='BadDomsList',
#                   saturation_windows='SaturationWindows',
#                   bright_doms='BrightDOMs',
                   benchmark=True)
    tray.AddModule(print_info, 'printer',
                   pulsemap = args.pulsemap,
                   Streams=[icetray.I3Frame.Physics])
    if os.path.exists(args.outfile):
        os.remove(args.outfile)
    hdf = I3HDFTableService(args.outfile, mode='w+')
    olist = ["TUM_dnn_classification_" + p for p in args.pulsemap]
    olist.extend(["classification",
                  "I3MCWeightDict",
                  "I3EventHeader",
                  "MCPrimary1",
                  "CausalQTot",
                  "HESE_CausalQTot",
                  "OnlineL2_SplineMPE",
                  "RNNReco",
                  "RNNReco_sigma",
                  "conv", "depE",
                  "IC_hit_doms","BrightDOMs",
                  "SaturatedDOMs",
                  "PolyplopiaCount",
                  'MuonWeight', 'MCTrack',
                  "first_interaction_pos"])
    print('\n Out Keys: {} \n'.format(olist))
    tray.AddModule('I3Writer', 'i3writer',
               skipkeys= args.pulsemap,
               Streams  = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
               Filename = args.i3_out)
    tray.AddModule(I3TableWriter,'writer',
               tableservice = hdf,
               keys         = olist,
               SubEventStreams=['InIceSplit', 'baseproctriggersplit'],)
    tray.Execute()
    tray.Finish()
    shutil.move(args.outfile,args.copy_dest)
