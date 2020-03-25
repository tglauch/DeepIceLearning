import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from icecube import dataio, icetray
from I3Tray import *
import argparse
import sys
sys.path.append('/home/tglauch/i3deepice/')
#from i3deepice.i3module import DeepLearningModule
from pulse_modifications import PulseModification
from icecube.hdfwriter import I3HDFTableService
from icecube.tableio import I3TableWriter
import shutil
from functions import add_weighted_primary, corsika_weight, \
                     atmo_weight, get_stream 
dil_path = '/data/user/tglauch/DeepIceLearning'
sys.path.append(dil_path)
sys.path.append(os.path.join(dil_path,'lib'))
import lib.reco_quantities as reco_q
import icecube.MuonGun

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
        "--batch_size", type=int, default=32)
    parser.add_argument(
        "--ncpus", type=int, default=1)
    parser.add_argument(
        "--ngpus", type=int, default=1)
    parser.add_argument(
        "--dataset_config", type=str, default='None')
    parser.add_argument(
        "--gcd", type=str, default='None')    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()

    if args.dataset_config is not 'None':
        from configparser import ConfigParser
        dataset_configparser = ConfigParser()
        dataset_configparser.optionxform = lambda option: option
        try:
            dataset_configparser.read(args.dataset_config)
            print "Config is found {}".format(args.dataset_config)
        except Exception as ex:
            raise Exception('Config File is missing or unreadable!!!!')
            print ex
        args.gcd = str(dataset_configparser.get('Basics', 'geometry_file'))
        args.pulsemap = [str(dataset_configparser.get('Basics', 'PulseSeriesMap'))]
        bpath = str(dataset_configparser.get('Basics', 'out_folder'))
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(args.gcd, padding=0)

    if isinstance(args.pulsemap, str):
        args.pulsemap = [args.pulsemap]
    scratch_bpath = os.getenv('_CONDOR_SCRATCH_DIR')
    if scratch_bpath is None:
        scratch_bpath = '.'
    scratch_bpath = os.path.join(scratch_bpath, 'tmp')
    if not os.path.exists(scratch_bpath):
        os.makedirs(scratch_bpath)
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
    args.outfile = os.path.join(scratch_bpath, fname.split('.')[0] + '.h5')
    dest = os.path.join(bpath, args.files[-1].split('/')[-2])
    if not os.path.exists(dest):
        os.makedirs(dest)
    args.copy_dest = os.path.join(dest,
                                  fname.split('.')[0] + '.h5')
    print('#### Run with Args ####')
    print(args)

    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader',
                   FilenameList=files)
    tray.AddModule(get_stream, "get_stream",
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(add_weighted_primary, "add_primary",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(corsika_weight, 'weighting',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.get_primary_nu, "primary_nu",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.classify_wrapper, "classify",
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
    tray.AddModule(reco_q.get_inelasticity, 'get_inelasticity',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.coincidenceLabel_poly, 'coincidence',
                   Streams=[icetray.I3Frame.Physics])


    # DNN Classification
    '''
    modifications = [('_smear_20ns', 'gaussian_smear_pulse_times', {'scale': 20}),
                     ('_add_white_noise_2', 'add_white_noise', {'noise_rate_factor':0.5, 'time_range': [0, 20000],
                                                               'charge_range': [0.25, 1.25]}),
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
                   benchmark=True)
    tray.AddModule(print_info, 'printer',
                   pulsemap = args.pulsemap,
                   Streams=[icetray.I3Frame.Physics])
    '''

    #Save 
    if os.path.exists(args.outfile):
         os.remove(args.outfile)
    hdf = I3HDFTableService(args.outfile, mode='w+')
    olist = ["TUM_dnn_classification_" + p for p in args.pulsemap]
    olist.extend(["classification", "corsika_weight",
                  "I3MCWeightDict", "QFilterMaks",
                  "I3EventHeader", "track_length",
                  "MCPrimary1", "signature",
                  "conv", "depE", "inelasticity",
                  "IC_hit_doms","BrightDOMs",
                  "PolyplopiaCount", "visible_track",
                  "first_interaction_pos", 'multiplicity',
                  "primary_nu" ])
    print('\n Out Keys: {} \n'.format(olist))
    tray.AddModule(I3TableWriter,'writer',
               tableservice = hdf,
               keys         = olist,
               SubEventStreams=['InIceSplit'],)
    tray.Execute(40) ###### REMOVE ASAP
    tray.Finish()

    #shutil.move(args.outfile,args.copy_dest)
