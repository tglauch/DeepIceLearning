# coding: utf-8

import sys
import os
dirname = os.path.dirname(__file__)
sys.path.insert(0, '/home/tglauch/virtualenvs/tf_env3/lib/python2.7/site-packages')
sys.path.insert(0, os.path.join(dirname, 'lib/'))
from model_parser import parse_functional_model
from helpers import *
import numpy as np
from icecube import icetray
from I3Tray import I3Tray
from configparser import ConfigParser
from collections import OrderedDict
import time
from icecube.dataclasses import I3MapStringDouble
from icecube import dataclasses, dataio
import argparse
from plotting import figsize, make_plot, plot_prediction


class DeepLearningClassifier(icetray.I3ConditionalModule):
    """IceTray compatible class of the  DeepLearning Classifier
    """

    def __init__(self,context):
        """Initialize the Class
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("pulsemap","Define the name of the pulsemap",
                          "InIceDSTPulses")
        self.AddParameter("save_as", "Define the Output key",
                          "Deep_Learning_Classification")
        self.AddParameter("batch_size", "Size of the batches", 40)
        print('Init Deep Learning Classifier..this may take a while')

    def Configure(self):
        """Read the network architecture and input, output information from config files
        """
        self._runinfo = np.load(os.path.join(dirname,'cfg/run_info.npy'), allow_pickle=True)[()]
        self._grid = np.load(os.path.join(dirname, 'cfg/grid.npy'), allow_pickle=True)[()]
        self._inp_shapes = self._runinfo['inp_shapes']
        self._out_shapes = self._runinfo['out_shapes']
        self._inp_trans = self._runinfo['inp_trans']
        self._out_trans = self._runinfo['out_trans']
        self._pulsemap = self.GetParameter("pulsemap") 
        self._save_as =  self.GetParameter("save_as")
        self.__batch_size =  self.GetParameter("batch_size")
        self.__frame_buffer = []
        print("Im configured with {} and {}".format(self._pulsemap,self._save_as))
        import cfg.model as func_model_def
        self._model = func_model_def.model(self._inp_shapes, self._out_shapes)
        self._model.load_weights(os.path.join(dirname, 'cfg/weights.npy'))
        dataset_configparser = ConfigParser()
        dataset_configparser.read(os.path.join(dirname,'cfg/config.cfg'))
        inp_defs = dict()
        for key in dataset_configparser['Input_Times']:
            inp_defs[key] = dataset_configparser['Input_Times'][key]
        for key in dataset_configparser['Input_Charges']:
            inp_defs[key] = dataset_configparser['Input_Charges'][key]
        self._inputs = []
        for key in self._inp_shapes.keys():
            binput = []
            branch = self._inp_shapes[key]
            for bkey in branch.keys():
                if bkey == 'general':
                    continue
                elif 'charge_quantile' in bkey:
                    feature = 'pulses_quantiles(charges, times, {})'.format(float('0.' + bkey.split('_')[3]))
                else:
                    feature = inp_defs[bkey.replace('IC_','')]
                trans = self._inp_trans[key][bkey]
                binput.append((feature, trans))
            self._inputs.append(binput)

    def BatchProcessBuffer(self, frames):
        """Batch Process a list of frames. This includes pre-processing, prediction and storage of the results  
        """        
        timer_t0 = time.time()
        f_slices = []
        for frame in frames:
            pulse_key = self._pulsemap
            if pulse_key not in frame.keys():
                print('No Pulsemap called {}'.format(pulse_key))
                return
            f_slice = []
            t0 = get_t0(frame, puls_key=pulse_key)
            pulses = frame[pulse_key].apply(frame)
            for key in self._inp_shapes.keys():
                f_slice.append(np.zeros(self._inp_shapes[key]['general']))
            for omkey in pulses.keys():
                dom = (omkey.string, omkey.om)
                if not dom in self._grid.keys():
                    continue
                gpos = self._grid[dom]
                charges = np.array([p.charge for p in pulses[omkey][:]])
                times = np.array([p.time for p in pulses[omkey][:]]) - t0
                widths = np.array([p.width for p in pulses[omkey][:]])
                for branch_c, inp_branch in enumerate(self._inputs):
                    for inp_c, inp in enumerate(inp_branch):
                        f_slice[branch_c][gpos[0]][gpos[1]][gpos[2]][inp_c] = inp[1](eval(inp[0]))
            processing_time = time.time() - timer_t0
            f_slices.append(f_slice)
        prediction = self._model.predict(np.squeeze(f_slices, axis=1), batch_size=self.__batch_size,
                                         verbose=0, steps=None)
        prediction_time = time.time() - processing_time - timer_t0
        for i, frame in enumerate(frames):
            output = I3MapStringDouble()
            output['Skimming'] = float(prediction[i][0])
            output['Cascade'] = float(prediction[i][1])
            output['Through_Going_Track'] = float(prediction[i][2])
            output['Starting_Track'] = float(prediction[i][3])
            output['Stopping_Track'] = float(prediction[i][4])
            frame.Put(self._save_as, output)
        tot_time = time.time() - timer_t0
        print('Total Time {:.2f}s [{:.2f}s], Processing Time {:.2f}s [{:.2f}s], Prediction Time {:.2f}s [{:.2f}s]'.format(
                tot_time, tot_time/len(frames), processing_time, processing_time/len(frames),
                prediction_time, prediction_time/len(frames)))

    def Physics(self, frame):
        """ Buffer physics frames until batch size is reached, then start processing  
        """
        self.__frame_buffer.append(frame)
        if len(self.__frame_buffer) == self.__batch_size:
            self.BatchProcessBuffer(self.__frame_buffer)
            for frame in self.__frame_buffer:
                self.PushFrame(frame)
            self.__frame_buffer[:] = []

    def DAQ(self,frame):
        #This runs on Q-Frames
       # if self.__frame_buffer:
       #     self.BatchProcessBuffer(self.__frame_buffer)

        #    for frame in self.__frame_buffer:
         #       self.PushFrame(frame)
          #  self.__frame_buffer = []
            #self.__frame_buffer.clear() # Python3 feature.
        self.PushFrame(frame) # don't forget to push the Q frame

    def Finish(self):
        """ Process the remaining (incomplete) batch of frames  
        """
        self.BatchProcessBuffer(self.__frame_buffer)
        for frame in self.__frame_buffer:
            self.PushFrame(frame)
        self.__frame_buffer[:] = []


def print_info(phy_frame):
    print('Run_ID {} Event_ID {}'.format(phy_frame['I3EventHeader'].run_id,
                                         phy_frame['I3EventHeader'].event_id))
    print(phy_frame["Deep_Learning_Classification"])
    return


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--plot", action="store_true",
        default=False)
    parser.add_argument(
        "--pulsemap", type=str,
        default="InIceDSTPulses")
    parser.add_argument(
        "--batch_size", type=int,
        default=40)    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()
    files = []
    for j in np.atleast_1d(args.files):
        if os.path.isdir(j):
            files.extend([os.path.join(j,i) for i in os.listdir(j) if '.i3' in i])
        else:
            files.append(j)
    files = sorted(files)
    tray = I3Tray()
    tray.AddModule('I3Reader','reader',
                   FilenameList = files)
    tray.AddModule(DeepLearningClassifier, "DeepLearningClassifier",
                   pulsemap=args.pulsemap, batch_size=args.batch_size)
    tray.AddModule(print_info, 'printer',
                   Streams=[icetray.I3Frame.Physics])
    if args.plot:
        tray.AddModule(make_plot, 'plotter',
                       Streams=[icetray.I3Frame.Physics])
    tray.Execute()
    tray.Finish()
