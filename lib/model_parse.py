#!/usr/bin/env python
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

import h5py
import numpy as np
import importlib
import os
import sys
from collections import OrderedDict


def prepare_io_shapes(inputs, outputs, exp_file):
    inp_transformations = OrderedDict()
    inp_shapes = OrderedDict()
    out_transformations = OrderedDict()
    out_shapes = OrderedDict()
    # open example file
    inp_file = h5py.File(exp_file, 'r')

    for br in inputs:
        inp_shapes[br] = {}
        inp_transformations[br] = {}
        for var, tr in zip(inputs[br]["variables"],
                           inputs[br]["transformations"]):
            if var in inp_file.keys():
                test_arr = np.array(inp_file[var][0])
            elif var in inp_file['reco_vals'].dtype.names:
                test_arr = np.array(inp_file['reco_vals'][var][0])
            else:
                print('{} does not exists in the input file'.format(var))
            res_shape = np.shape(np.squeeze(tr(test_arr))) if not \
                    isinstance(tr(test_arr), np.float) else None
            print(br,var,res_shape)
            inp_shapes[br][var] = res_shape
            inp_transformations[br][var] = tr
        if len(inputs[br]["variables"]) > 1:
            if res_shape != None:
                inp_shapes[br]["general"] = \
                        res_shape + (len(inputs[br]["variables"]),)
            else: 
                inp_shapes[br]["general"] = (len(inputs[br]["variables"]),)
        else:
            if len(res_shape) >1:
                inp_shapes[br]["general"] = res_shape +  (1,)
            else:
                inp_shapes[br]["general"] = (1,)

    for br in outputs:
        out_shapes[br] = {}
        out_transformations[br] = {}
        for var, tr in zip(outputs[br]["variables"],
                           outputs[br]["transformations"]):
            test_arr = np.array(inp_file['reco_vals'][var][0])
            res_shape = np.shape(np.squeeze(tr(test_arr, inp_file["reco_vals"][:][0])))
            if res_shape == ():
                 res_shape = (1,)
            print(br,var,res_shape)
            out_shapes[br][var] = res_shape
            out_transformations[br][var] = tr
        if len(outputs[br]["variables"]) > 1:
           out_shapes[br]["general"] = \
                res_shape + (len(outputs[br]["variables"]),)
        else:
            if len(res_shape) > 1:
                out_shapes[br]["general"] = res_shape + (1,)
            else:
                out_shapes[br]["general"] = res_shape
    inp_file.close()
    return inp_shapes, inp_transformations, out_shapes, out_transformations

def parse_reference_output(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        mname = os.path.splitext(os.path.basename(cfg_file))[0]
        func_model_def = importlib.import_module(mname)
        sys.path.pop()
    except Exception:
        raise Exception('Import of model.py failed: {}'.format(cfg_file))
    ref_outputs = func_model_def.reference_outputs
    return ref_outputs

def parse_functional_model(cfg_file, exp_file, only_model=False):
    # fancy relative imports..
    sys.path.append(os.path.dirname(cfg_file))
    sys.path.append(os.getcwd()+"/"+os.path.dirname(cfg_file))
    mname = os.path.splitext(os.path.basename(cfg_file))[0]
    func_model_def = importlib.import_module(mname)
    sys.path.pop()
    if only_model:
        return func_model_def
    inputs = func_model_def.inputs
    outputs = func_model_def.outputs
    loss_dict = {}
    if hasattr(func_model_def, 'loss_weights'):
        loss_dict['loss_weights'] = func_model_def.loss_weights
    if hasattr(func_model_def, 'loss_functions'):
        loss_dict['loss'] = func_model_def.loss_functions
    if hasattr(func_model_def, 'mask'):
        mask_func = func_model_def.mask
    else:
        mask_func = None
    in_shapes, in_trans, out_shapes, out_trans = \
        prepare_io_shapes(inputs, outputs, exp_file)
    print('----In  Shapes-----\n {}'.format(in_shapes))
    print('----Out Shapes----- \n {}'.format(out_shapes))
    print('--- Loss Settings ---- \n {}'.format(loss_dict))
    model = func_model_def.model(in_shapes, out_shapes)
    return model, in_shapes, in_trans, out_shapes, out_trans, loss_dict, mask_func


