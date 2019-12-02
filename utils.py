"""
This file contains utility functions for iSeeBetter.
Aman Chadha | aman@amanchadha.com
"""

import os
import torch

def loadPreTrainedModel(gpuMode, model, modelPath):
    if os.path.exists(modelPath):
        if gpuMode and torch.cuda.is_available():
            state_dict = torch.load(modelPath)
        else:
            state_dict = torch.load(modelPath, map_location=torch.device('cpu'))

        # Handle the usual (non-DataParallel) case
        try:
            model.load_state_dict(state_dict)

        # Handle DataParallel case
        except:
            # create new OrderedDict that does not contain module.
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[len('module.'):]  # remove module.
                new_state_dict[name] = v

            # load params
            model.load_state_dict(new_state_dict)
        print('Pre-trained SR model loaded from:', modelPath)
    else:
        print('Couldn\'t find pre-trained SR model at:', modelPath)