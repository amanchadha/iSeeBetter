"""
This file contains utility functions for iSeeBetter.
Aman Chadha | aman@amanchadha.com
"""

import os
import torch
import logger

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
                name = "module." + k if not k.startswith("module.") else k  # remove module.
                new_state_dict[name] = v

            # load params
            model.load_state_dict(new_state_dict)
        print('Pre-trained SR model loaded from:', modelPath)
    else:
        print('Couldn\'t find pre-trained SR model at:', modelPath)

def printCUDAStats():
    logger.info("# of CUDA devices detected: %s", torch.cuda.device_count())
    logger.info("Using CUDA device #: %s", torch.cuda.current_device())
    logger.info("CUDA device name: %s", torch.cuda.get_device_name(torch.cuda.current_device()))

def _printNetworkArch(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(net)
    logger.info('Total number of parameters: %d' % num_params)

def printNetworkArch(netG, netD):
    logger.info('------------- iSeeBetter Network Architecture -------------')
    if netG:
        logger.info('----------------- Generator Architecture ------------------')
        _printNetworkArch(netG)

    if netD:
        logger.info('--------------- Discriminator Architecture ----------------')
        _printNetworkArch(netD)
        logger.info('-----------------------------------------------------------')