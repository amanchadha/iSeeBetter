from __future__ import print_function
import argparse

import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_test_set
import numpy as np
import utils
import time
import cv2
import math
import logger

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('-m', '--model', default="weights/netG_epoch_4_1.pth", help="Model")
#parser.add_argument('-m', '--model', default="weights/netG_epoch_4_1.pth", help="Model")
#parser.add_argument('-m', '--model', default="weights/RBPN_4x.pth", help="Model")
parser.add_argument('-o', '--output', default='Results/', help="Location to save test results")
parser.add_argument('-s', '--upscale_factor', type=int, default=4, help="Super-Resolution Scale Factor")
parser.add_argument('-r', '--residual', action='store_true', required=False, help="")
parser.add_argument('-c', '--gpu_mode', action='store_true', required=False, help="Use a CUDA compatible GPU if available")
parser.add_argument('--testBatchSize', type=int, default=1, help="Testing Batch Size")
parser.add_argument('--chop_forward', action='store_true', required=False, help="")
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")
parser.add_argument('--seed', type=int, default=123, help="Random seed")
parser.add_argument('--gpus', default=1, type=int, help="How many GPU's to use")
parser.add_argument('--data_dir', type=str, default="./Vid4")
parser.add_argument('--file_list', type=str, default="foliage_test.txt")
parser.add_argument('--other_dataset', type=bool, default=True, help="Use a dataset that isn't vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="Use future frame")
parser.add_argument('--nFrames', type=int, default=7, help="")
parser.add_argument('--model_type', type=str, default="RBPN", help="")
parser.add_argument('-d', '--debug', action='store_true', required=False, help="Print debug spew.")
parser.add_argument('-u', '--upscale_only', action='store_true', required=False, help="Upscale mode - without downscaling.")

args = parser.parse_args()

gpus_list=range(args.gpus)
print(args)

cuda = args.gpu_mode
if cuda:
    print("Using GPU mode")
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode")

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

print('==> Loading datasets')
test_set = get_test_set(args.data_dir, args.nFrames, args.upscale_factor, args.file_list, args.other_dataset, args.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)

print('==> Building model ', args.model_type)
if args.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=args.nFrames, scale_factor=args.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    # Initialize Logger
    logger.initLogger(args.debug)

    # print iSeeBetter architecture
    utils.printNetworkArch(netG=model, netD=None)

    # load model
    modelPath = os.path.join(args.model)
    utils.loadPreTrainedModel(gpuMode=args.gpu_mode, model=model, modelPath=modelPath)

    model.eval()
    count = 0
    upscale_only = args.upscale_only
    if not upscale_only:
        avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        
        with torch.no_grad():
            if cuda:
                input = Variable(input).cuda(gpus_list[0])
                bicubic = Variable(bicubic).cuda(gpus_list[0])
                neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
                flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]
            else:
                input = Variable(input).to(device=device, dtype=torch.float)
                bicubic = Variable(bicubic).to(device=device, dtype=torch.float)
                neigbor = [Variable(j).to(device=device, dtype=torch.float) for j in neigbor]
                flow = [Variable(j).to(device=device, dtype=torch.float) for j in flow]

        t0 = time.time()
        if args.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, flow, model, args.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor, flow)
        
        if args.residual:
            prediction = prediction + bicubic
            
        t1 = time.time()
        print("==> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        save_img(prediction.cpu().data, str(count), True)
        save_img(target, str(count), False)
        
        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction*255.
        
        target = target.squeeze().numpy().astype(np.float32)
        target = target*255.
        if not upscale_only:
            psnr_predicted = PSNR(prediction, target, shave_border=args.upscale_factor)
            print("PSNR Predicted = ", psnr_predicted)
            avg_psnr_predicted += psnr_predicted
        count += 1
        
    if not upscale_only:  # Otherwise the print will error on '-u'
        print("Avg PSNR Predicted = ", avg_psnr_predicted/count)

def save_img(img, img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    save_dir=os.path.join(args.output, args.data_dir, os.path.splitext(args.file_list)[0]+'_'+str(args.upscale_factor)+'x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+args.model_type+'F'+str(args.nFrames)+'.png'
    else:
        save_fn = save_dir +'/'+ img_name+'.png'
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[1:3]
    pred = pred[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def chop_forward(x, neigbor, flow, model, scale, shave=8, min_size=2000, nGPUs=args.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neigbor], [j[:, :, 0:h_size, 0:w_size] for j in flow]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neigbor], [j[:, :, 0:h_size, (w - w_size):w] for j in flow]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neigbor], [j[:, :, (h - h_size):h, 0:w_size] for j in flow]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor], [j[:, :, (h - h_size):h, (w - w_size):w] for j in flow]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch[0], input_batch[1], input_batch[2])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval()
