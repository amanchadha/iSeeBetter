import argparse
import gc
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from data import get_training_set
import logger
from rbpn import Net as RBPN
from rbpn import GeneratorLoss
from SRGAN.model import Discriminator
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils

################################################## iSEEBETTER TRAINER KNOBS ############################################
UPSCALE_FACTOR = 4
########################################################################################################################

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train iSeeBetter: Super Resolution Models')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='RBPN_4x.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--APITLoss', action='store_true', help='Use APIT Loss')
parser.add_argument('--useDataParallel', action='store_true', help='Use DataParallel')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')

def trainModel(epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, args):
    trainBar = tqdm(training_data_loader)
    runningResults = {'batchSize': 0, 'DLoss': 0, 'GLoss': 0, 'DScore': 0, 'GScore': 0}

    netG.train()
    netD.train()

    # Skip first iteration
    iterTrainBar = iter(trainBar)
    next(iterTrainBar)

    for data in iterTrainBar:
        batchSize = len(data)
        runningResults['batchSize'] += batchSize

        ################################################################################################################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ################################################################################################################
        if args.APITLoss:
            fakeHRs = []
            targets = []
        fakeScrs = []
        realScrs = []

        DLoss = 0

        # Zero-out gradients, i.e., start afresh
        netD.zero_grad()

        input, target, neigbor, flow, bicubic = data[0], data[1], data[2], data[3], data[4]
        if args.gpu_mode and torch.cuda.is_available():
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            bicubic = Variable(bicubic).cuda()
            neigbor = [Variable(j).cuda() for j in neigbor]
            flow = [Variable(j).cuda().float() for j in flow]
        else:
            input = Variable(input).to(device=device, dtype=torch.float)
            target = Variable(target).to(device=device, dtype=torch.float)
            bicubic = Variable(bicubic).to(device=device, dtype=torch.float)
            neigbor = [Variable(j).to(device=device, dtype=torch.float) for j in neigbor]
            flow = [Variable(j).to(device=device, dtype=torch.float) for j in flow]

        fakeHR = netG(input, neigbor, flow)
        if args.residual:
            fakeHR = fakeHR + bicubic

        realOut = netD(target).mean()
        fakeOut = netD(fakeHR).mean()

        if args.APITLoss:
            fakeHRs.append(fakeHR)
            targets.append(target)
        fakeScrs.append(fakeOut)
        realScrs.append(realOut)

        DLoss += 1 - realOut + fakeOut

        DLoss /= len(data)

        # Calculate gradients
        DLoss.backward(retain_graph=True)

        # Update weights
        optimizerD.step()

        ################################################################################################################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ################################################################################################################
        GLoss = 0

        # Zero-out gradients, i.e., start afresh
        netG.zero_grad()

        if args.APITLoss:
            idx = 0
            for fakeHR, fake_scr, HRImg in zip(fakeHRs, fakeScrs, targets):
                fakeHR = fakeHR.to(device)
                fake_scr = fake_scr.to(device)
                HRImg = HRImg.to(device)
                GLoss += generatorCriterion(fake_scr, fakeHR, HRImg, idx)
                idx += 1
        else:
            GLoss = generatorCriterion(fakeHR, target)

        GLoss /= len(data)

        # Calculate gradients
        GLoss.backward()

        # Update weights
        optimizerG.step()

        realOut = torch.Tensor(realScrs).mean()
        fakeOut = torch.Tensor(fakeScrs).mean()
        runningResults['GLoss'] += GLoss.item() * args.batchSize
        runningResults['DLoss'] += DLoss.item() * args.batchSize
        runningResults['DScore'] += realOut.item() * args.batchSize
        runningResults['GScore'] += fakeOut.item() * args.batchSize

        trainBar.set_description(desc='[Epoch: %d/%d] D Loss: %.4f G Loss: %.4f D(x): %.4f D(G(z)): %.4f' %
                                       (epoch, args.nEpochs, runningResults['DLoss'] / runningResults['batchSize'],
                                       runningResults['GLoss'] / runningResults['batchSize'],
                                       runningResults['DScore'] / runningResults['batchSize'],
                                       runningResults['GScore'] / runningResults['batchSize']))
        gc.collect()

    netG.eval()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (args.nEpochs / 2) == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 10.0
        logger.info('Learning rate decay: lr=%s', (optimizerG.param_groups[0]['lr']))

    return runningResults

def saveModelParams(epoch, runningResults, netG, netD):
    results = {'DLoss': [], 'GLoss': [], 'DScore': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

    # Save model parameters
    torch.save(netG.state_dict(), 'weights/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'weights/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

    logger.info("Checkpoint saved to {}".format('weights/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
    logger.info("Checkpoint saved to {}".format('weights/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))

    # Save Loss\Scores\PSNR\SSIM
    results['DLoss'].append(runningResults['DLoss'] / runningResults['batchSize'])
    results['GLoss'].append(runningResults['GLoss'] / runningResults['batchSize'])
    results['DScore'].append(runningResults['DScore'] / runningResults['batchSize'])
    results['GScore'].append(runningResults['GScore'] / runningResults['batchSize'])
    #results['PSNR'].append(validationResults['PSNR'])
    #results['SSIM'].append(validationResults['SSIM'])

    if epoch % 1 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(data={'DLoss': results['DLoss'], 'GLoss': results['GLoss'], 'DScore': results['DScore'],
                                  'GScore': results['GScore']},#, 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'iSeeBetter_' + str(UPSCALE_FACTOR) + '_Train_Results.csv', index_label='Epoch')

def main():
    """ Lets begin the training process! """

    args = parser.parse_args()

    # Initialize Logger
    logger.initLogger(args.debug)

    # Load dataset
    logger.info('==> Loading datasets')
    train_set = get_training_set(args.data_dir, args.nFrames, args.upscale_factor, args.data_augmentation,
                                 args.file_list,
                                 args.other_dataset, args.patch_size, args.future_frame)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize,
                                      shuffle=True)

    # Use generator as RBPN
    netG = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=args.nFrames,
                scale_factor=args.upscale_factor)
    logger.info('# of Generator parameters: %s', sum(param.numel() for param in netG.parameters()))

    # Use DataParallel?
    if args.useDataParallel:
        gpus_list = range(args.gpus)
        netG = torch.nn.DataParallel(netG, device_ids=gpus_list)

    # Use discriminator from SRGAN
    netD = Discriminator()
    logger.info('# of Discriminator parameters: %s', sum(param.numel() for param in netD.parameters()))

    # Generator loss
    generatorCriterion = nn.L1Loss() if not args.APITLoss else GeneratorLoss()

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_mode else "cpu")

    if args.gpu_mode and torch.cuda.is_available():
        utils.printCUDAStats()

        netG.cuda()
        netD.cuda()

        netG.to(device)
        netD.to(device)

        generatorCriterion.cuda()

    # Use Adam optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.APITLoss:
        logger.info("Generator Loss: Adversarial Loss + Perception Loss + Image Loss + TV Loss")
    else:
        logger.info("Generator Loss: L1 Loss")

    # print iSeeBetter architecture
    utils.printNetworkArch(netG, netD)

    if args.pretrained:
        modelPath = os.path.join(args.save_folder + args.pretrained_sr)
        utils.loadPreTrainedModel(gpuMode=args.gpu_mode, model=netG, modelPath=modelPath)

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        runningResults = trainModel(epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, args)

        if (epoch + 1) % (args.snapshots) == 0:
            saveModelParams(epoch, runningResults, netG, netD)

if __name__ == "__main__":
    main()
