'''
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import torch
import argparse
import os
from torch.utils.data import DataLoader
from edanet import EDANet
import numpy as np
import matplotlib.pyplot as plt
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from dataset import test
from transform import Relabel, ToLabel, Colorize

from miou.iou import IoU
import time

input_transform = Compose([
    Resize((480,480)),
    ToTensor(),
])
target_transform = Compose([
    Resize((480,480)),
    ToLabel(),
])

def eval(model,dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        total_miou=[]
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
 
            t1=time.time()
            predict = model(data)
            print 'time:',time.time()-t1,'\n'
            predict = predict.squeeze()
            predict = reverse_one_hot(predict).unsqueeze(0)
            label = label.squeeze().unsqueeze(0)

            metric = IoU(num_classes=2, ignore_index=None)
            metric.reset()

            metric.add(predict.data, label.data)
            iou, miou = metric.value()


            precision = compute_global_accuracy(predict, label)
            print ('precision: %.3f' %precision,'mIOU: %.3f'  %miou)
           # predict = predict.cpu().data.numpy()
           # data = data.cpu().data.numpy()
           # label = label.cpu().data.numpy()
           # fig,ax=plt.subplots(1,2)
            #ax[0].imshow(data)
           # ax[0].imshow(predict)
           # ax[1].imshow(label)
           # plt.show()
            total_miou.append(miou)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        total_miou = np.mean(total_miou)
        #tq.close()
        print('precision for test: %.3f' % precision)
        print('total mIOU for test: %.3f' % total_miou)
        return precision,total_miou

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    args = parser.parse_args(params)

    # create dataset and dataloader
 
    dataloader = DataLoader(test(input_transform, target_transform),num_workers=1, batch_size=1, shuffle=True)
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = EDANet()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    eval(model, dataloader, args)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './save/model/main-edanet_my-step-998-epoch-100.pth',
        '--data', '/path/to/CamVid',
        '--cuda', '0'
    ]
    main(params)
