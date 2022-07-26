from model import *
from data import *

from modelHoG import main as HoG

import PIL
import tensorflow as tf
import argparse
import os
from pathlib import Path

def eval(opt):
    model = unet()
    model.load_weights(str(str(Path(__file__).parent) + opt.weights))
    archivos = os.listdir(str(opt.data))
    testGene = testGenerator(opt.data)
    results = model.predict_generator(testGene,len(archivos),verbose=1)
    saveResult(opt.results,results, archivos)

    iou_promedio = 0
    for arch in archivos:
        mask_original = cv.imread( str(Path(__file__).parent) + '/'+str(opt.mask+'/'+arch),0)
        mask_result = cv.imread( str(Path(__file__).parent) +'/'+str(opt.results+'/'+arch),0)
        iou = mask_iou(mask_original,mask_result)
        #print("iou =", iou)
        iou_promedio += iou
    iou_promedio = iou_promedio / len(archivos)
    print("iou promedio = ",iou_promedio)

    HoG()

def mask_iou(gt, dt):
    intersection = ((gt * dt) > 0).sum()
    union = ((gt + dt) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = "data/salmones/test_all",type=str,help='dir to the dataset')
    parser.add_argument('--mask', default = "data/salmones/test_mask",type=str,help='dir to the dataset')
    parser.add_argument('--results', default = "results/",type=str,help='dir to the dataset')
    parser.add_argument('--weights', default = "/weights/best.hdf5",type=str,help='File to save the weights of the model')

    opt = parser.parse_args()
    return opt

def main(opt):
	eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
