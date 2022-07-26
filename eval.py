from model import *
from data import *

from modelHoG import main as HoG

import argparse
import os
from pathlib import Path

def eval(opt):
    model = unet()
    model.load_weights(str(str(Path(__file__).parent) + opt.weights))
    archivos = os.listdir(str(opt.data))
    testGene = testGenerator(opt.data)
    results = model.predict_generator(testGene,len(archivos),verbose=1)
    saveResult("results/",results, archivos)

    HoG()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = "data/salmones/test_all",type=str,help='dir to the dataset')
    parser.add_argument('--results', default = "data/salmones/test",type=str,help='dir to the dataset')
    parser.add_argument('--weights', default = "/weights/best.hdf5",type=str,help='File to save the weights of the model')

    opt = parser.parse_args()
    return opt

def main(opt):
	eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
