from model import *
from data import *

import argparse
import os
from pathlib import Path

def train(opt):
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=False,
                        fill_mode='nearest')

    myGene = trainGenerator(2,opt.data,'image','label',data_gen_args,save_to_dir = None)

    model = unet()
    model_checkpoint = ModelCheckpoint(opt.save, monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=opt.step_per_epoch,epochs=opt.epochs,callbacks=[model_checkpoint])

    model.summary()
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 100 ,type=int,help='epoch to train')
    parser.add_argument('--step_per_epoch', default = 300 ,type=int,help='epoch to train')
    parser.add_argument('--data', default = "data/salmones/train",type=str,help='dir to the dataset')
    parser.add_argument('--save', default = "weights/best.hdf5",type=str,help='File to save the weights of the model')

    opt = parser.parse_args()
    return opt

def main(opt):
	train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)