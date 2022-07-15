from model import *
from data import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=False,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/salmones/train','image','label',data_gen_args,save_to_dir = None,target_size = (200,100))

model = unet(input_size=(200,100,1))
model_checkpoint = ModelCheckpoint('unet_salmones.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=1000,epochs=5,callbacks=[model_checkpoint])
