from model import *
from data import *
from pathlib import Path

model = unet()
model.load_weights(str(str(Path(__file__).parent) + '/best.hdf5'))
testGene = testGenerator("data/salmones/test",num_image=30)
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/salmones/test",results)
