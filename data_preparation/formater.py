import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path


#names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
#for name in names:
savepath = str( str(Path(__file__).parent) + '/dots')
loadPath = str( str(Path(__file__).parent) + '/originals')

archivos = os.listdir(str(loadPath))
for arch in archivos:
    img = cv.imread(str(loadPath+'/'+arch),0)
    img = cv.resize(img,(256,256))
    img = cv.medianBlur(img,5)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,4)
    result = cv.bitwise_not(th2)
    try:
        cv.imwrite( os.path.join(savepath , str(savepath+'/'+arch)),result )
    except OSError as error:
        print(error)
        # titles = ['Original Image', 'Adaptive Thresholding']
        # images = [org, th2]
        # for i in range(2):
        #     plt.subplot(2,1,i+1),plt.imshow(images[i],'gray')
        #     plt.title(titles[i])
        #     plt.xticks([]),plt.yticks([])
        # plt.show()
        # exit(1)
