import csv
import matplotlib.image as mpimg
import os
import numpy as np
from tqdm import tqdm

def ImportImageData(imgdatafile):
    print('Processing : '+imgdatafile)
    dirname = os.path.dirname(imgdatafile)
    imgfolder = 'IMG'
    lines = []
    with open(imgdatafile,'r') as datafile:
        reader = csv.reader(datafile)
        for line in reader:
            lines.append(line)
    
    cimages = []
    limages = []
    rimages = []
    measurements = []
    with tqdm(total=len(lines)) as pbar:
        for line in lines:
            pbar.update(1)
            (center,left,right,sangle,throttle,breakst,speed)=line
            cimage = mpimg.imread(os.path.join(dirname,imgfolder,os.path.basename(center)))
            limage = mpimg.imread(os.path.join(dirname,imgfolder,os.path.basename(left)))
            rimage = mpimg.imread(os.path.join(dirname,imgfolder,os.path.basename(right)))
            cimages.append(cimage)
            limages.append(limage)
            rimages.append(rimage)
            measurements.append(float(sangle))

    return (cimages,rimages,limages,measurements)



# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# (cimages,limage,rimages,sangles)=ImportImageData('..\\..\\Track1\\Center\\driving_log.csv')

# import matplotlib.pyplot as plt
# plt.imshow(cimages[0])
# plt.show()