#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# PyWAD is open-source software and consists of a set of modules written in python for the WAD-Software medical physics quality control software. 
# The WAD Software can be found on https://github.com/wadqc
# 
# The pywad package includes modules for the automated analysis of QC images for various imaging modalities. 
# PyWAD has been originaly initiated by Dennis Dickerscheid (AZN), Arnold Schilham (UMCU), Rob van Rooij (UMCU) and Tim de Wit (AMC) 
#
#
# Changelog:
#   20180802: first version
#
# python CORanalyzer.py -r results.json -c config\config_GE.json -d images\study

from __future__ import print_function

__version__ = '20180802'
__author__ = 'jmgroen'

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

import numpy as np
import scipy
if not 'MPLCONFIGDIR' in os.environ:
    # using a fixed folder is preferable to a tempdir, because tempdirs are not automatically removed
    os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

def logTag():
    return "[CORanalyzer] "

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        import pydicom as dicom
    except ImportError:
        import dicom
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt)
     

def header_series(data, results, action):
    
    # function based on pyWAD function by A. Schilham
    
    # get the first (and only) file
    instances = data.getAllInstances()
    
    if len(instances) != 1:
        print('%s Error! Number of instances not equal to 1 (%d). Exit.'%(logTag(),len(instances)))
    instance=instances[0]
    
    # we need pydicom to read out dicom tags
    try:
        import pydicom as dicom
    except ImportError:
        import dicom
    
    # look in the config file for tags and write them as results, nested tags are supported 2 levels
    for key in action['tags']:
        varname=key
        tag=action['tags'][key]
        if tag.count('/')==0:
            value=instance[dicom.tag.Tag(tag.split(',')[0],tag.split(',')[1])].value
        elif tag.count('/')==1:
            tag1=tag.split('/')[0]
            tag2=tag.split('/')[1]
            value=instance[dicom.tag.Tag(tag1.split(',')[0],tag1.split(',')[1])][0]\
            [dicom.tag.Tag(tag2.split(',')[0],tag2.split(',')[1])].value
        elif tag.count('/')==2:
            tag1=tag.split('/')[0]
            tag2=tag.split('/')[1]
            tag3=tag.split('/')[2]
            value=instance[dicom.tag.Tag(tag1.split(',')[0],tag1.split(',')[1])][0]\
            [dicom.tag.Tag(tag2.split(',')[0],tag2.split(',')[1])][0]\
            [dicom.tag.Tag(tag3.split(',')[0],tag3.split(',')[1])].value
        else:
            # not more then 2 levels...
            value='too many levels'

        # write results
        results.addString(varname, str(value)[:min(len(str(value)),100)])    
    
        
def findpeak(img):

    import pylab as plt    
    
    # we define a mask to limit the number of point sources to 1
    mask=np.zeros((256,256),dtype=int)
    mask[120:136,:]=1
    img=img*mask
    
    # find the indices of the maximum pixel value
    y_max,x_max=plt.unravel_index(img.argmax(), img.shape)
    img_max=img.max()

    # define a threshold
    threshold=0.15*img_max
    
    pixsum=0
    rowsum=0
    colsum=0
    
    # calculate a weighted average around the peak with a threshold
    for i in range(y_max-7,y_max+7):
        for j in range(x_max-7,x_max+7):
            if img[i,j]>threshold:
                pixsum=pixsum+img[i,j]
                rowsum=rowsum+img[i,j]*i
                colsum=colsum+img[i,j]*j
    x_c=rowsum/pixsum
    y_c=colsum/pixsum

    # return the coordinates of the peak
    return y_c,x_c
        
def COR(data, results, action):

    import pylab as plt 

    try:
        params = action['params']
    except KeyError:
        params = {}
    
    # assume that there is 1 file with multiple images
    instances = data.getAllInstances()
    instance=instances[0]
    pixel_data=instance.pixel_array
    pixel_size_x,pixel_size_y=instance.PixelSpacing[0],instance.PixelSpacing[1]
	
    # empty arrays to store the coordinates of the peak
    h1_x_peaks,h1_y_peaks=[],[]
    h2_x_peaks,h2_y_peaks=[],[]
    
    
    #first half of the images is head1
    for i in range(0,int(np.shape(pixel_data)[0]/2)):
        img=pixel_data[i,:,:]
        x,y=findpeak(img)
        h1_x_peaks.append(x)
        h1_y_peaks.append(y)

    # second half of the images is head2
    for i in range(int(np.shape(pixel_data)[0]/2),int(np.shape(pixel_data)[0])):
        img=pixel_data[i,:,:]
        x,y=findpeak(img)
        h2_x_peaks.append(x)
        h2_y_peaks.append(y)
    
    # change to numpy array to allow multiplication
    h1_x_peaks=np.array(h1_x_peaks)
    h1_y_peaks=np.array(h1_y_peaks)   
    h2_x_peaks=np.array(h2_x_peaks)
    h2_y_peaks=np.array(h2_y_peaks) 
    
    #change units to mm instead of pixels
    h1_x=h1_x_peaks*pixel_size_x
    h1_y=h1_y_peaks*pixel_size_y
    h2_x=h2_x_peaks*pixel_size_x
    h2_y=h2_y_peaks*pixel_size_y
    
    # calculate the error to the ideal sine and straight line
    h1_x_err=(h1_x-sinusfit(h1_x))
    h1_y_err=(h1_y-np.mean(h1_y))
    h2_x_err=(h2_x-sinusfit(h2_x))
    h2_y_err=(h2_y-np.mean(h2_y))
   
    # put all graphs in 1 image
    fig, ax = plt.subplots(nrows=2,ncols=2)
        
    plt.subplot(2,4,1)
    plt.plot(h1_y)
    plt.gca().set_title('H1 Y position')
    plt.gca().set_ylabel('mm')
        
    plt.subplot(2,4,2)
    plt.plot(h1_y_err)
    plt.gca().set_title('H1 Y error')

    results.addFloat('H1 Y mean error', round(np.mean((h1_y_err**2)**0.5),5))
    results.addFloat('H1 Y max error', round(np.max((h1_y_err**2)**0.5),5))
    
    plt.subplot(2,4,3)
    plt.plot(h1_x)
    plt.gca().set_title('H1 X position')
    
    plt.subplot(2,4,4)
    plt.plot(h1_x_err)
    plt.gca().set_title('H1 X error')

    results.addFloat('H1 X mean error', round(np.mean((h1_x_err**2)**0.5),5))
    results.addFloat('H1 X max error', round(np.max((h1_x_err**2)**0.5),5))
    
    plt.subplot(2,4,5)
    plt.plot(h2_y)
    plt.gca().set_title('H2 Y position')
    plt.gca().set_ylabel('mm')
    
    plt.subplot(2,4,6)
    plt.plot(h2_y_err)
    plt.gca().set_title('H2 Y error')
    
    results.addFloat('H2 Y mean error', round(np.mean((h2_y_err**2)**0.5),5))
    results.addFloat('H2 Y max error', round(np.max((h2_y_err**2)**0.5),5))
    
    plt.subplot(2,4,7)
    plt.plot(h2_x)
    plt.gca().set_title('H2 X position')
    
    plt.subplot(2,4,8)
    plt.plot(h2_x_err)
    plt.gca().set_title('H2 X error')
    
    results.addFloat('H2 X mean error', round(np.mean((h2_x_err**2)**0.5),5))
    results.addFloat('H2 X max error', round(np.max((h2_x_err**2)**0.5),5))
    
    fn='graphs.png'
    plt.savefig(fn, bbox_inches='tight')
    results.addObject('Graphs', fn)
    
    
def sinusfit(data):

    # function to fit data to a sine function, data fit is returned

    from scipy.optimize import leastsq
    
    # some estimates based on the input
    N=len(data)
    t=np.linspace(0,2*np.pi, N)
    guess_mean=np.mean(data)
    guess_std=3*np.std(data)/(2**0.5)
    guess_phase=0
    
    # first guess 
    data_first_guess = guess_std*np.sin(t+guess_phase)+guess_mean
    
    # optimizing the fit
    optimize_func = lambda x: x[0]*np.sin(t+x[1])+x[2] - data
    est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]
    data_fit = est_std*np.sin(t+est_phase)+est_mean
    
    # return the data fit
    return data_fit
    
    
if __name__ == "__main__":
    #import the pyWAD framework and get some objects
    data, results, config = pyWADinput()

    # look in the config for actions and run them
    for name,action in config['actions'].items():
        if name=='ignore':
            s='s'
        
        # save acquisition time and date as result        
        elif name == 'acqdatetime':
           acqdatetime_series(data, results, action)

        # save whatever tag is requested as result
        elif name == 'header_series':
           header_series(data, results, action)

        # run the COR analysis
        elif name == 'cor_series':
            COR(data, results, action)

    results.write()

    # all done
