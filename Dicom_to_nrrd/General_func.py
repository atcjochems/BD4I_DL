#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 22:40:52 2018

@author: S.Primakov
Initial functions was taken from Py_Rex, https://github.com/zhenweishi/Py-rex
Visualisation progress bar was taken from https://github.com/alexanderkuk/log-progress
"""

from __future__ import print_function

import logging
import os
import radiomics
from radiomics import featureextractor
import pydicom
import numpy as np
from skimage import draw
import SimpleITK as sitk
import re


def log_progress(sequence, every=None, size=None, name='Progress'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
        
    


#class DcmRTstruct_Reader:
def match_ROIid(rtstruct_path,ROI_name):
    mask_vol = Read_RTSTRUCT(rtstruct_path)
    M= mask_vol[0]
    
    for i in range(len(M.StructureSetROISequence)):
        if str(ROI_name)==M.StructureSetROISequence[i].ROIName:
            ROI_number = M.StructureSetROISequence[i].ROINumber
           # print(ROI_number)
            break
    for j in range(len(M.StructureSetROISequence)):
        if (ROI_number==M.ROIContourSequence[j].ReferencedROINumber):
            #print(ROI_number)
            break
    return j

def ROI_match(ROI,rtstruct_path):
    mask_vol=Read_RTSTRUCT(rtstruct_path)
    M=mask_vol[0]
    target = []
    if ROI.lower()=='all':
        target = [M.StructureSetROISequence[x].ROIName for x in range(0,len(M.StructureSetROISequence))]
    else:   
        for i in range(0,len(M.StructureSetROISequence)):
            if ROI.lower()[0]=='!':
                ROI1=ROI[1:]
                if ROI1.lower()== str(M.StructureSetROISequence[i].ROIName).lower():
                    target.append(M.StructureSetROISequence[i].ROIName)       ## only roi with the same name
            
            elif re.search(ROI.lower(),str(M.StructureSetROISequence[i].ROIName).lower()):
                target.append(M.StructureSetROISequence[i].ROIName)
                
             
                
    #if len(target)==0:
        #for j in range(0,len(M.StructureSetROISequence)):
         #   print M.StructureSetROISequence[j].ROIName
            #break
        #print 'Input ROI is: '
      #  ROI_name = raw_input()
        #target.append(ROI_name)
    #print('------------------------------------')
    #print(target)
    #print('------------------------------------')
    return target

def Read_scan(path):
    scan=[]
    skiped_files=[]
    for s in os.listdir(path):
        try:
            temp_file = pydicom.read_file(os.path.join(path, s), force=True)
            temp_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            temp_mod = temp_file.Modality
            scan.append(temp_file)
            if (temp_mod == 'RTSTRUCT') or (temp_mod == 'RTPLAN') or (temp_mod == 'RTDOSE'):
                scan.remove(temp_file)
        except:
            skiped_files.append(s)
                
    try:
        scan.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    except:
        print('Some problems with sorting scans')
        
    #print('skiped files in scans: ',skiped_files)    
    return scan

def Read_RTSTRUCT(path):
    scan = [pydicom.read_file(path)] 
    return scan

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def get_pixels_hu(scans): # Units to Hounsfield Unit (HU) by multiplying rescale slope and add intercept
    #print('Here is ok')
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16) #convert to int16
    # the code below checks if the image has slope and intercept
    # since MRI images often do not provide these
    try:
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
    except AttributeError:
        pass
    else:
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)        
        image += np.int16(intercept)    
    return np.array(image, dtype=np.int16)

def Img_Bimask(img_path,rtstruct_path,ROI_name,npat='',convert=False,exportDir=''): # generating image array and binary mask
    #print('Loading SCANs and RTSTRUCT ......')
    sliceOK=0
    img_vol = Read_scan(img_path)
    mask_vol=Read_RTSTRUCT(rtstruct_path)

    IM=img_vol[0] # Slices usually have the same basic information including slice size, patient position, etc.
    IM_P=get_pixels_hu(img_vol)
    M=mask_vol[0]
    num_slice=len(img_vol)
    #print('Creating binary mask ......')
    mask=np.zeros([num_slice, IM.Rows, IM.Columns],dtype=np.uint8)
    xres=np.array(IM.PixelSpacing[0])
    yres=np.array(IM.PixelSpacing[1])
    zres=np.abs(img_vol[1].ImagePositionPatient[2]-img_vol[0].ImagePositionPatient[2])
    ROI_id = match_ROIid(rtstruct_path,ROI_name)
    #Check DICOM file Modality
    if IM.Modality == 'CT' or 'PT':
        for k in range(len(M.ROIContourSequence[ROI_id].ContourSequence)):    
            Cpostion_rt = M.ROIContourSequence[ROI_id].ContourSequence[k].ContourData[2]
            #print(Cpostion_rt,'Cposition')
            
            for i in range(num_slice):
                if np.abs(Cpostion_rt - img_vol[i].ImagePositionPatient[2])<0.001: # match the binary mask and the corresponding slice
                    sliceOK = i
                    break
            #print('Slice:',sliceOK)
            x=[]
            y=[]
            z=[]
            m=M.ROIContourSequence[ROI_id].ContourSequence[k].ContourData
            for i in range(0,len(m),3):
                x.append(m[i+1])
                y.append(m[i+0])
                z.append(m[i+2])
        
            x=np.array(x)
            y=np.array(y)
            z=np.array(z)
            #print('point-2')
            x-= IM.ImagePositionPatient[1]
            y-= IM.ImagePositionPatient[0]
            z-= IM.ImagePositionPatient[2]
            pts = np.zeros([len(x),3])  
            pts[:,0] = x
            pts[:,1] = y
            pts[:,2] = z
            a=0
            b=1
            p1 = xres
            p2 = yres
            m=np.zeros([2,2])             
            m[0,0]=img_vol[sliceOK].ImageOrientationPatient[a]*p1 
            m[0,1]=img_vol[sliceOK].ImageOrientationPatient[a+3]*p2
            m[1,0]=img_vol[sliceOK].ImageOrientationPatient[b]*p1
            m[1,1]=img_vol[sliceOK].ImageOrientationPatient[b+3]*p2
            #print('point-3') 
            # Transform points from reference frame to image coordinates
            m_inv=np.linalg.inv(m)
            pts = (np.matmul((m_inv),(pts[:,[a,b]]).T)).T
            mask[sliceOK,:,:] = np.logical_or(mask[sliceOK,:,:],poly2mask(pts[:,0],pts[:,1],[IM_P.shape[1],IM_P.shape[2]]))
         
    elif IM.Modality == 'MR':
        slice_0 = img_vol[0]
        slice_n = img_vol[-1]

        # the screen coordinates, including the slice number can then be computed 
        # using the inverse of this matrix
        transform_matrix = np.r_[slice_0.ImageOrientationPatient[3:], 0, slice_0.ImageOrientationPatient[:3], 0, 0, 0, 0, 0, 1, 1, 1, 1].reshape(4, 4).T # yeah that's ugly but I didn't have enough time to make anything nicer
        T_0 = np.array(slice_0.ImagePositionPatient)
        T_n = np.array(slice_n.ImagePositionPatient)
        col_2 = (T_0 - T_n) / (1 - len(img_vol))
        pix_s = slice_0.PixelSpacing
        transform_matrix[:, -1] = np.r_[T_0, 1] 
        transform_matrix[:, 2] = np.r_[col_2, 0] 
        transform_matrix[:, 0] *= pix_s[1]
        transform_matrix[:, 1] *= pix_s[0]
        
        transform_matrix = np.linalg.inv(transform_matrix)
        #print('point-1')
        for s in M.ROIContourSequence[ROI_id].ContourSequence:    
            Cpostion_rt = np.r_[s.ContourData[:3], 1] # the ROI point to get slice number from
                                                      # in homogenous coordinates

            roi_slice_nb = int(transform_matrix.dot(Cpostion_rt)[2]) # the slice number according to the 
                                                                     # inverse transform
            for i in range(num_slice):
                print(roi_slice_nb, i)
                if roi_slice_nb == i:
                    sliceOK = i
                    break
            x=[]
            y=[]
            z=[]                            
            m=s.ContourData
            for i in range(0,len(m),3):
                x.append(m[i+1])
                y.append(m[i+0])
                z.append(m[i+2])
    
            x=np.array(x)
            y=np.array(y)
            z=np.array(z)
            #print('point-2')
            x-= IM.ImagePositionPatient[1]
            y-= IM.ImagePositionPatient[0]
            z-= IM.ImagePositionPatient[2]
            pts = np.zeros([len(x),3])	
            pts[:,0] = x
            pts[:,1] = y
            pts[:,2] = z
            a=0
            b=1
            p1 = xres
            p2 = yres
            m=np.zeros([2,2])             
            m[0,0]=img_vol[sliceOK].ImageOrientationPatient[a]*p1 
            m[0,1]=img_vol[sliceOK].ImageOrientationPatient[a+3]*p2
            m[1,0]=img_vol[sliceOK].ImageOrientationPatient[b]*p1
            m[1,1]=img_vol[sliceOK].ImageOrientationPatient[b+3]*p2
            #print('point-3') 
            # Transform points from reference frame to image coordinates
            m_inv=np.linalg.inv(m)
            pts = (np.matmul((m_inv),(pts[:,[a,b]]).T)).T
            mask[sliceOK,:,:] = np.logical_or(mask[sliceOK,:,:],poly2mask(pts[:,0],pts[:,1],[IM_P.shape[1],IM_P.shape[2]]))

    #IM_P-=np.min(IM_P)
    #IM_P=IM_P.astype(np.float32)
    #IM_P/=np.max(IM_P)
    #IM_P*=255
        
    Img=sitk.GetImageFromArray(IM_P.astype(np.float32)) # convert image_array to image
    Mask=sitk.GetImageFromArray(mask)
    #print('Image spacing',Mask.GetSpacing())
    Img.SetSpacing((float(xres),float(yres),float(zres)))
    Mask.SetSpacing((float(xres),float(yres),float(zres)))
    #print('Image spacing',Img.GetSpacing())
    #print('shape',IM_P.shape)
    if convert:
        if not os.path.exists(exportDir+'/converted_nrrds'):
            os.makedirs(exportDir+'/converted_nrrds')
        if not os.path.exists(exportDir+'/converted_nrrds/'+npat):
            os.makedirs(exportDir+'/converted_nrrds/'+npat)    
                
        expdir=os.path.join(exportDir,'converted_nrrds',npat)
        image_file_name='image.nrrd'
        mask_file_name='mask.nrrd' 
        sitk.WriteImage(Img,os.path.join(expdir,image_file_name)) # save image and binary mask locally
        sitk.WriteImage(Mask,os.path.join(expdir,mask_file_name))
        
    return Img, Mask


def CalculationRun(imageName,maskName,paramsFile,loggenabled):
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
        print('Error getting testcase!')
        exit()
        
    # Get the PyRadiomics logger (default log-level = INFO
    if loggenabled:
        logger = radiomics.logger
        logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
        
        # Write out all log entries to a file
        handler = logging.FileHandler(filename='testLog.txt', mode='w')
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeaturesExtractor(paramsFile)

    # Uncomment one of these functions to show how PyRadiomics can use the 'tqdm' or 'click' package to report progress when
    # running in full python mode. Assumes the respective package is installed (not included in the requirements)
    #print("Calculating features:")
    featureVector = extractor.execute(imageName, maskName)  
    return featureVector


def Parsing_dir(general_path,patient,Patient_dict):
    dcm_files=[]
    flag=False
    for root, dirs, files in os.walk(general_path+'/'+patient):
        for file in files:
            if file.endswith('.dcm'):
                dcm_files.append(os.path.join(root, file))
    structfile =''        
    for file in dcm_files:
        structfile ='' 
        try:
            temp_file = pydicom.read_file(file,force = True)
            if temp_file.Modality == 'RTSTRUCT':
                structfile = file

                #datafile =os.path.dirname(os.path.abspath(file))    #Angela data case
                #Patient_dict[patient]=[datafile,structfile]

                #print('STRUCTURE FILE FOUND: ',structfile)

                datafile=''
                for dfile in dcm_files:
                    temp_dfile = pydicom.read_file(dfile,force = True)
                    if (temp_dfile.StudyInstanceUID == temp_file.StudyInstanceUID) and (temp_dfile.Modality != 'RTSTRUCT') and (temp_dfile.Modality != 'RTPLAN') and (temp_dfile.Modality != 'RTDOSE'):

                        datafile = os.path.dirname(os.path.abspath(dfile)) #getting directory name

                        #print('STUDY FILES FOUND IN DIRECTORY: ',datafile)

                        Patient_dict[patient]=[datafile,structfile]
                        flag=True
                        break
            if flag:
                flag=False
                break
                   
        except:
            print('Some problems occures with this file ',file)
    if structfile=='':
        print('RTSTRUCT for %s patient not found'%patient)
    return Patient_dict
  

def Batch_calculation(Patient_dict,patient,Region_of_interest,convert_to_nrrd,exportDir,paramsFile,logg_enable,Output_dictionary,on):
    Img_path = Patient_dict[patient][0]
    RT_path = Patient_dict[patient][1]

    try:
        target = ROI_match(Region_of_interest,RT_path)
    except:
        print('Error: ROI extraction failed')
        
    PatientID = patient
    
    # Calculation starting
    try:
        for k in range(0,len(target)):
            try:
                if convert_to_nrrd:
                    _,_ = Img_Bimask(Img_path,RT_path,target[k],npat= str(patient+target[k]),exportDir=exportDir,convert=True)
                else:
                    Image,Mask = Img_Bimask(Img_path,RT_path,target[k]) #create image array and binary mask
                    featureVector = CalculationRun(Image,Mask,paramsFile,logg_enable) #compute radiomics
                    if featureVector['general_info_ImageHash'] != '':
                        featureVector.update({'Patient':PatientID,'ROI':target[k]}) #add patient ID and contour
                        Output_dictionary[on] = featureVector   
                        on+=1
            except:
                print ('ROI : ',target[k],'skipped')
    except:
        print('Error: Failed')      
    return Output_dictionary,on