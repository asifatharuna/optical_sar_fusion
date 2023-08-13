from osgeo import gdal,osr
import numpy as np                      # scientific computing
import xarray as xr
import dask.array as da
import pandas as pd                     # data analysis and manipulation
import os
from os.path import join                # data access in file manager  
from glob import iglob,glob                 # data access in file manager
import time 
from tqdm import tqdm
from joblib import Parallel , delayed
import gc
###----------------------------------------------------
from utils import nearest_date
#from config import *
from config import s1_api, s1date_range, s2date_range,vv_dir as vv_out,vh_dir as vh_out, nworkers,_dtype,_chunk  #output_directory as out_dir

s1_path_input='./data/sar/sar_download/'

if (os.path.exists(vv_out)):
    pass 
else: os.makedirs(vv_out)
if (os.path.exists(vh_out)):
    pass
else: os.makedirs(vh_out)
s1_input_files= list(sorted(iglob(join(s1_path_input, "S1*.tif"),recursive=True)))



#### since its the files contains both S1a and s1b with different naming convention wrt to my output names 
# we need to resort the files according to dates 
def sor_id(list):
    tmp=os.path.basename(list).split('_')[5][:8]
    return tmp


s1_input_files=sorted(s1_input_files, key=sor_id)


####more general approach
vv_name=os.path.basename(s1_input_files[1]).split('.')[0]+ "_vv.tif" ###output_naming where i will be iterated
vh_name=os.path.basename(s1_input_files[1]).split('.')[0]+ "_vh.tif" ###output_naming where i will be iterated

##needed to update the data input repo when new data is downloaded
#so u can update before ingesting the data
#TO DO make parallel using joblib/ setting the multi processing option ingdaloptions/converting to dask array
def translate_s1_bands(polarization=None):
    assert(type(polarization)==list) , "The polarization has to be a type list with string elements e.g ['vv',''vh']"
    assert(np.isin(polarization,['vv','VV','vh','VH']).all()), 'The polarization  list  elements with value vv and/or vh are only allowed'
    ### for band vv
    if  'vv' in polarization or 'VV' in polarization:
        
        ##### predefined parameter for the gdalTranslate utility 
        translateOptions_VV = gdal.TranslateOptions(format="GTiff",bandList=[1],xRes=10,yRes=10,creationOptions={"COMPRESS:LZW","TILED:YES"})
        t0 = time.time()
        #warpoptions=gdal.WarpOptions(format="GTiff",xRes=10,yRes=10,targetAlignedPixels=True,srcSRS="EPSG:4326" ,dstSRS="EPSG:25832",dstNodata=99999,copyMetadata=True,creationOptions={"COMPRESS:LZW","TILED:YES" })
        for i in tqdm(enumerate (s1_input_files), desc='Translanting band VV', colour='blue'):
            tmp_name= os.path.basename(s1_input_files[i[0]]).split('.')[0]+"_vv.tif"

            tmp_in = s1_input_files[i[0]]                          
            tmp_out = os.path.join(vv_out,tmp_name) 
            if os.path.exists(tmp_out) :
                pass
            else:
                gdal.Translate(tmp_out,tmp_in ,options=translateOptions_VV)

        t1= time.time() 
        print("completed translating vv successfully! in {} seconds".format((t1-t0)))

        ### for band vh
    if  'vh' in polarization or 'VH' in polarization:
       
        translateOptions_VH = gdal.TranslateOptions(format="GTiff",bandList=[2],xRes=10,yRes=10,creationOptions={"COMPRESS:LZW","TILED:YES"},)
        t0 = time.time()
        #warpoptions=gdal.WarpOptions(format="GTiff",xRes=10,yRes=10,targetAlignedPixels=True,srcSRS="EPSG:4326"                   ,dstSRS="EPSG:25832",dstNodata=99999,copyMetadata=True,creationOptions={"COMPRESS:LZW","TILED:YES" })
        for i in tqdm(enumerate (s1_input_files), desc='Translanting band VH', colour='blue'):
            tmp_name= os.path.basename(s1_input_files[i[0]]).split('.')[0]+ "_vh.tif"
            tmp_in = s1_input_files[i[0]]                          
            tmp_out = os.path.join(vh_out,tmp_name)  
            if os.path.exists(tmp_out) :
                pass
            else:
                gdal.Translate(tmp_out,tmp_in ,options=translateOptions_VH)

        t1= time.time() 
        print("completed translating vh successfully! in {} seconds".format((t1-t0)))
   



s1_input_vv= list(sorted(iglob(join(vv_out, "S1*_Gamma*vv.tif"),recursive=True)))
s1_input_vh= list(sorted(iglob(join(vh_out, "S1*_Gamma*vh.tif"),recursive=True)))



#### function to ingest s1a and 1b the data in timeseries 

def ingest_s1(polarization=None, start=None, end=None,auto_detect=False):#, subset=None
    assert(type(polarization)==list), "The polarization has to be a type list with string elements e.g ['vv',''vh']"
    #assert(np.isin(polarization,['VV','VH']).all()), 'The polarization  list elements of value VV and/or VH are only allowed'
    assert(np.isin(polarization,['vv','VV','vh','VH']).all()), 'The polarization  list elements of value vv and/or vh are only allowed'
    
    polarization=[i.upper() for i in polarization]
    
    #subset or not
    sub= s1_api()
    
    s1_pol_inputs=[s1_input_vv,s1_input_vh]
    if  not np.isin(['VV','VH'],polarization).all():
        s1_pol_inputs=[s1_pol_inputs[np.argwhere(np.isin(['VV','VH'],polarization)).ravel()[0]]] ##repo
    

    date_list=sorted([sor_id(i) for i in s1_pol_inputs[0]])
    interval=[None, None]
    subset=[0,len(date_list)]
    
    if not auto_detect:
        if start and end:
            interval=nearest_date(start=start, end=end, datelist=date_list)
            #return the dateformat to plane string format
            interval=[it.replace('-','') for it in interval]
            try:
                subset=[date_list.index(i) for i in interval]
            except Exception as err:
                raise ValueError ("The specified date isn't in the list/repo")
            else:
                date_list=date_list[subset[0]:subset[1]+1]
                start=date_list[0]
                start=start[:4]+"-"+start[4:6]+"-"+start[6:]                  
                end=date_list[-1]
                end=end[:4]+"-"+end[4:6]+"-"+end[6:] 
                               
    if  auto_detect:
        ## automatically detects the time interval corresponding to sentinel2 using its specified dates range (s2date_range) in the config file
        #in event any of requested dates interval(start and end)the auto detected date are not in the data repo, 
        #this function will auto detect the enarest possible date 
        s1_rng_update=nearest_date(start=s2date_range[0],end=s2date_range[1],datelist=date_list, format=True)
        interval=s1_rng_update
        
        #convert the nereast date format to plane string
        interval=[it.replace('-','') for it in interval]
        subset=[date_list.index(it) for it in interval]
        
        date_list=date_list[subset[0]:subset[1]+1]
        start=date_list[0]
        end=date_list[-1]
    
        start=start[:4]+"-"+start[4:6]+"-"+start[6:]
        end=end[:4]+"-"+end[4:6]+"-"+end[6:]
   
    s1_polz=[]
    for order, pol in  enumerate (polarization):
        #print("prepocessing:",pol)
        
        tmp_input=sorted(s1_pol_inputs[order], key=sor_id)
        tmp_input=tmp_input[subset[0]:subset[1]+1]
        #s1_image_list=[]
        #k=1
        ### sorted inputs
        #for i in tqdm(tmp_input,desc=f'Ingesting {pol}', colour='blue'):
        def ingest_s1_parallel(sorted_input=None, sub_response=None,idx=None,polz=None):
            if sub['response'].upper() in ['Y','YES']:
                subset=sub['subset']
                image=gdal.Open(sorted_input).ReadAsArray(*subset) #*subset
            if sub['response'].upper() in ['N','NO']:
                image=gdal.Open(sorted_input).ReadAsArray()
            
            #using the np.array to create an artificial n-dimension array
            image=(np.array([image],dtype=_dtype))
            return image
            #image=None
            #print("image {} for {} has been appended successfully".format(k,pol))
            #k+=1
        try:
            s1_image_list=Parallel(n_jobs=nworkers)(delayed(ingest_s1_parallel)(tmp_input[idx],sub,idx,pol)\
                    for idx in tqdm(range(len(tmp_input)),desc=f'Ingesting {pol}',colour='blue', initial=1))

        except:
            s1_image_list=Parallel(n_jobs=nworkers,backend='threading')(delayed(ingest_s1_parallel)(tmp_input[idx],sub,idx,pol)\
                for idx in tqdm(range(len(tmp_input)),desc=f'Ingesting {pol}',colour='blue', initial=1))
        #stack all images along axis 0== time axis
        s1_image_stack=np.ma.concatenate(s1_image_list)
        mask=s1_image_stack.mask
        ##convert to dask array
        s1_image_stack=da.ma.masked_array(da.from_array(s1_image_stack, chunks=_chunk), mask=mask).astype(_dtype)
        s1_polz.append(s1_image_stack)

    sub.update({"start date":start, "end date":end,"image dimension":s1_image_stack.shape})
    return [sub,date_list, s1_polz]  #sub is the request


#----------------------------------------------------------------------------------------------------------------------------------------
###convert the amplitude from db to a linear unit such as power

def db_to_pwr(image):
    image_pwr=np.power(10.,(image/10.))
    return image_pwr
#----------------------------------------------------------------------------------------------------------------------------------------
###convert  power back to db

def pwr_to_db(image):
    image_db= 10.*np.log10(image)
    return pwr_to_db
#----------------------------------------------------------------------------------------------------------------------------------------


#TO DO make parallel using joblib or pandas
### function to remove extreme values from the time series 
def clip_percentile(data_array, lb=2, ub=98, axis=0):
    gc.collect()
    shp=data_array.shape
    #msk=data_array.mask
    chunks=data_array.chunks
    msk=da.ma.getmaskarray(data_array)
    try:
        temp=da.ma.getdata(data_array) 
    except:
        temp=data_array.copy() 
    try:
        if axis in [1,2]:
            if len(shp) <3:
                def _clip_per(data):

                    array=data.ravel()
                    mask=np.logical_and(array>np.nanpercentile(array,lb),array<np.nanpercentile(array,ub))
                    array=np.where(mask, array, np.nan)
                    #array=np.reshape(array,shp)
                    return array
            else:
                raise ValueError ("its a n-dimensional array consider changing the axis to 0 ")
                    

        if axis==0 and len(shp)==3:
            def _clip_per(data):
                
                temporal=[]
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        array=data[:,i,j].ravel()
                        mask=np.logical_and(array>np.nanpercentile(array,lb),array<np.nanpercentile(array,ub))
                        array=np.where(mask, array, np.nan)
                        array=array.reshape(-1,1)
                        temporal.append(array)
                temporal=np.hstack(([temporal[i] for i in range(len(temporal))]))
                temporal=temporal.reshape(data.shape)
                return temporal
        if  axis==0 and len(shp)<3:
            raise ValueError("it a matrix consider changeing the axis to value of 1 or 2")
    except Exception as err:
        print(err)
        
    else:
        #its faster but can overflood the ram if the data is large
        try:
            temp=temp.map_blocks(_clip_per,dtype=_dtype).compute()
            #temp=np.hstack(([temp[i] for i in range(len(temp))]))
            temp=temp.reshape(shp)
            #cast back to dask array(original imput structure and type)
            temp=da.ma.masked_array(da.from_array(temp, chunks=chunks), mask=msk).astype(_dtype)
        #its slower but prevent out of memory error
        except:
            temp=temp.map_blocks(_clip_per,dtype=_dtype)
            #temp=np.hstack(([temp[i] for i in range(len(temp))]))
            temp=temp.reshape(shp)
            temp=temp.rechunk(chunks)
        gc.collect()
        return temp
      

    
   