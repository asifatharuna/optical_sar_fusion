from osgeo import gdal,osr
import numpy as np                      # scientific computing
import dask.array as da
import xarray as xr
import pandas as pd                     # data analysis and manipulation
import os,sys
from os.path import join                # data access in file manager  
from glob import iglob,glob                 # data access in file manager
import time 
from tqdm import tqdm
from joblib import Parallel , delayed
###-------------------------------------------------------------------------
from utils import nonpixel_count,image_sieve,time_weighted_interpn, update #to update nestesd dictionary
from utils import mseries_time_weighted_interpn
from config import *
from config import vi_dir ,s2_api,output_directory as out_dir, s1date_range,s2date_range,nworkers,_chunk,_dtype,_dtype_int
##---------------------------------------------------------------------------

cwd=os.getcwd()
#os.chdir(vi_dir)
s2_path_input='./data/optical/optical_download/'
s2_out= './data/optical/vi/'


input_files=list(sorted(iglob(join(s2_path_input,"*.tif"))))
##svi=["NDVI","NDRE2","NBR","TCW","TCB","TCG"]


##need to update the data input repo when new data is downloaded
def translate_s2_bands(indices=None):
    assert(type(indices)==list) , "The indices has to be a type list with string elements e.g ['vv',''vh']"
    assert(np.isin(indices,svi).all()), f'The indices supported are {svi}. Please check svi in the config file'
  
    
    #t0 = time.time()
    band=0
    for i in indices:
        band+=1
        translate_option=gdal.TranslateOptions(format="GTiff",bandList=[band],xRes=10,yRes=10,creationOptions={"COMPRESS:LZW","TILED:YES"},noData=-99999)
        
        for j in tqdm(input_files,colour='blue',desc='Translating band: ' + f'{i}'):
            #tmp_name= j.split("/")[-1][0:8]+"_{}.tif".format(i)
            tmp_name= os.path.basename(j)[0:8]+"_{}.tif".format(i)
            tmp_in = j 
            ###export the vi bands into different out subdirectories 
            sub_dir= s2_out+"/{}".format(i)
            ###check if directory exist to prevent duplication or overwriting 
            if (os.path.exists(sub_dir)):
                pass
            else:
                os.makedirs(sub_dir)
            tmp_out = os.path.join(sub_dir,tmp_name) 
            if os.path.exists(tmp_out) :
                pass
            else:
                data=gdal.Translate(tmp_out,tmp_in ,options=translate_option)
                #data.FlushCache()
                data= None 
    #t1= time.time() 
    #print("completed successfully! in {} seconds".format((t1-t0)))


def ingest_s2(files_list=None, vi_name='image', scale=10000, sort_files=False, start=s2date_range[0], end=s2date_range[1]):
    
    #for order,vi in  tqdm(enumerate(vi_name),desc='Preprocessing'+f'{vi_name}',colour='blue'):
    order=0
    tmp_input=files_list
    if sort_files:
        tmp_input= sorted(tmp_input)#list(sorted(iglob(join(vi_dir+f"/{vi}",f"*_{vi}.tif"))))
    ## extract the dates from the image naming convention 
    global date_list
    date_list=[os.path.basename(i).split("_")[0] for i in tmp_input]

    srt_date=start.replace('-','')
    end_date=end.replace('-','')
    interval=[srt_date,end_date]
    try:
        subset=[date_list.index(i) for i in interval]
    except Exception as err:
        print(err)
        raise ValueError ("The specified date isn't in the list")
    else:
        date_list=date_list[subset[0]:subset[1]+1]
        tmp_input=tmp_input[subset[0]:subset[1]+1]


    sub=s2_api()
    #for i in tqdm(tmp_input, desc='ingesting ' +f'{vi_name}',colour='blue'):
    def ingest_s2_parallel(sorted_input=None, sub_response=None,idx=None,name=None):
        
        if sub['response'].upper() in ['Y','YES']:
            subset=sub['subset']
            image=gdal.Open(sorted_input).ReadAsArray(*subset)
        if sub['response'].upper() in ['N','NO']:
            image=gdal.Open(sorted_input).ReadAsArray()

        #image= np.where(image==-99999,np.nan,image)

        #using the np.array to create an artificial n-dimension array
        image=np.array([image],dtype=_dtype)
        #image=None
        return image
        #print("image {} for {} has been appended successfully".format(k,vi))
        #k+=1

    try:
        image_list=Parallel(n_jobs=nworkers)(delayed(ingest_s2_parallel)(tmp_input[idx],sub,idx, vi_name)\
                for idx in tqdm(range(len(tmp_input)),desc='ingesting ' +f'{vi_name}',colour='blue', initial=1))
    except:
        image_list=Parallel(n_jobs=nworkers,backend='threading')(delayed(ingest_s2_parallel)(tmp_input[idx],sub,idx, vi_name)\
                for idx in tqdm(range(len(tmp_input)),desc='ingesting ' +f'{vi_name}',colour='blue', initial=1))
    #stack all images along axis 0== time axis
    image_stack=np.ma.concatenate(image_list)
    
    
    mask=image_stack.mask
    image_stack=da.ma.masked_array(da.from_array(image_stack, chunks=_chunk), mask=mask).astype(_dtype)
    
    
    ###rescale VI back to the original scale 
    image_stack=image_stack/scale
    #image_stack=image_stack[subet[0]:subset[1]+1,:,:]
    sub.update({"start date":start, "end date":end,"image dimension":image_stack.shape})
    return [sub,image_stack]
        

         
        
def preprocess_s2(image_stack, param=None, vi_name="image"):
    """
    image_stack: stack of image to preprocess. which much be in 3D array

    param: preprocessing operation that should be carried out

    vi_image: Defaualt value as '<image>_mask_pixel_value.tif'. At pixel level, it reports the percentage value of non NA
             cells retained after mask operation.
    
    """
    #svi_dates=[[] for _ in range(len(svi))] #better for automation
    svi_dates=[]  #[[]]
    svi_prep=[]
    
    #defualt   
    defaultparam={"mask":True, "method":"median", "tolerance":0,\

        "threshold":None, "interpolate":True ,'export_mask_value':False,'name':f'./reports/{vi_name}_mask_pixel_value.tif',\
        'geotrans':None,'projection':None}    
    if param:
        update(defaultparam, param)
        
    
    if not defaultparam["mask"]:
        image_stack=image_stack
        svi_prep.append(image_stack)
        #TO DO: implement autodetect technique to acount for this
        svi_dates.extend([None])
        print('!!!note Without mask dates is returned as an empty list')
        
    if defaultparam["mask"]:
        method= defaultparam["method"]
        tolerance=defaultparam['tolerance']
        threshold=defaultparam["threshold"]
        export=defaultparam['export_mask_value']
        geoT=defaultparam['geotrans']
        proj=defaultparam['projection']
        name=defaultparam['name']
        if method=="median":
            mask_thres=nonpixel_count(image_stack,median=True,k=tolerance,\
                                      export_mask_value=export,name=name,geotrans=geoT,projection=proj)#k=0.1
        if method=="threshold":
            assert (defaultparam["threshold"] in range(1,100)), "The threshold has to be a numeric vale  between 1 -100%"
            mask_thres=nonpixel_count(image_stack,threshold=threshold,median=False,export_mask_value=export,name=name,geotrans=geoT,projection=proj)
        
        
        ## update the mask with dsame number of bands as the unmasked image
        mask_thres=da.broadcast_to(mask_thres,image_stack.shape)
        ####General mask 
        mask=mask_thres
        ##filter the stacked image 
        image_stack=da.ma.masked_array(image_stack,mask=mask).astype(_dtype)
        if np.all(mask)==True:
            image_stack=da.ma.masked_array((image_stack*np.nan),mask=mask).astype(_dtype)
            raise ValueError ("There isn't enough data in the pixel/scene consider reducing the lesser threshold or use a median method !!Note lesser observations") 
        else : 
             pass
        ##to sieve out scenes in the stack that does have any pixel value
        index_sieved,image_stack=image_sieve(image_stack)
        ### can be used to filter the date index
        date_list_sieved=[date_list[i] for i in index_sieved]
        date_list_sieved

        if defaultparam['interpolate']:
            t0 = time.time()
            image_stack=time_weighted_interpn(image_stack,date=date_list_sieved)
            t1 = time.time()
            print("completed the interpolation of {} successfully! in {} seconds".format(vi_name,(t1-t0)))
        #svi_prep.append(image_stack)
        svi_dates.extend(date_list_sieved)
        print("finished preprocessing of:",vi_name)
        
    return [svi_dates,image_stack]



def track_obs_id(imdate_list=None, start_date=None, end_date=None, freq=None):
    """
    imdate_list = the observed dates prior to periodic temporal interpolation in a list format.
    start_date= observed start date of periodic time series at a specified frequency
    end_date= obsevred end date of periodic time series with thesame frequency as  start date 
    freq= frequency of observation. For sentinel2 the temporal resolution is five days("5D")
    """
    try:
        if type(imdate_list)==str:
            imdate_list=os.path.normpath(imdate_list)
            if os.path.exists(imdate_list):
                try:
                    imdate_list=pd.read_csv(imdate_list)
                    imdate_list=imdate_list['Date'].to_list()
                except Exception as err:
                    print(err)
    except:
        assert (type(imdate_list)==list),"imdate:list must be of class list or pathlike"
    
    assert(type(start_date)==str and type(end_date)==str), "Start and end date date must be strings e.g '2022-06-20'"
    assert (type(freq)==str and freq[0].isdigit() and freq[-1].isalpha() and freq!=None), "frequny can be either in days or pandas freuency format e.g '5D'" 
    t_res=pd.date_range(start=start_date,end=end_date,freq=freq)

    ts=[str(j).split(" ")[0].replace("-","") for j in t_res]
    d1_index=[ts.index(str(i)) for i in imdate_list]
    return [ts,d1_index]
    

    
#TO DO: parallel processing uisng joblib not necessary
def regular_temporal_interpolation(im_array=None,imdate_list=None,start_date=None, end_date=None,freq=None,suffix=""):
    
    assert (type(imdate_list)==list),"imdate:list must be of class list"
    assert(type(start_date)==str and type(end_date)==str), "Start and end date date must be strings e.g '2022-06-20'"
    assert(isinstance(im_array, xr.DataArray) or isinstance(im_array,da.Array)), 'The input array must be a of class xarray.DataArray or dask array'
    assert(type(freq)==str and freq[0].isdigit() and freq[-1].isalpha() and freq!=None), "please specify the frequency argument.This can be either in days or any other pandas freuency format e.g '5D'"
    """
    ---------------------------
    BFAST prep for sentinel2 
    ---------------------------
    Interpolate inorder to have a periodic timeseries 
    
    example of input date format: '2022-06-20'
    
    """
    ts,d1_index=track_obs_id(imdate_list,start_date,end_date,freq=freq)   
    

    d2_index=[]
    for i in range(len(ts)):
        if i in d1_index:
            pass
        if not (i in d1_index):
            d2_index.append(i)

    date_1=imdate_list
    date_2=[ts[i] for i in d2_index]
    image_bfast_prep=mseries_time_weighted_interpn(im_array,d1=date_1,d2=date_2)
       
    print('Completed the interpolation for regular/periodic time interval')
    ##trying to insert na for visualation purpose
    date_obs=ts.copy()

    for j in d2_index:
        date_obs[j] =np.nan

    dic_date_obs={"Date":date_obs}
    date_obs_df=pd.DataFrame(dic_date_obs)
    date_obs_df.to_csv(join(out_dir,f"dates_obs_in_ts{suffix}.csv"))

    date_obs_df=date_obs_df.dropna(axis=0)
    date_obs_df.to_csv(join(out_dir,f"obs_dates{suffix}.csv"))
    #date_obs_df.to_csv(join(out_dir,"dates_obs_in_ts.csv"+suffix))


    dic_date_obs_id={"D1_index":d1_index}
    date_obs_id_df=pd.DataFrame(dic_date_obs_id)
    date_obs_id_df.to_csv(join(out_dir,f"dates_obs_index{suffix}.csv"))
    #date_obs_id_df.to_csv(join(out_dir,"dates_obs_index.csv"+suffix))
    return [ts,image_bfast_prep]

def resample_to_s1(im_array=None,imdate_list=None,targetdate_list=None):
    assert (isinstance(im_array, xr.DataArray) or isinstance(im_array,da.Array)), 'The input array must be a of class xarray.DataArray or dask array'
    assert (type(imdate_list)==list),"imdate:list must be of class list"
    assert (type(targetdate_list)==list),"targetdate_list:list must be of class list"
    assert (len(im_array.shape)>2), "The imarray must be an array(dask/xrray) with 3 axis" #and len(im_array.shape)>2
    
    temp_array=im_array.copy()
    try:
        temp_array=im_array.data
    except:
        pass
    
    #resmaple sentinel2 temporal resolution to sentinel 1 resolution
    d1=pd.date_range(start=targetdate_list[0],end=targetdate_list[-1],freq="6D")# for sentinel 1
    s1_date_series=[str(i).split(" ")[0].replace("-","") for i in d1]
    ts_comb= s1_date_series+imdate_list   # combination of observed s1(+interpolation dates for s1b prior lunch) dates and sentinel(withoutinterpolation) two dates 
    ts_comb.sort()
    #remove duplicates. this implies time steps with common acquisations dates btw s1 and s2 will remain dsame
    ts_comb=np.unique(ts_comb).tolist()
    #now find the index of only sentinel-1 dates since this will be the index of the resampled /interpolated s image 
    #in my mseries_time_weighted_interpn fnction

    s2_resampled_index=[ts_comb.index(s1) for s1 in s1_date_series]

    #ts better to use the original s2 image array values
    tcw_bfast_prep_inp=mseries_time_weighted_interpn(temp_array,d1=imdate_list,d2=s1_date_series)
    print("s2 time series has been resmapled to s1's temporal resolution\n")
    print('Validating...........\n')

    tcw_bfast_prep_inp_resampled=tcw_bfast_prep_inp[s2_resampled_index]
    if len(tcw_bfast_prep_inp_resampled)== len(s1_date_series):
        print("Accurately resmapled " +'\u2705'*10)
    else:
        raise ("XXXXXXX The lenght of the resampled and observed s1 dates are not equal")
   
    
    return tcw_bfast_prep_inp_resampled
    
    
if __name__ == "__main__":
    ingest_s2()
    preprocess_s2()
    regular_temporal_interpolation()
    resample_to_s1()
