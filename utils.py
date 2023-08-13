import numpy as np
from osgeo import gdal
#from IPython.display import clear_output as clear
import numpy as np 
import scipy 
from tqdm import tqdm
import pandas as pd
import collections.abc #to update nested dictionary of parameters
from config import nworkers, _dtype,_dtype_int,_chunk
from joblib import Parallel , delayed
import os
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
import gc
#########----------------------------------------------------------------------------------------
def nonpixel_count(image_array,threshold=None,median=True,k=0, export_mask_value=False,\
                   name=None,geotrans=None,projection=None):
    assert k<=1, "k=tolerance from median value has to be a ratio ranging from 0-1"
    assert (isinstance(image_array, xr.DataArray) or isinstance(image_array,da.Array)), 'The input array must be a of class xarray.DataArray'
    #convert xarray to dask array
    print('Locating time series pixels with inadequate value')
    try:
        temp=image_array.data
    except:
        temp=image_array
  
    total_image_length=temp.shape[0]
    non_na_length=da.where(np.isnan(temp),0,1).sum(axis=(0)) #time axis
    per_ofnon_napixel=(non_na_length/total_image_length)*100

    if threshold:
        if export_mask_value:
            keep_pixel_mask_value(name=name,array=per_ofnon_napixel, geotrans=geotrans, projection=projection)
        mask=da.where(per_ofnon_napixel>=threshold,False,True) #since we want to use it as mask the True values will be masked off
        print("the % of non-na pixel is:",per_ofnon_napixel)
        return mask
    if median:
        mn_value=int(da.median(non_na_length,axis=(0,1)))
        tolerance=round(mn_value*k)
        tolerance=mn_value-tolerance
        if export_mask_value:
            keep_pixel_mask_value(name=name,array=non_na_length, geotrans=geotrans, projection=projection)
        mask=da.where(non_na_length>=tolerance,False,True)
        #mask=np.broadcast_to(mask,(93,200,200))
        return mask
#2------------------------------------------------------------------------------------------------------------------------------     
#####removing all scenes in the stack  that has no pixel value
def image_sieve(data):
    assert (isinstance(data, xr.DataArray) or isinstance(data,da.Array)), 'The input array must be a of class xarray.DataArray or dask array'
    print('Sieveing out scenes without  values')
    filtered_image=data.copy()
    if isinstance(filtered_image, xr.DataArray):
        filtered_image=filtered_image.data
    index=(~np.isnan(filtered_image).all(axis=(1,2))).compute() 
    index=index.ravel().tolist()
    mask=da.ma.getmaskarray(filtered_image)
    mask=da.compress(index,mask,axis=0) 
    filtered_image=da.ma.masked_array(da.compress(index,filtered_image, axis=0),mask=mask).astype(_dtype)
    
    index=np.argwhere(index).ravel().tolist()
    return index,filtered_image
    

#3  --------------------------------------------------------------------------------------------------------------------
#multiple series time weighted interpolation for ndimensional array 
def mseries_time_weighted_interpn(data,d1,d2,method="time"):
    print('Interpolating for desired time steps...')
    assert(isinstance(d1,list) and isinstance(d2,list)), 'd1 and d2 must be a list data type'
    gc.collect()
    #validity check
    if data.shape[0]==len(d1):
        d1_copy=d1.copy()
        d1_copy.extend(d2)
        ###remove identical dates. For correct interpolation similar dates in d1 and d2 has to be removed 
        d1_copy=list(set(d1_copy))
        ##sort ur date values 
        d1_copy=sorted(d1_copy)
        d1_id=[d1_copy.index(i) for i in d1]
        dt_index=pd.DatetimeIndex([pd.to_datetime(i,format="%Y-%m-%d") for i in d1_copy])
        ##create  a new ndimension array for d1 + d2
   
        _chunks=[i[0] for i in data.chunks]
        _chunks[0]=len(dt_index)
        _chunks=tuple(_chunks)
        #a template
        temp=da.zeros((len(dt_index),data.shape[1],data.shape[2])).astype(_dtype)*np.nan
        temp=temp.rechunk(_chunks)
        shape=temp.shape
        ####insert the d1 data in their new index in the extend time axis 
        k=0
        for i in d1_id:
            temp[i]=data[k]
            k+=1
        #create a dataframe with datetime index
        temp=temp.reshape((temp.shape[0],temp.shape[1]*temp.shape[2]))
    
        def _mtw_interpn(da_df, index=dt_index):

            df=da_df
            df= pd.DataFrame(df,index=index)
            #interpolate along time-axis
            df=df.interpolate(method='time')
            return df.values
        try:
            #this returns a numparray with a the modified shape
            temp=temp.map_blocks(_mtw_interpn,dtype=_dtype).compute()
            temp=temp.reshape(shape)
            #cast back to dask array
            temp=da.from_array(temp, chunks=_chunks).astype(_dtype)
        except:
            #this returns a numparray with a the modified shape
            temp=temp.map_blocks(_mtw_interpn,dtype=_dtype)
            temp=temp.reshape(shape)
            temp=temp.rechunk(_chunks)
        print(".....Done!")
        gc.collect()
        return  temp   
    else:
        raise ValueError("The lenght of the dates must be equal to the image temporal depth")
        
#4-------------------------------------------------------------------------------------------------------------------------------
def time_weighted_interpn(data,date=None):
    assert(isinstance(data, xr.DataArray) or isinstance(data,da.Array)), 'The input array must be a of class xarray.DataArray or dask array'
    print('Time weighted interpolation for the retained pixels')
    gc.collect()
    temp=data.copy()
    
    if isinstance(data, xr.DataArray) :
        temp=temp.data
    
    if temp.shape[0]==len(date):
        shape=temp.shape
        chunks=temp.chunks
        try:
            print('Calculating available NA pixels')
            all_na_loc=da.argwhere(da.isnan(temp)).compute()
            all_na_loc=all_na_loc.ravel().tolist()
        except:
            pass
        dt_index=pd.DatetimeIndex([pd.to_datetime(i,format="%Y-%m-%d") for i in date])
        ##create a dataframe 
        ###modify it takes time try doing the reshape in the sub function
        temp=temp.reshape((temp.shape[0],temp.shape[1]*temp.shape[2]))
        
        def _time_weighted_interpn(da_df, index=dt_index):
            df=da_df.data
            df= pd.DataFrame(df,index=index)
            #interpolate along time-axis
            df=df.interpolate(method='time')
            return df.values
        try:
            #this returns a numparray with a the modified shape
            temp=temp.map_blocks(_time_weighted_interpn,dtype=_dtype).compute()
            temp=temp.reshape(shape)
            #cast back to dask array
            temp=da.from_array(temp, chunks=chunks).astype(_dtype)
        except:
            #this returns a numparray with a the modified shape
            temp=temp.map_blocks(_time_weighted_interpn,dtype=_dtype)
            temp=temp.reshape(shape)
            temp=temp.rechunk(chunks)
        try:
            all_na_loc_after=da.argwhere(da.isnull(temp)).compute()
            all_na_loc_after= all_na_loc_after.ravel().tolist()              

            filled_percent=((len(all_na_loc)-len(all_na_loc_after))/len(all_na_loc))*100
        
            print(".....Done!:\n interpolated {}% of the available NA \n {}% reamaining".format(filled_percent,(100-filled_percent)))
        except:
            pass
        gc.collect()
        return temp 
    else:
        raise ValueError("The length of the dates must be equal to the image temporal depth")
 


#5 ----------------------------------------------------------------------------------------------------------------------------------
def na_fill(data, fill_values=None):
    na_loc=np.argwhere(np.isnan(data.data)).tolist()
    if len(na_loc)==len(fill_values):
        j=0
        for i in na_loc:
            data[i[0],i[1],i[2]]= fill_values[j]
            j+=1
        print("The data has been completelely filled with the provided fill values ")
    else:
        raise ValueError("The length of the nan locations ({})in  data doesn't correspond with the length({}) of the fill values provided".format(len(na_loc),len(fill_values)))


#6--------------------------------------------------------------------------------------------------- 
###export function written to make it scalable and efficient in writing large goetiff files
#without running into out of memory issues
def CreateGeoTiff(Name, Array, DataType, NDV,bandnames=None,ref_image=None, 
                  GeoT=None, Projection=None, interpolate=True):
    # If it's a 2D image we fake a third dimension:
    gc.collect()
    if len(Array.shape)==2:
        if isinstance(Array, np.ndarray):
            Array=np.array([Array])
        if isinstance(Array, da.Array):
            Array=da.array([Array])
            #Array=Array.reshape(1,Array[0],Array[1])

    if ref_image==None and (GeoT==None or Projection==None):
        raise RuntimeWarning('ref_image or settings required.')
    if bandnames != None:
        if len(bandnames) != Array.shape[0]:
            raise RuntimeError('Need {} bandnames. {} given'
                               .format(Array.shape[0],len(bandnames)))
    else:
        bandnames=['Band {}'.format(i+1) for i in range(Array.shape[0])]
    if ref_image!= None:
        refimg=gdal.Open(ref_image)
        GeoT=refimg.GetGeoTransform()
        Projection=refimg.GetProjection()
    driver= gdal.GetDriverByName('GTIFF')
    Array[np.isnan(Array)] = NDV
    DataSet = driver.Create(Name, 
            Array.shape[2], Array.shape[1], Array.shape[0], DataType)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection)
    fname=os.path.basename(Name)
    #for i, image in enumerate(Array, 1):
    def write_out_array(idx=None,im_array=None):
        if interpolate:
            if isinstance(im_array, np.ndarray):
                DataSet.GetRasterBand(idx).WriteArray(nearest_neigbor_pixel_interp(im_array))
            if isinstance(im_array, da.Array):
                DataSet.GetRasterBand(idx).WriteArray(nearest_neigbor_pixel_interp(im_array.compute()))
        else:
            if isinstance(im_array, np.ndarray):
                DataSet.GetRasterBand(idx).WriteArray(im_array)
            if isinstance(im_array, da.Array):
                DataSet.GetRasterBand(idx).WriteArray(im_array.compute())
        DataSet.GetRasterBand(idx).SetNoDataValue(NDV)
        DataSet.SetDescription(bandnames[idx-1])
    if  get_size(Array)<=1.0:
        if isinstance(Array, np.ndarray):
            Parallel(n_jobs=nworkers,backend='threading')(delayed(write_out_array)(idx,im_array)\
                    for idx,im_array in tqdm(enumerate(Array,1),desc="Exporting: "+f'{fname}',colour='blue', initial=1))
        if isinstance(Array, da.Array):
            with ProgressBar(): 
                Array=Array.compute()
                Parallel(n_jobs=nworkers,backend='threading')(delayed(write_out_array)(idx,im_array)\
                    for idx,im_array in tqdm(enumerate(Array,1),desc="Exporting: "+f'{fname}' ,colour='blue', initial=1))
    if  get_size(Array)>1.0:
        if isinstance(Array, da.Array):
            xy_chunk=Array.chunks[1:]
            x_offsets,y_offsets=np.cumsum(xy_chunk[0])-np.array(xy_chunk[0]),np.cumsum(xy_chunk[1])-np.array(xy_chunk[1])
            grid_offset=np.meshgrid(x_offsets,y_offsets)
            xy_offsets=np.c_[grid_offset[0].ravel(),grid_offset[1].ravel()]
            x_chunk=np.array(xy_chunk[0])
            y_chunk=np.array(xy_chunk[1])
            grid_chunk=np.meshgrid(x_chunk,y_chunk)
            xy_grid_chunk=np.c_[grid_chunk[0].ravel(),grid_chunk[1].ravel()]
            xy_grid_chunk=xy_grid_chunk.tolist()
        if isinstance(Array, np.ndarray):
            xy_chunk=_chunk[1:]
            array_shape=Array.shape
            x_blocks=np.ceil(array_shape[-2]/xy_chunk[0]).astype(int)
            y_blocks=np.ceil(array_shape[-1]/xy_chunk[1]).astype(int)
            x_offsets=np.arange(x_blocks,dtype=int) *xy_chunk[0]
            y_offsets=np.arange(y_blocks,dtype=int) *xy_chunk[1]
            grid_offset=np.meshgrid(x_offsets,y_offsets)
            xy_offsets=np.c_[grid_offset[0].ravel(),grid_offset[1].ravel()]
            x_chunk=(np.repeat(xy_chunk[0],x_blocks).astype(int))
            x_chunk[-1]=array_shape[1]%xy_chunk[0]
            y_chunk=(np.repeat(xy_chunk[1],y_blocks).astype(int))
            y_chunk[-1]=array_shape[2]%xy_chunk[1]
            grid_chunk=np.meshgrid(x_chunk,y_chunk)
            xy_grid_chunk=np.c_[grid_chunk[0].ravel(),grid_chunk[1].ravel()]
            xy_grid_chunk=xy_grid_chunk.tolist()
           
        def write_out_blk(idx=None,im_array=None,xoffset=None,yoffset=None):
            #[Data.GetRasterBand(idx).WriteArray(im_array[y:y+xy_chunk[1],x:x+xy_chunk[0]].compute(),xoff=x,yoff=y) for x,y in offset.tolist()]
            if interpolate:
                DataSet.GetRasterBand(idx).WriteArray(nearest_neigbor_pixel_interp(im_array),xoff=xoffset,yoff=yoffset)
            else:
                DataSet.GetRasterBand(idx).WriteArray(im_array,xoff=xoffset,yoff=yoffset)
            DataSet.GetRasterBand(idx).SetNoDataValue(NDV)
            DataSet.SetDescription(bandnames[idx-1])
        blk=0
        for x,y,xc,yc in np.hstack((xy_offsets,xy_grid_chunk)).tolist():
            if isinstance(Array, da.Array):
                #im_blk=Array[:,y:y+yc,x:x+xc].compute()
                im_blk=Array[:,x:x+xc,y:y+yc].compute()
            if isinstance(Array, np.ndarray):
                #im_blk=Array[:,y:y+yc,x:x+xc]
                im_blk=Array[:,x:x+xc,y:y+yc]
            Parallel(n_jobs=nworkers,backend='threading')(delayed(write_out_blk)(idx,im_array,xoffset=y,yoffset=x)\
                for idx,im_array in tqdm(enumerate(im_blk,1),desc="Exporting: "+f'{fname} of block{blk}' ,colour='blue', initial=1))
            blk+=1 
    DataSet.FlushCache()
    gc.collect()
    return Name



#7---------------------------------------------------------------------------------------------------------------------------------------

#TODO parallel processsing with joblib
##spatial interpolation of na pixels
def nearest_neigbor_pixel_interp(tmp_array):
    #tmp_array=array.copy()
    if isinstance(tmp_array, da.Array):
        chunks=tmp_array.chunks
    size=get_size(tmp_array)
    if len(tmp_array.shape)==2:
        mask=~(np.isnan(tmp_array))
        xx,yy = np.meshgrid(np.arange(tmp_array.shape[1]), np.arange(tmp_array.shape[0]))
        xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
        tmp_nonna_array = np.ravel( tmp_array[mask] )
        interp_instance = scipy.interpolate.NearestNDInterpolator( xym, tmp_nonna_array)
        array_inp = interp_instance(np.ravel(xx), np.ravel(yy)).reshape( xx.shape)

        return array_inp 
        
    elif len(tmp_array.shape)==3:

        def _nnpi_parallel(tmp_array,size=None):
            mask=~(np.isnan(tmp_array))
            mask_idx=mask
            tmp_array_idx=tmp_array
            xx,yy = np.meshgrid(np.arange(tmp_array_idx.shape[1]), np.arange(tmp_array_idx.shape[0]))
            xym = np.vstack( (np.ravel(xx[mask_idx]), np.ravel(yy[mask_idx])) ).T
            tmp_nonna_array = np.ravel( tmp_array_idx[mask_idx] )
            interp_instance = scipy.interpolate.NearestNDInterpolator( xym, tmp_nonna_array)
            array_inp = interp_instance(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
            array_inp=np.array([array_inp])
            if size>=1.0 and isinstance(tmp_array, da.Array):
                array_inp=da.from_array(array_inp)
            return(array_inp)
        #array_inp=Parallel(n_jobs=nworkers)(delayed(_nnpi_parallel)(tmp_array[idx].compute()) for idx in range(tmp_array.shape[0]))
        array_inp=Parallel(n_jobs=nworkers, backend='threading')(delayed(_nnpi_parallel)(tmp_array,size=size) for tmp_array in tmp_array)
        array_inp=np.concatenate(array_inp)
        if isinstance(tmp_array, da.Array):
            if size>=1.0:
                array_inp=array_inp.rechunk(chunks=chunks)
            else:
                array_inp=da.from_array(array_inp,chunks=chunks)
    return array_inp


#8---------------------------------------------------------------------------
#
def update(d, u):
    """
    Source:https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


##9--------------------------------------------------------------------------------------------------
def nearest_s1_s2_dates(s1_series_range=None,s2_series_range=None,obs_im_id=None,show=True):
    
    '''
    obs_im_id: observation image date index in the series
    returns the nearest date index of the observed date with the interpolated series
    '''
    assert (type(obs_im_id)==list), "observation image index(obs_im_id) must be a list"
    t1=pd.date_range(start=s1_series_range[0],end=s1_series_range[1],freq="6D")# for sentinel 1
    t2=pd.date_range(start=s2_series_range[0],end=s2_series_range[1], freq="5D")# for sentinel 2
    if show:
        print("sentinel 1 series",t1)
        print("sentinel 2 series",t2)

    s1_date_series=[int(str(i).split(" ")[0].replace("-",""))for i in t1]
    s2_date_series=[int(str(i).split(" ")[0].replace("-",""))for i in t2]
    s2_date_series_obs=[dates for idx,dates in enumerate(s2_date_series) if idx in obs_im_id ]
    
  

    combined=s1_date_series+s2_date_series

    comb=combined
    #comb=list(set(comb))
    comb.sort()

    #remap the observed dates index to nearest to match temporal resolution of the nearest s1 image
    # now search for the observed dates and extract the index for the nearesr dates to s1
    nearest_date=[]
    # padding to accomodate the first and last values 
    comb=[np.inf] +comb +[np.inf]
    comb_array=np.array(comb)
    for i in s2_date_series_obs:
        #steps necessary in situation in events where s1 and s2 are similar. index of the list only
        # picks one 
        temp=np.where(comb_array==i)[0]
        if len(temp)>1:
            temp=temp[1]
        else:
            temp=temp[0]

        before= comb[temp-1]
        after=comb[temp+1]
        nearest=np.argmin([abs(i-before),abs(i-after)])
        if nearest==0 :
            nearest_date.append(comb[temp-1])
        if nearest==1 :
            nearest_date.append(comb[temp+1])
        
    #find the nearest index in the combined time series 
    #intend to use this index to penalize the weight value 
    #if the s2 image falls within this index value, the weight will be equallly assigned  by the fusion function 
    #otherwise we constrain the weigth to a lesser bounds since the dates was interpolated and not observed
    #or we optimize for respective weight with the optuna module
    s2_nearest_2_s2_6D=[s1_date_series.index(i) for i in nearest_date]
    return s2_nearest_2_s2_6D

##10-----------------------------
def nearest_date(start, end, datelist, format=True):
    # if specified dates are not in the list this bl of code trys to find the nearet possible dates in repo(datelist)
    srt_date=start.replace('-','')
    end_date=end.replace('-','')
    interval=[srt_date,end_date]
        
    if not np.isin(interval, datelist).all():
        #identify wc of the two that is not in the datelist
        nodate=np.argwhere(~np.isin(interval,datelist)).ravel()
        date_list_int=np.array(sorted([int(i) for i in datelist]))
        # date_list_int=np.array(sorted([int(i) for i in datelist]))
        for i in nodate:
            if i==0:
                idx=np.where(int(interval[i])<=date_list_int)[0][0]
                #update ur intervals
                interval[0]=str(date_list_int[idx])
            if i==1:
                idx=np.where(int(interval[i])>=date_list_int)[0][-1]
                interval[1]=str(date_list_int[idx])
    if format:
        ##return in dateformat
        interval=[j[:4]+"-"+j[4:6]+"-"+j[6:] for j in interval]
    return interval


#11-------------------------------------------------------
def adjust_date (target_range=None, source_range=None, source_freq=None, format=False):
    
    """
    function to make the date range of two different series as close as possible
    """
    assert(isinstance(target_range,tuple) and isinstance(source_range,tuple)), "The target or source range must be tuple of  elements each e.g ('2017-05-11', '2022-06-20')"
    assert(len(target_range)==2 and len(source_range)==2), "The target or source range must have two elements each"
    assert(type(source_freq)==str and source_freq[0].isdigit() and source_freq[-1].isalpha() and source_freq!=None), \
    "please specify the frequency argument.This can be either in days or any other pandas freuency format e.g '5D'"
   
    #notice we selected the target_range[1] instead of the source range jst to extend the time range
    pseudo_target_series=pd.date_range(start=source_range[0], end=target_range[1],freq= source_freq)
    
    if not format:
        adjusted_source_series=[str(j).split(" ")[0].replace("-","") for j in pseudo_target_series]
    if format:
        adjusted_source_series=[str(j).split(" ")[0] for j in pseudo_target_series]
    return adjusted_source_series


#13---------------------------------------------------------------------------------------
def keep_normalize_transformer(name=None,array=None,positive_transformer=True, geotrans=None, projection=None):
    tm=str(pd.Timestamp.now()).split('.')[0]
    tm=tm.replace(' ', 'T')
    tm=tm.replace('-', '')
    tm=tm.replace(':', '')+'.tif'
    if positive_transformer:
        norm_max=np.nanmax(1+array,axis=0)
    else:
        norm_max=np.nanmax(array,axis=0)
    _dir=os.path.dirname(name)
    temp_name=os.path.basename(name)
    temp_name=temp_name.split('.')[0]
    temp_name=_dir+'/'+temp_name
    temp_name=f'{temp_name}_{tm}'

    if isinstance(array,np.ndarray):
        CreateGeoTiff(temp_name,norm_max,gdal.GDT_Float32,np.nan,bandnames=['max_pixel'], GeoT=geotrans,Projection=projection,interpolate=False)
    if isinstance(array, da.Array):
        CreateGeoTiff(temp_name,norm_max.compute(),gdal.GDT_Float32,np.nan,bandnames=['max_pixel'], GeoT=geotrans,Projection=projection, interpolate=False)
    gc.collect()
    return temp_name
#14---------------------------------------------------------------------------------------
def get_size(array=None):
    '''
    returns the size of an array in GiB
    '''
    size=array.nbytes/(1024**3)
    return size
#15-----------------------------------------------------------------------------------------------
def keep_pixel_mask_value(name=None,array=None, geotrans=None, projection=None):
    tm=str(pd.Timestamp.now()).split('.')[0]
    tm=tm.replace(' ', 'T')
    tm=tm.replace('-', '')
    tm=tm.replace(':', '')+'.tif'
    
    _dir=os.path.dirname(name)
    temp_name=os.path.basename(name)
    temp_name=temp_name.split('.')[0]
    temp_name=_dir+'/'+temp_name
    temp_name=f'{temp_name}_{tm}'
    CreateGeoTiff(temp_name,array,gdal.GDT_Int32,0,bandnames=['mask_pixel'], GeoT=geotrans,Projection=projection,interpolate=False)
    gc.collect()
    return temp_name




