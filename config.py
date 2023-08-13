import os
from os.path import join    
from glob import glob
from osgeo import gdal



##visualization
viz = True
extend_s2=True
_dtype='float32'
_dtype_int='int32'

#define chuck size if the analysis will be carried out on a large data set
_chunk=(None,100,100)

#If the "extends_s2" parameter is set to true, it means that the optical data from Sentinel-2 will be 
# #interpolated or extended to match the end date of the SAR data from Sentinel-1. This interpolation allows
# # both datasets to have the same temporal resolution and aligns their time series for further analysis or comparison.
#otherwise, the "auto_detect" is activated and both datasets will be cropped to match the shortest end date without any interpolation.
if extend_s2==False:
    auto_detect=True
if extend_s2==True:
    auto_detect=False



global vi_dir
global vv_dir
global vh_dir
global  output_directory
#setting the data input directory

vi_dir='./data/optical/vi/'
vv_dir='./data/sar/vv/'
vh_dir='./data/sar/vh/'

#setting the data output directory
output_directory='./products/'



global s1date_range
global s2date_range

global s1_freq
global s2_freq

#the default temporal resolution of sar and optical repsectively
s1_freq="6D"
s2_freq="5D"

#In this context, "nworkers" represents the number of cores that will be used for parallel processing to maintain efficiency and reduce processing time
nworkers=int(os.cpu_count()-1)

#List of adopted index and this list of indices must match with bands available in each file in the "./data/optical/optical_download" directory
#svi=["NDRE2","NBR","TCW","TCB","TCG","DI"]
svi=["NBR"]

#list of polarization that will be used in the fusion analysis
polarization= ['vv'] #['vv', 'vh'] 



# Modify the date range to fit your study period
s1date_range=('2017-05-11','2023-03-29')  

s2date_range=('2017-05-10','2023-02-13')  

#TO DO: automatically grab the sorted first and last image from the local repository

#if the data needs to be rescale to its true value
s2_scale=10000

s2_files=[]

for i in svi:
    file_list=sorted(glob(join(vi_dir,f"{i}","*.tif")))
    if len(file_list)>=1:
        s2_files.append(file_list)
    else: 
        s2_files.append(sorted(glob('./data/optical/optical_download/*.tif')))
        #C:\Users\OLAWALE\Desktop\Portfolio_all\optical_sar_fusion\data\optical\optical_download
    break



s1_files_vv= sorted(glob(join(vv_dir, "S1*.tif"),recursive=True))
s1_files_vh= sorted(glob(join(vh_dir, "S1*.tif"),recursive=True))
if not len(s1_files_vv)>=1:
    s1_files_vv=sorted(glob('./data/sar/sar_download/*.tif'))  
    
    
###hard coded reference image    
ref_image_s2=s2_files[0][0]
ref_image_s1=s1_files_vv[0]



#This is used for coverting points from coordinate reference systems to image reference systems
def pixel_position(input_image,x,y):
    ref=gdal.Open(input_image)
    ref=ref.GetGeoTransform()
    x_pixel_offset= int((x-ref[0])/ref[1])
    y_pixel_offset=int((y-ref[3])/ref[5])
    
    return (x_pixel_offset,y_pixel_offset)
    
#api model for accessing data from local directory specified above 
def s2_api():
    #s1_s2=str(input(' Which sentinel product s1/s2/both?'))
    #if s1_s2.upper() in ['S2','BOTH']
    sub=str(input('Do want to run the preprocessing on the image subset [Y/N]?'))
    if sub.upper() in ['Y','YES']:
        #geotrans=input('Please input the geotransform information in order \n long,lat,pixels_x,pixe_y')
        bounds=input('Please input the bounds information \n offset long:\n')+','+input('offset lat:\n')+','+\
        input('no of pixels_x:\n')+','+input('no of pixel_y:\n')
        bounds=bounds.split(',')
        
    
        try:
            bounds=[float(geo) for geo in bounds]
            x_cc,y_cc=pixel_position(ref_image_s2,bounds[0],bounds[1])
            subset=(x_cc,y_cc,int(bounds[2]),int(bounds[3]))
            
            ### needed parameters to export the array with the right coordinate properties
            ref_image_info=gdal.Open(ref_image_s2)
            proj=ref_image_info.GetProjection()
            geotrans=list(ref_image_info.GetGeoTransform())
            subset_xoff=geotrans[0]+subset[0]*geotrans[1]  
            subset_yoff=geotrans[3]+subset[1]*geotrans[5] 
            geotrans[0]=subset_xoff
            geotrans[3]=subset_yoff
            geotrans=tuple(geotrans)
        except Exception as err:
            print(err)
        
        else:
            return {"response": sub, "subset":subset, "proj":proj, "geotransformation":geotrans}
    if sub.upper() in ['N','No']:
        ref_image_info=gdal.Open(s2_files[0][0])
        proj=ref_image_info.GetProjection()
        geotrans=list(ref_image_info.GetGeoTransform())
        geotrans=tuple(geotrans)
    
        return {'response': sub, "proj":proj, "geotransformation":geotrans}
    
def s1_api():
    
    sub=str(input('Do want to run the preprocessing on the image subset [Y/N]?'))
    if sub.upper() in ['Y','YES']:
        #geotrans=input('Please input the geotransform information in order \n long,lat,pixels_x,pixe_y')
        bounds=input('Please input the bounds information \n offset long:\n')+','+input('offset lat:\n')+','+\
        input('no of pixels_x:\n')+','+input('no of pixel_y:\n')
        bounds=bounds.split(',')
        
        try:
            bounds=[float(geo) for geo in bounds]
            x_cc,y_cc=pixel_position(ref_image_s1,bounds[0],bounds[1])
            subset=(x_cc,y_cc,int(bounds[2]),int(bounds[3]))
            
            ### needed parameters to export the array with the right coordinate properties
            ref_image_info=gdal.Open(ref_image_s1)
            proj=ref_image_info.GetProjection()
            geotrans=list(ref_image_info.GetGeoTransform())
            subset_xoff=geotrans[0]+subset[0]*geotrans[1]  
            subset_yoff=geotrans[3]+subset[1]*geotrans[5] 
            geotrans[0]=subset_xoff
            geotrans[3]=subset_yoff
            geotrans=tuple(geotrans)
        except Exception as err:
            print(err)
        else:
             return {"response": sub, "subset":subset, "proj":proj, "geotransformation":geotrans}
    if sub.upper() in ['N','No']:
        ref_image_info=gdal.Open(s1_files_vv[0])
        proj=ref_image_info.GetProjection()
        geotrans=list(ref_image_info.GetGeoTransform())
        geotrans=tuple(geotrans)
        return {'response': sub, "proj":proj, "geotransformation":geotrans}
    

