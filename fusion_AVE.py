import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd
from osgeo import gdal 
import os
import optuna
from tqdm import tqdm
from os.path import join
from glob import glob
from joblib import Parallel,delayed  #, parallel_backend
from skimage import segmentation, exposure
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from skimage import segmentation
import skimage.metrics as skm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
np.warnings.filterwarnings('ignore')
import gc
##-------------------------------------------------------------------------------------------------------------------
from utils import CreateGeoTiff,nearest_neigbor_pixel_interp as spatial_intern, nearest_s1_s2_dates,update, adjust_date,nearest_date,get_size##to updated nested dictionary, 

from config import output_directory as out_dir, s2_freq,s1_freq,s2date_range, s1date_range , auto_detect,_chunk,_dtype,_dtype_int,nworkers#extend_s2#,s1date_range(updated from the save temp dates file)

from s2_prep import track_obs_id


if not auto_detect:
    adjusted_s2_series=adjust_date(target_range=s1date_range,source_range=s2date_range,source_freq='5D', format=True)
    s2date_range=(adjusted_s2_series[0],adjusted_s2_series[-1])
if  auto_detect:
    date_list=pd.date_range(start=s1date_range[0],end=s1date_range[1], freq='6D')
    date_list=[str(j).split(" ")[0].replace("-","") for j in date_list]
    s1_rng_update=nearest_date(start=s2date_range[0],end=s2date_range[1],datelist=date_list, format=True)
    s1date_range=tuple(s1_rng_update)
    
bandnames=pd.date_range(s1date_range[0],s1date_range[1],freq=s1_freq)
bandnames=[str(str(i).split(" ")[0].replace("-",""))for i in bandnames]


#defualt parameter
#global fparam
fparam={'priority_ids':None,
            'seed':None,
            'saveproducts':True,
            'geotrans':None,
            'proj': None,
            'prefix':'',
            'nworkers':nworkers,
            'weight':[1]
            }

def _StandardScaler(images):
    scale=StandardScaler()
    shape=images[0].shape
    stack_images=np.hstack([imgs.ravel().reshape(-1,1) for imgs in images]) #two dimension array
    stack_images=scale.fit_transform(stack_images)
    stack_images=[stack_images[:,i].reshape(shape) for i in range(stack_images.shape[-1])]
    return stack_images



def evaluation(*images, fused_image=None, optimize_weight={'optimize':False,'metrics':[]}):
    supported_metric=['psnr','nmi','nrmse','mse','ssim','all']
    if optimize_weight['optimize']:
        trial=[print(i ,'is not a supported metric') for i in optimize_weight['metrics'] if i not in supported_metric]
        if len(trial)>0:
            raise RuntimeError("The supported metrics values are: psnr,nmi,nrmse,mse,ssim or,all")
    #import skimage.metrics as skm
    #metrics=pd.DataFrame(columns=['PSNR','NMI','NRMSE','MSE','SSIM'])
    metrics_dic={'psnr':[],'nmi':[],'nrmse':[],'mse':[],'ssim':[]}
    img_list=[images[0][i] for i in range(len(images[0]))]
    n_true=len(img_list)
    if  n_true>1:
        _iter=img_list[0].shape[0]
    else:
        _iter=images[0].shape[0]
    operation='Evaluating fused images'
    if optimize_weight['optimize']:
        operation='Optimizing weight value for fusion'

    for step in tqdm(range(_iter), colour='blue', desc=operation):
        if  isinstance(img_list[0], da.Array) and isinstance(fused_image, np.ndarray):
            if n_true>1:
                try:
                    metrics_dic['psnr'].append(np.sum([skm.peak_signal_noise_ratio(img_list[i][step].compute(),fused_image[step]) for i in range(n_true)])/n_true)
                except:
                    #psnr.append(None)
                    metrics_dic['psnr'].append(np.nan)
                metrics_dic['nmi'].append(np.sum([skm.normalized_mutual_information(img_list[i][step].compute(),fused_image[step]) for i in range(n_true)])/n_true)
                metrics_dic['nrmse'].append(np.sum([skm.normalized_root_mse(img_list[i][step].compute(),fused_image[step]) for i in range(n_true)])/n_true)
                metrics_dic['mse'].append(np.sum([skm.mean_squared_error(img_list[i][step].compute(),fused_image[step]) for i in range(n_true)])/n_true)
                metrics_dic['ssim'].append(np.sum([skm.structural_similarity(img_list[i][step].compute(),fused_image[step]) for i in range(n_true)])/n_true)
            else:
                try:
                    metrics_dic['psnr'].append(skm.peak_signal_noise_ratio(images[0][step].compute(),fused_image[step]))
                except:
                    metrics_dic['psnr'].append(np.nan)
                metrics_dic['nmi'].append(skm.normalized_mutual_information(images[0][step].compute(),fused_image[step]))
                metrics_dic['nrmse'].append(skm.normalized_root_mse(images[0][step].compute(),fused_image[step]))
                metrics_dic['mse'].append(skm.mean_squared_error(images[0][step].compute(),fused_image[step]))
                metrics_dic['ssim'].append(skm.structural_similarity(images[0][step].compute(), fused_image[step]))

            if optimize_weight['optimize']:
                    mcopy=metrics_dic.copy()
                    #inverse NRMSE,MSE
                    mcopy['nrmse']=1-mcopy['nrmse'][-1]
                    mcopy['mse']=abs(1-mcopy['mse'][-1])
                    in_metrics=optimize_weight['metrics']
                    mcopy={k:metrics_dic[k] for k in in_metrics}
                    mcopy=np.ravel(list(mcopy.values()))
                    mcopy=np.nanmean(mcopy)
                    return mcopy
        if  isinstance(img_list[0], da.Array) and isinstance(fused_image, da.Array):
            if n_true>1:
                try:
                    metrics_dic['psnr'].append(np.sum([skm.peak_signal_noise_ratio(img_list[i][step].compute(),fused_image[step].compute()) for i in range(n_true)])/n_true)
                except:
                    metrics_dic['psnr'].append(None)
                metrics_dic['nmi'].append(np.sum([skm.normalized_mutual_information(img_list[i][step].compute(),fused_image[step].compute()) for i in range(n_true)])/n_true)
                metrics_dic['nrmse'].append(np.sum([skm.normalized_root_mse(img_list[i][step].compute(),fused_image[step].compute()) for i in range(n_true)])/n_true)
                metrics_dic['mse'].append(np.sum([skm.mean_squared_error(img_list[i][step].compute(),fused_image[step].compute()) for i in range(n_true)])/n_true)
                metrics_dic['ssim'].append(np.sum([skm.structural_similarity(img_list[i][step].compute(),fused_image[step].compute()) for i in range(n_true)])/n_true)
            else:
                try:
                    metrics_dic['psnr'].append(skm.peak_signal_noise_ratio(images[0][step].compute(),fused_image[step].compute()))
                except:
                    metrics_dic['psnr'].append(None)
                metrics_dic['nmi'].append(skm.normalized_mutual_information(images[0][step].compute(),fused_image[step].compute()))
                metrics_dic['nrmse'].append(skm.normalized_root_mse(images[0][step].compute(),fused_image[step].compute()))
                metrics_dic['mse'].append(skm.mean_squared_error(images[0][step].compute(),fused_image[step].compute()))
                metrics_dic['ssim'].append(skm.structural_similarity(images[0][step].compute(), fused_image[step].compute()))

            if optimize_weight['optimize']:
                    mcopy=metrics_dic.copy()
                    #inverse NRMSE,MSE
                    if optimize_weight['metrics'][0]=='all':
                        mcopy['nrmse']=1-mcopy['nrmse'][-1]
                        mcopy['mse']=abs(1-mcopy['mse'][-1])
                        in_metrics=supported_metric[:-1]
                        mcopy={k:metrics_dic[k] for k in in_metrics}
                    else:
                        try:
                            in_metrics=optimize_weight['metrics']
                            mcopy={k:metrics_dic[k] for k in in_metrics}
                        except Exception as err:
                            print(err)

                    mcopy=np.ravel(list(mcopy.values()))
                    mcopy=np.nanmean(mcopy)
                    return mcopy

    metrics=pd.DataFrame(metrics_dic)
    return metrics

## a subet argument so as to be able to read from part of the s1 and s2 imageTO DO put
def fusion(*image_tuple, obs_datelist=None, fusion_param=None, evaluate=True):
    assert(isinstance(image_tuple, tuple)), 'Please provide images to be fused in as  a tple e.g (image1,image2,image3,...) '
    assert(len(image_tuple[0])>=2), "An array of more than or equal to two must be provided"
    nimage=len(image_tuple[0])
    #assert (np.sum(fparam['weight'])/nimage==1), 'The sum mean of the weights provided must be equal to 1 '
      
    
    """
    This function allows the fusion of multiple ndimesional arrays
    """
    _geotrans=[]
    _project=[]
    try:
    
        if type(image_tuple[0][0])==str:
            arrays=[]
            _geotrans=[]
            _project=[]
            for file in image_tuple[0]:
                infile=os.path.normpath(file)
                if os.path.exists(infile):
                    try:
                        infile=gdal.Open(infile)
                        _geotrans.append(infile.GetGeoTransform())
                        _project.append(infile.GetProjection())
                        #convert to dask array
                        infile=infile.ReadAsArray()
                        arrays.append(da.from_array(infile,chunks=_chunk).astype(_dtype))
                    except :
                        #print(err)
                        infile=xr.open_dataarray(infile, chunks=_chunk)#for .nc file
                        arrays(infile.data) #convert to dask array
                else:
                    raise ValueError("file doesn't exist")

            image_tuple=[arrays] #to make it conform with *image_tuple
            del(arrays)
            gc.collect()
            _geotrans=_geotrans[-1]
            _project=_project[-1]
        if type(obs_datelist)==str:
            obs_datelist=os.path.normpath(obs_datelist)
            if os.path.exists(obs_datelist):
                try:
                    obs_datelist=pd.read_csv(obs_datelist)
                    obs_datelist=obs_datelist['Date'].to_list()
                except Exception as err:
                    print(err)
       
        #defualt parameters
        fparam={'priority_ids':None,
            'seed':None,
            'saveproducts':True,
            'geotrans':None,
            'proj': None,
            'prefix':'',
            'nworkers':nworkers,
            'optimization':{'optimize':False,'direction':'maximize','opt_metric':['all'],'variable_bounds':[0,1],'timeout':25,'n_trials':10},
            'weight':[1],
            'visualization':{'viz':True,'bands':[0],'figsize':(15,15),'segment':True }
            }
        
        wfactor=fparam['weight']*nimage
        #equal weighting if the desired weights are not specifified
        wfactor=(np.array(wfactor)/nimage).tolist()
        array_shape=image_tuple[0][0].shape

        if fusion_param:
            assert (type(fusion_param)==dict),"The fusion parameters kwargs has to be a dictionary"
            #fparam.update(fusion_param)
            
            fparam=update(fparam,fusion_param)
            if  fparam['optimization']['optimize']:
                assert(nimage==2), 'if optimization is set to True, The maximum amount of image that could be fused is two'
            #if not fparam['optimization']['optimize']:
            else:
                if fparam['weight'] and fparam['optimization']['optimize']==False:
                    #assert (np.sum(fparam['weight'])==1), 'The sum of the weights ratio provided must be equal to 1 '
                    assert (isinstance(fparam['weight'],list) and len(fparam['weight'])==nimage),'The weight must be a list with len equal to the number of images to be fused if optimze is set False'
                    wfactor=fparam['weight']

        if len( _geotrans)>0:
            if fparam['geotrans']==None and fparam['proj']==None:
                fparam['geotrans']=_geotrans
                fparam['proj']=_project


        # to get a repetitive result               
        if fparam['seed']:
            np.random.seed(int(fparam['seed']))
            ## necessary to make the parallel processes random
            random_state = np.random.randint(np.iinfo(np.int32).max, size=array_shape[0]).tolist()
        if not fparam['seed']:
            random_state=[None]*array_shape[0]


        #observation date index in the series
        _,observed_date_index=track_obs_id(obs_datelist, start_date=s2date_range[0], end_date=s2date_range[1], freq=s2_freq)

        ##necessary to update the obsevred image index to the nearest corresponding resampled s1 dates
        s2_nearest_2_s2_6D=nearest_s1_s2_dates(s1_series_range=s1date_range,s2_series_range=s2date_range,obs_im_id=observed_date_index)

        if fparam['priority_ids'] :
            assert (isinstance(fparam['priority_ids'],list) and len(fparam['priority_ids'])<=array_shape[0]),"priority_ids must be of class list with length less than or equal to timeseries"
            s2_nearest_2_s2_6D=nearest_s1_s2_dates(s1_series_range=s1date_range,s2_series_range=s2date_range,obs_im_id=fparam['priority_ids'],show=False)

     
        
        #Note takes more processing time so its advisable for lesser image size
        if np.mean([get_size(i) for i in image_tuple[0] ])<1.0:
            image_tuple=[spatial_intern(image_tuple[0][i]) for i in tqdm(range(len(image_tuple[0])),colour='blue',desc='Spatial Interpolation of NA pixels')]
        else:
            image_tuple=[image_tuple[0][i] for i in range(len(image_tuple[0]))]
        gc.collect()
        def objective(trial,images,variable_bnd):
            #def fusion_obj_MI(w2, im1, im2):
            """
            objective function to optimize the fitness of  parent / optimize the weight that increase
            the relationship of metric of fused image 
            """
            #w2=trial.suggest_float("w2",  self.variable_bnd[0], self.variable_bnd[1])
            w1=trial.suggest_float("w1",  variable_bnd[0], variable_bnd[1])
            w2=1-w1
            #fused image 
            fused=((w1*images[0])+(w2*images[1]))/(w1+w2)
            score=evaluation(images,fused_image=fused, optimize_weight={'optimize':True,'metrics':fparam['optimization']['opt_metric']})

            return score #-score # to adjust maximization #return -mi_score_overall
        
        def fusion_sub_wrapper(imgs=None,obs_date_idx_list=None,wgt=None,nimgs=None, seed=None, idx=None):
           
            shape=imgs[0].shape
            if idx in obs_date_idx_list:
                #stack_images=[imgs[image][idx,:,:].compute() * (1/nimgs) for image in range(nimgs)]              
                stack_images=[imgs[image][idx,:,:] * (1/nimgs) for image in range(nimgs)]
                stack_images=np.stack(stack_images)
                stack_images= np.sum(stack_images,axis=0)
                w1_opt=1/nimgs
                score=None
                observed='observed'
            if not idx in obs_date_idx_list:
                if fparam['optimization']['optimize']:
                    stack_images=[imgs[image][idx,:,:] for image in range(nimgs)]
                    fusion_obj= lambda trial: objective(trial,stack_images,fparam['optimization']['variable_bounds'])
                    sampler = optuna.samplers.RandomSampler(seed=seed)
                    study = optuna.create_study(direction=fparam['optimization']['direction'],study_name='Fusion',sampler=sampler)
                    study.optimize(fusion_obj, n_trials=fparam['optimization']['n_trials'], gc_after_trial=True,timeout=fparam['optimization']['timeout'])
                    score=study.best_value
                    w1_opt=study.best_params['w1']
                    w2_opt=1-w1_opt
                    wgt_opt=[w1_opt,w2_opt]
                    stack_images=np.stack([stack_images[image]*wgt_opt[image] for image in range(nimgs)])
                    stack_images= np.sum(stack_images,axis=0)
                    observed='interpolated'
                else:
                    #stack_images=[imgs[image][idx,:,:].compute() for image in range(nimgs)]
                    stack_images=[imgs[image][idx,:,:] for image in range(nimgs)]
                    stack_images=np.stack([stack_images[image]*wgt[image] for image in range(nimgs)])
                    stack_images= np.sum(stack_images,axis=0)
                    w1_opt=wgt[0]
                    score=None
                    observed='interpolated'
                 
            return (stack_images,[w1_opt,score,observed])
      
        fused_series=Parallel(n_jobs=nworkers, backend='threading')(delayed(fusion_sub_wrapper)(image_tuple,s2_nearest_2_s2_6D,wfactor,nimage,seed, idx) \

                for seed,idx in tqdm(zip(random_state,range(array_shape[0])),desc='Fusing '+ f'{nimage}'+' images',colour='blue',total=array_shape[0]))
        
        #if fparam['optimization']['optimize']:
        scores=[fs_info[1] for fs_info in fused_series]
        fused_weight=pd.DataFrame(np.array(scores),columns=["Weight_image1","Best_objective_value","Observed"])
        try:
            fused_weight.to_csv(join(out_dir,'fusion_optimization_info.csv'))
            print(f"dumped optimal weight info:{join(out_dir,'fusion_optimization_info.csv')}")
        except:
            pass
        fused_series=[fs_info[0] for fs_info in fused_series]
        fused_series=np.concatenate(fused_series).reshape(array_shape)
        gc.collect()
        #else:
            #TO DO use map_blocks
            #pass
    
        if evaluate:
            evaluate_df=evaluation(image_tuple,fused_image=fused_series)
            evaluate_df['observed']='Interpolated'
            evaluate_df.iloc[s2_nearest_2_s2_6D,-1]='Observed'   #s2_nearest_2_s2_6D
            evaluate_df.to_csv(join(out_dir,'fusion_evaluation.csv'))

        if fparam['saveproducts']:
            out_product_name=join(out_dir,fparam['prefix']+"fused_image.tif")
            CreateGeoTiff(out_product_name,fused_series,gdal.GDT_Float32,np.nan,bandnames=bandnames, GeoT=fparam['geotrans'],Projection=fparam['proj'])
            print(f"dumped fused image:{out_product_name}")
        #cast back to dask array
        if not isinstance(fused_series, da.Array):
            fused_series=da.from_array(fused_series, chunks=_chunk).astype(_dtype)
        
        if fparam['visualization']['viz']:
            bands=fparam['visualization']['bands']
            figsize=fparam['visualization']['figsize']
            segment=fparam['visualization']['segment']
            temp_img=list(image_tuple)+[fused_series]
            temp_img=temp_img*len(bands)
            img_title=[f'Image {i}' for i in range(1,nimage+1)]+[' Fused Image']
            img_title=img_title*len(bands)
            fig, ax = plt.subplots(nrows=len(bands), ncols=nimage+1,figsize=figsize,sharex=True,sharey=True)
            bands=np.repeat(bands,nimage+1).tolist()
            titles= [f'Band {i} of {j}' for i,j in zip(bands,img_title)]
            axes=ax.ravel()
            for ax, band,tit,img in zip(axes,bands,titles,temp_img):
                if segment:
                    segments_fz = segmentation.felzenszwalb(img[band].compute(), scale=img[band].shape[-1], sigma=0.5, min_size=1)
                    ax.imshow(segmentation.mark_boundaries(fused_series[band].compute(), segments_fz))
                else:
                    ax.imshow(exposure.equalize_hist(img[band].compute()))
                ax.axis('off')
                ax.set(title=tit)
            plt.show()
            del(temp_img)
            gc.collect()
    except Exception as err:
        raise(err)

    else:
        print("Fusion of images completed")
        print("The fusion paramters used are",fparam)
        
        return fused_series




