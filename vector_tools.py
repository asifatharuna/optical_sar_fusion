
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import os ,sys
from shapely.ops import transform
from shapely.wkt import loads
from shapely.geometry import Point, MultiPoint,Polygon,MultiPolygon, GeometryCollection,mapping
#import pyproj
import fiona, simplekml
from skimage import  measure
from osgeo import gdal,ogr,osr
from joblib import Parallel, delayed
from config import nworkers
from shapely.ops import polygonize as plgnz
import tempfile
import shutil


def pixel_position(input_image,x,y, geo_transformation=None):
    if geo_transformation:
        assert(len(geo_transformation)==6)
    ref=geo_transformation
    if  input_image:
        ref=gdal.Open(input_image)
        ref=ref.GetGeoTransform()
    x_pixel_offset= int(np.round_((x-ref[0])/ref[1]))
    y_pixel_offset=int(np.round_((y-ref[3])/ref[5]))
    return (x_pixel_offset,y_pixel_offset)

def pixel_position_reverse(input_image,x,y,geo_transformation=None):
    if geo_transformation:
        assert(len(geo_transformation)==6)
    ref=geo_transformation
    if  input_image:
        ref=gdal.Open(input_image)
        ref=ref.GetGeoTransform()
    xx=ref[0]+(ref[1]*x)
    yy=ref[3]+(ref[-1]*y)    
    return (xx,yy)

def transform_coordinate(coordinate=None,source_srs=None,target_srs=None, xy=False):
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Point
    import numpy as np
    point=Point(coordinate[0],coordinate[1])
    src_srs = pyproj.CRS(f'EPSG:{source_srs}')
    trg_srs = pyproj.CRS(f'EPSG:{target_srs}')
    project = pyproj.Transformer.from_crs(src_srs ,trg_srs, always_xy=xy).transform
    point_projected = transform(project, point)
    return tuple(np.array(point_projected.xy).ravel())


def reproject_vector(vector=None,source_srs=None,target_srs=None ):
    from shapely.ops import transform
    import pyproj
    src_srs = pyproj.CRS(f'EPSG:{source_srs}')
    trg_srs = pyproj.CRS(f'EPSG:{target_srs}')
    project = pyproj.Transformer.from_crs(src_srs ,trg_srs, always_xy=True).transform
    vector_projected = transform(project, vector)
    return vector_projected


def polygon_toKML(polygon=None, filename=None):
    #import simplekml 
    fname="polygon.kml"
    if filename:
        direc,fname=os.path.split(filename)
        fname=fname.split('.')[0]+'.kml'
        fname=os.path.join(direc, fname)  
    try:
        kml = simplekml.Kml()
        # Add the polygon to the KML object
        kml.newpolygon(outerboundaryis=polygon.exterior.coords)
        kml.save(fname)
    except:
        kml = simplekml.Kml()
        multipolygon = kml.newmultigeometry()
        #for poly in polygon.geoms:
        def multipolygon_wrapper(poly,ml_polygon=None):
            p = ml_polygon.newpolygon(outerboundaryis=poly.exterior.coords)
            return p
        kml_list=Parallel(n_jobs=nworkers,backend='threading')(delayed(multipolygon_wrapper)(poly,ml_polygon=multipolygon) for poly in polygon.geoms)

        kml.save(fname)
        
    return fname


def polygonize(image,values=None, display=False,return_geometry=True, export=True,file_name=None, _format=['kml'],\
               source_srs=None,target_srs=None,geotransform=None, ref_image=None):
 
    if isinstance(image,str):
        image=os.path.normpath(image)
        if os.path.exists(image):
            image=gdal.Open(image)
            _geotrans=image.GetGeoTransform()
            _project=image.GetProjection()
            image=image.ReadAsArray()
    else:
        try:
            _geotrans=geotransform
        except:
            ref=os.path.normpath(ref_image)
            ref=gdal.Open(ref)
            _geotrans=ref.GetGeoTransform()
            ref=None
   
    temp=np.where(np.isin(image,values),1,0)
    shape=image.shape
    # Find the contours of the binary mask
    contours = measure.find_contours(temp, 0.5)
    ##to make the rows and pixels conform with the standard array 
    #contours = [[(x, y) for x, y in zip(contour[:, 1], contour[:, 0])] for contour in contours ]
    contours=[np.flip(i) for i in contours]
    #contours=[np.round_(np.flip(i)) for i in contours]
    geometry=[Polygon(np.round_(contours[i]).astype(int)) for i in range(len(contours))]
    multiparts=len(geometry)
    if multiparts>1:
        geometry=MultiPolygon(geometry)
        #pass
    if multiparts==1:
        geometry=geometry[0]
    if export:
        
        assert (_geotrans!=None),'Please provide the geotransform/path to a reference image in other to export the shape file'
        supported_drivers=[drivers for drivers,_ in fiona.supported_drivers.items()]
        drivers={'geojson':'GeoJSON','shp':'ESRI Shapefile'}
        extentions={value:key for key,value in drivers.items()}
        fname="polygon"
        
        #transform the geometry from row and columns to lines and pixels or long, latitude
        if geometry.geom_type=='MultiPolygon':
            def geotransfrom_wrapper(poly):
                xx,yy=np.array(poly.exterior.xy)
                #Transformation from image coordinate space to georeferenced coordinate space 
                xx = _geotrans[0] + xx * _geotrans[1] + yy * _geotrans[2]
                yy = _geotrans[3] + xx * _geotrans[4] + yy * _geotrans[5]
                new_coordinates=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
                new_coordinates=new_coordinates.tolist()
                _geometry=Polygon(new_coordinates)
                #geometries.append(_geometry)
                return _geometry
            geometries=Parallel(n_jobs=nworkers,backend='threading')(delayed(geotransfrom_wrapper)(poly) for poly in geometry.geoms)
            geometry=MultiPolygon(geometries)
            
           
            if target_srs:
                geometry=reproject_vector(geometry, source_srs=source_srs, target_srs=target_srs)
        if geometry.geom_type=='Polygon':
            xx,yy=np.array(geometry.exterior.xy)
            xx=_geotrans[3]+(_geotrans[-1]*xx)
            yy=_geotrans[0]+(_geotrans[1]*yy)
            #new_coordinates=np.hstack((yy.reshape(-1,1),xx.reshape(-1,1)))
            new_coordinates=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
            new_coordinates=new_coordinates.tolist()
            geometry=Polygon(new_coordinates)
            if target_srs:
                geometry=reproject_vector(geometry, source_srs=source_srs, target_srs=target_srs)
                
        # Define the KML schema
        schema = {'geometry': 'Polygon','properties': {'id': 'int'},}
        trg_crs = {'init': f'epsg:{target_srs}'}
        
        for fmt in _format:
            if fmt in drivers:
                if file_name:
                    direc,fname=os.path.split(file_name)
                    fname=fname.split('.')[0]+'.' +extentions[drivers[fmt]]
                    fname=os.path.join(direc,fname)
                with fiona.open(fname, "w", drivers[fmt], schema, crs=trg_crs) as file:
                    file.write({
                        'geometry': mapping(geometry),
                        'properties': {'id': 1},
                    })
            if fmt =='kml':
                if file_name:
                    direc,fname=os.path.split(file_name)
                    fname=fname.split('.')[0]+'.kml'
                    fname=os.path.join(direc, fname)
                polygon_toKML(polygon=geometry, filename=fname)    
            print(f'Dumped:{fname} successfully')
    if display:
        fig,ax=plt.subplots(figsize=(15,10))
        ax.imshow(image, cmap='gray')
        for n, contour in enumerate(contours):
            #ax.plot(contour[:, 1], contour[:, 0], linewidth=1 , color='red')
            ax.plot(contour[:, 0], contour[:, 1], linewidth=1 , color='red')
        plt.show()
    if return_geometry:
        return geometry 
    else:
        return None
    
#TODO include other geometry types such as multilinesring etc
def apply_geotransform(vector=None, geotransformation=None):
    if vector.geom_type=='Point':
        xx,yy=np.array(vector.coords.xy)
        #Transformation from image coordinate space to georeferenced coordinate space 
        xx = geotransformation[0] + xx * geotransformation[1] + yy * geotransformation[2]
        yy = geotransformation[3] + xx * geotransformation[4] + yy * geotransformation[5]
        new_coordinates=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
        new_coordinates=new_coordinates.ravel().tolist()
        _geometry=Point(new_coordinates)
        return _geometry

    if vector.geom_type=='MultiPoint':
        mpoint=[]
        for _point in vector.geoms:
            xx,yy=np.array(_point.coords.xy)
            #Transformation from image coordinate space to georeferenced coordinate space 
            xx = geotransformation[0] + xx * geotransformation[1] + yy * geotransformation[2]
            yy = geotransformation[3] + xx * geotransformation[4] + yy * geotransformation[5]
            new_coordinates=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
            new_coordinates=new_coordinates.tolist()
            _geometry=Point(new_coordinates)
            mpoint.append(_geometry)
        mpoint=MultiPoint(mpoint)
        return  mpoint
    
    if vector.geom_type=='Polygon':
        xx,yy=np.array(vector.exterior.xy)
        #Transformation from image coordinate space to georeferenced coordinate space 
        xx = geotransformation[0] + xx * geotransformation[1] + yy * geotransformation[2]
        yy = geotransformation[3] + xx * geotransformation[4] + yy * geotransformation[5]
        new_coordinates=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
        new_coordinates=new_coordinates.tolist()
        _geometry=Polygon(new_coordinates)
        return _geometry
    elif vector.geom_type=='MultiPolygon':
        mpolygon=[]
        for poly in vector.geoms:
            if poly.is_closed:
                xx,yy=np.array(poly.exterior.xy)
                #Transformation from image coordinate space to georeferenced coordinate space 
                xx = geotransformation[0] + xx * geotransformation[1] + yy * geotransformation[2]
                yy = geotransformation[3] + xx * geotransformation[4] + yy * geotransformation[5]
                new_coordinates=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
                new_coordinates=new_coordinates.tolist()
                _geometry=Polygon(new_coordinates)
                mpolygon.append(_geometry)
        mpolygon=MultiPolygon(mpolygon)
        return mpolygon


#TODO for preservation of points
def export_GeoDataFrame(geodataframe=None, file_name=None, target_srs=None,preserve='Polygon', source_srs=4326):
    #assert (isinstance(geodataframe, gpd.GeoDataFrame)), 'geodataframe must be of class a geopandas.geodataframe.GeoDataFrame'
    assert ('geometry' in geodataframe.columns), 'The geometry column is not available'
    assert(isinstance(file_name, str)),'Please provide an output file name with'
    temp=geodataframe.copy()
    try:
        temp['geometry_image_coord']=temp['geometry_image_coord'].astype(str)
        temp['geometry_image_coord']=temp['geometry_image_coord'].apply(wkt.loads)
    except:
        pass
    #parse to a geodataframe
    try:
        temp=gpd.GeoDataFrame(temp,geometry='geometry',crs=source_srs)
    except Exception as err:
        print(err)

    try:
        if preserve=='Polygon':
            try:
                #to convert MultiLineString' to polygon
                temp['geometry']=temp['geometry'].apply(lambda x: next(plgnz(x)))
            except:
                pass
            con=[temp[['geometry']].iloc[i].values[0].type=='Polygon' for i in range(temp.shape[0])]
            temp=temp.loc[con]
            fname=os.path.normpath(file_name)
            ext=os.path.splitext(fname)[-1]
            if target_srs !=None:
                temp=temp.to_crs(target_srs)
            if ext!='.kml':
                temp.to_file(fname)
            if ext=='.kml':
                temp=temp.to_crs(4326)
                temp_dir=tempfile.mkdtemp()
                temp_path=os.path.join(temp_dir,'temporary.shp')
                temp.to_file(temp_path)
                #TODO write it in python format
                fname=os.path.splitext(fname)[0]+'.kml'
                cmd='ogr2ogr -f KML {} {} -dsco AltitudeMode=absolute'.format(fname, temp_path)
                os.system(cmd)
                #os.rmdir(temp_dir)
                shutil.rmtree(temp_dir)
        #if preserve=='Point':
                
        print(f'dumped {fname}')
        return temp
    except Exception as err:
        print(err)
        return None
   
        
        
    
    