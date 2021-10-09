# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:32:39 2021

@author: Jasper Dijkstra

This script can be used to calculate the annual sum of burned area:
    - download .hdf files via: sftp:\\fuoco.geog.umd.edu
        username: fire
        password: burnt
    - store hdf files in defined wdir, where each MODLAND tile is a foldername
    - the annual sum of ba, will be calculated on a 500m resolution on a mosaicked grid of all tiles in wdir
    - files are stored at: wdir + BA{year}.tif

Define variables below:
    
"""

import os
import re
import glob
from collections import defaultdict
import numpy as np
import rasterio as rio
from rasterio.merge import merge
from osgeo import gdal

startyear = 2001
endyear = 2019
wdir = r"C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project\c0Scratch"

output_dir = r"C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project\a0Data\b01Rasters\06_MODIS_BA"


def WriteTileToGeoTiff(hdf_in, tif_out):
    # open dataset
    dataset = gdal.Open(hdf_in,gdal.GA_ReadOnly)
    subdataset =  gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
    
    # gdalwarp
    kwargs = {'format': 'GTiff', 'dstSRS': 'EPSG:4326'}
    ds = gdal.Warp(destNameOrDestDS = tif_out, srcDSOrSrcDSTab = subdataset, **kwargs)
    del ds
    
    return


# Get a list with all tiles to be processed
folderlist = glob.glob(os.path.join(wdir + os.path.sep + 'MODIS_MCD64A1\\*'))
tilelist = [re.search(r'h..v..', f).group() for f in folderlist]  # list of tiles in folder.
print(tilelist)

all_tiff_tiles = defaultdict(dict) # Initiate dict, to store tile outputs sorted per year

for tile in tilelist: # Iterate over all tilenames
    print(f"Computing the annual burned area for tile: {tile}")
    
    for year in range(startyear, endyear+1): # Iterate over all years
        
        # Identify all tiles egible for processing (equalling defined year)
       	filelist = glob.glob(os.path.join(wdir + rf"\MODIS_MCD64A1\{tile}", '*.hdf'))  # list of files in folder.
       	filelist = [i for i in filelist if int(i[-36:-32]) == year]
        
        for i, hdf_file in enumerate(filelist): # For each HDF file (~month)
            
            temp_tif = hdf_file.replace('.hdf', '.tif') # Temporary output path
            WriteTileToGeoTiff(hdf_file, temp_tif) # Warp hdf to a tif
            
            with rio.open(temp_tif) as src:
                arr = np.mean(np.dstack(src.read()), axis=2).astype('int16') # Read the raster into a (rows, cols, depth) array,
                
                arr[arr <= 0] = 0 # Set all non burnt pixels to 0
                arr[arr > 0] = 1 # Set all burnt pixels to 1
                
                if i == 0: annual_sum = np.zeros(arr.shape).astype('int16') # Initiate an output array, if first iteration of loop
                annual_sum += arr # Add results to the output array
                
                srcprof = src.profile.copy() # Read the file profile

            src.close() # Close the dataset to save memory
            try:
                os.remove(temp_tif) # Delete the tiff file, as it is no longer necessary
            except PermissionError:
                print(f"Failed to delete {temp_tif}")

        # Update the file opts to one band
        srcprof.update(count=1, nodata=None, dtype=arr.dtype)
         
        if annual_sum.ndim < 3: # Assert the dimensions are right, else add one
            annual_sum = np.expand_dims(annual_sum, axis = 0)

        with rio.open(os.path.join(wdir + rf"\MODIS_MCD64A1\BA_{year}_{tile}.tif"), 'w', **srcprof) as dst:
            dst.write(annual_sum) # Write the output
        
        # Sort annual output tile per year
        all_tiff_tiles[year][tile] = os.path.join(wdir + rf"\MODIS_MCD64A1\BA_{year}_{tile}.tif")


for year in all_tiff_tiles:
    print(f"Mosaicking year {year}")
    
    out_file = os.path.join(output_dir + os.path.sep + f"BA{year}.tif") # Save file
    mosaic_files = list(all_tiff_tiles[year].values()) # Files to mosaic into a single raster
    
    # Open the first file to retrieve its metadata
    with rio.open(list(all_tiff_tiles[year].values())[0]) as src:
        meta = src.meta.copy()
    
    # The merge function returns a single array and the affine transform info
    arr, out_trans = merge(list(all_tiff_tiles[year].values()))
    
    meta.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "transform": out_trans
    })
    
    # Write the mosaic raster to disk
    with rio.open(out_file, "w", **meta) as dest:
        dest.write(arr)
    
    # Delete the inputs from disk
    for f in list(all_tiff_tiles[year].values()):
        os.remove(f)

