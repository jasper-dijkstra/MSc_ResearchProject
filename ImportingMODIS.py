# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:58:25 2021

@author: Jasper Dijkstra based on a script from Rebecca Scholten

This script can be used to calculate the annual sum of burned area:
    - download .hdf files via: sftp:\\fuoco.geog.umd.edu
        username: fire
        password: burnt
    - store hdf files in defined wdir, where each MODLAND tile is a foldername
    - the annual sum of ba, will be calculated on a 500m resolution on a mosaicked grid of all tiles in wdir
    - files are stored at: wdir + BA{year}.tif

Define variables below:

"""

# Import Modules
import os, glob
import numpy as np
import pyproj
import gdal, gdalconst, osr
import re
import time
import arcpy
arcpy.CheckOutExtension('Spatial')


#%% ===== DEFINING VARIABLES ======

wdir = r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project\c0Scratch'

# Define Startyear and Endyear of fire data
startyear = 2001
endyear = 2019 # (includes this year)

# Path to file with total pixels file (define this for more accurate resolution, else total_pix = None)
total_pix = r"C:\Users\jaspd\Desktop\AM_1265_Research_Project\03Repository\MSc_ResearchProject\00_Total_pixels_025d.npy"



# %% ========== Annual Sum Class ==========

class AnnualSumTile:
    
    def __init__(self, wdir, year, tile, total_pix = None):
        self.wdir = wdir
        self.year = year
        #self.bbox = bbox
        self.dtype = 'Float32'
        self.tile = tile # Tile (str) in the form of 'h..v..'
        assert self.tile == re.search("^h..*v..$", tile).group(), "Tile is not in the form of hXXvXX, cannot proceed!"
        if total_pix != None:
            self.resdeg = self.__ComputeResolutionInDegrees(total_pix) # resolution in degrees
        else:
              self.resdeg = 0.0045 # resolution in degrees, assuming 1deg = 111 km!
        
        # Now, calculate the annual sum of ba:
        self.AnnualSum, self.lats, self.lons, self.metadata, self.ComputationTime = self.__ComputeAnnualSum__(wdir, year, tile)
        
        return 
    
    
    def __ComputeResolutionInDegrees(self, total_pix):
        # Load the amount of MODIS pixels per 0.25 deg grid cell
        pixelgrid = np.load(total_pix)
        
        # Amount of 0.25 deg cells per hv tile
        modis_per_tile_x = np.round(pixelgrid.shape[1] / 36, 0).astype(int)
        modis_per_tile_y = np.round(pixelgrid.shape[0] / 18).astype(int)
        
        # Determine h and v
        h = int(self.tile[1:3])
        v = int(self.tile[4:6])
        
        # Determine the index at which h and v start in the 0.25 grid
        modis_ix = np.round(h * modis_per_tile_x, 0).astype(int)
        modis_iy = np.round(v * modis_per_tile_y, 0).astype(int)
        
        # Retrieve the hv tile grid at 0.25 deg
        modis_tile = pixelgrid[modis_iy:modis_iy+modis_per_tile_y, modis_ix:modis_ix+modis_per_tile_x]
        
        # Now compute the mean resolution in degrees, assuming a MODIS grid cell is 500m
        mean_square_pixel_length_m = np.sqrt(modis_tile) * 500
        resolution_deg = 0.25 * 500 / mean_square_pixel_length_m
                
        return np.mean(resolution_deg)
    
    
    def WriteTif(self, filename, scale_factor = None):
        '''writes a 3D array to geotiff, first dimension should be time
        automativally creates a nodata value based on data type
        compulsory inputs: filename, array, projection in epsg code, geotransform, data type
        optional inputs: metadata, scale-factor'''
         
        
        # set projection and geotransform
        epsg = 4326 # Define output crs (EPSG code)
        lon_min = np.min(self.lons)
        lat_max = np.max(self.lats)
        geotrans = (lon_min, self.resdeg, 0.0, lat_max, 0.0, -self.resdeg)        
        #geotrans = (self.lons[0,0], self.resdeg, 0.0, self.lats[0,0], 0.0, -self.resdeg)
        dsSRS = osr.SpatialReference()
        dsSRS.ImportFromEPSG(epsg)
        
        cols = self.AnnualSum.shape[1]
        rows = self.AnnualSum.shape[0]
        bands = 1 #len(self.AnnualSum)
        arr = np.expand_dims(self.AnnualSum, axis = 0)
        
        # set data type and nodatavalue
        if self.dtype in ['Byte', 'Int16']:
            arr = np.round(arr)
        
        if self.dtype == 'Byte':         NoDataValue = np.iinfo(np.uint8).max
        elif self.dtype == 'Int16':      NoDataValue = np.iinfo(np.int16).max
        elif self.dtype == 'Float32':    NoDataValue = -1   # Max float32 doesnt work, and danger of rounding errors: np.finfo(np.float32).max
        arr[np.isnan(arr)] = NoDataValue
        
        if self.dtype == 'Byte':         gdal_dtype = gdal.GDT_Byte  # Alternatively use: gdal_array.NumericTypeCodeToGDALTypeCode()
        elif self.dtype == 'Int16':      gdal_dtype = gdal.GDT_Int16
        elif self.dtype == 'Float32':    gdal_dtype = gdal.GDT_Float32
        
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(filename, cols, rows, bands, gdal_dtype, options=['COMPRESS=LZW', 'INTERLEAVE=BAND', 'TILED=YES'])
        ds.SetGeoTransform(geotrans)
        ds.SetProjection(dsSRS.ExportToWkt())
        
        ds.SetMetadata(self.metadata)
        
        for i in range(bands):
            band = ds.GetRasterBand(i + 1)
            if NoDataValue:
                band.SetNoDataValue(NoDataValue)
            if scale_factor is not None:
                band.SetScale(scale_factor)
            band.WriteArray(arr[i, :, :])
        
        ds = None   # close file.
        print(f'Saved File: {filename}')
        
        return None

    def __TileMap__(self, sampleloc, mapping, mask_earth=True, mres=None):
        '''
        Works like the MODIS tile calculator: https://landweb.modaps.eosdis.nasa.gov/cgi-bin/developer/tilemap.cgi
        Geolocation array can be made using 'reverse' mapping and:  index_x, index_y = np.meshgrid(range(2400), range(2400)). (see also the function 'construct_geolocation').
        FORWARD mapping = [lat, lon] -> [tile, index_y, index_x]
        REVERSE mapping = [tile, index_y, index_x] -> [lat, lon]    # Also works for noniteger and negative indices! e.g. ul tile corner is ['h19v09', -0.5, -0.5]
        FORWARD Also works for multi-dim sampleloc arrays, REVERSE does not!
        :param sampleloc: sample location [lat, lon] in degrees ('forward') or [tile, index_y, index_x] ('reverse')
        :param mapping: 'forward' or 'reverse' (output in degrees) or 'reverse_m' (output in meters).
        :param mask_earth (default=True): mask outside earth as NaN [bool]. mask_earth=False only for debugging purposes.
        :param mres (default=500): for MODIS resolution other than 500m, e.g. 250m (mres=250) or 1000m (mres=1000).
        :return: output: [tile, index_y, index_x], reverse: [lat, lon] in degrees or meters (reverse_m).
        '''
        
        if mres is None: mres = 500
        ndim = (500 * 2400) / mres  # number of row and column pixels in MODIS tile.
        
        sphere_radius = 6371007.181
        proj4str = ("+proj=sinu +a=%f +b=%f +units=m" % (sphere_radius, sphere_radius))
        p_modis_grid = pyproj.Proj(proj4str)
        
        R0 = 6371007.181000  # Earth radius in [m]
        limit_left = -20015109.354  # left limit of MODIS grid in [m]
        #limit_top = 10007554.677  # top limit of MODIS grid in [m]
        #realres = 463.312716528
        realres = ((abs(limit_left) * 2) / 36) / ndim  # actual size of each MODIS tile  (alternative: cell_size = ((limit_top*2)/18) / 2400)
        T = ndim * realres  # size of MODIS tile in meters.
        
        if mapping == 'forward':     # [lat, lon]
            
            lat = sampleloc[0]
            lon = sampleloc[1]
            
            x, y = p_modis_grid(lon, lat)
            
            lon_frac = (x / T) + 36 / 2
            lat_frac = - (y / T) + 18 / 2
            
            hn = np.floor(lon_frac).astype(int)
            vn = np.floor(lat_frac).astype(int)
            
            index_x = np.floor((lon_frac - hn) * ndim).astype(int)      # floor to get integer index. Checked -> floor is correct!
            index_y = np.floor((lat_frac - vn) * ndim).astype(int)
            
            # ### METHOD 2, avoiding the use of pyproj (Same result):
            # x = R0 * np.deg2rad(lon) * np.cos(np.deg2rad(lat))
            # y = R0 * np.deg2rad(lat)
            # 
            # hn = np.floor((x - limit_left) / T).astype(int)
            # vn = np.floor((limit_top - y) / T).astype(int)
            # 
            # index_x = np.floor(( (x - limit_left) % T) / realres).astype(int)
            # index_y = np.floor(( (limit_top - y) % T) / realres).astype(int)
            
            if index_x.ndim == 0:                       # if it concerns one value.
                tilen = 'h%02dv%02d' % (hn, vn)
                output = [tilen, index_y, index_x]
                
            elif index_x.ndim == 1:                    # if it concerns a 1d list/array of values.
                tilen = ['h%02dv%02d' % (hni, vni) for hni, vni in zip(hn, vn)]
                output = [tilen, index_y, index_x]
                
            else:                                       # if it concers a multi-dimensional array.
                output = [hn, vn, index_y, index_x]
        
        elif 'reverse' in mapping:
            
            tilen = sampleloc[0]
            index_x = sampleloc[2]
            index_y = sampleloc[1]
            
            hn = int(tilen[1:3])
            vn = int(tilen[4:])
            
            lon_frac = (index_x + 0.5) / ndim + hn      # +0.5 to get cell midcenter.
            lat_frac = (index_y + 0.5) / ndim + vn
            
            x = (lon_frac - 36/2) * T
            y = - (lat_frac - 18/2) * T
            
            lon, lat = p_modis_grid(x, y, inverse=True)
            
            # ### METHOD 2, avoiding the use of pyproj (Same result):
            # x = (index_x + 0.5) * realres + hn * T + limit_left
            # y = limit_top - (index_y + 0.5) * realres - vn * T
            # 
            # lat2 = np.rad2deg(y / R0)
            # lon2 = np.rad2deg(x / (R0 * np.cos(np.deg2rad(lat))))
            
            if mask_earth == True:
                
                phi = np.deg2rad(lat)   # phi
                # lam = np.deg2rad(lon)   # lambda
                # y2 = R0 * phi                   # Recalculate x and y. https://en.wikipedia.org/wiki/Sinusoidal_projection
                # x2 = R0 * lam * np.cos(phi)
                x_border = np.deg2rad(180.0) * R0 * np.cos(phi)
                
                outside_earth = np.abs(x) > x_border
                
                if (type(outside_earth) is np.bool_):
                    if outside_earth == True:
                        lat = np.nan
                        lon = np.nan
                        y = np.nan
                        x = np.nan
                    else: pass
                else:
                    lat[outside_earth] = np.nan
                    lon[outside_earth] = np.nan
                    y[outside_earth] = np.nan
                    x[outside_earth] = np.nan
            
            if mapping == 'reverse_m':
                output = [y, x]
            else:
                output = [lat, lon]
            
        return output


    def __ConstructGeoLocation__(self, tile, mask_earth=True, mres=None):
        """
        Construct MODIS tile geolocation arrays.
        
        :param tilen: MODIS tile [str]
        :param mask_earth: mask outside earth as NaN [bool]
        :param mres (default=500): for MODIS resolution other than 500m, e.g. 250m (mres=250) or 1000m (mres=1000).
        :return: lats_geo: latitude geolocation array
        :return: lons_geo: longitude geolocation array
        """
        
        if mres is None: mres = 500
        ndim = int((500 * 2400) / mres)    # number of row and column pixels in MODIS tile.
        
        index_x, index_y = np.meshgrid(range(ndim), range(ndim))
        lats_geo, lons_geo = self.__TileMap__([tile, index_y, index_x], mapping='reverse', mask_earth=mask_earth, mres=mres)
        
        # No need to save: loading Float64 saved .tif geoloc opens in 0.5 s, just as fast as calculating. Float64 precision is necessary.
        
        return lats_geo, lons_geo
    
    
    # __ComputeAnnualSum__
    def __ComputeAnnualSum__(self, wdir, year, tile):
        # Noting the starting time
        t0 = time.time()
        
        monthly_files = glob.glob(os.path.join(self.wdir + rf"\MODIS_MCD64A1\{self.tile}\MCD64A1.A{self.year}*.hdf"))
        
        # Generate Aggregation Matrix
        lats_geo, lons_geo = self.__ConstructGeoLocation__(self.tile, mask_earth=True, mres=500) # load geolocation arrays.
        
        outside_earth = np.isnan(lons_geo)  # pixels outside the real globe (in case of h10v02 for example).
        #lats_geo[outside_earth] = 0#999.999
        #lons_geo[outside_earth] = 0#999.999
        
        # Make sure the correct min and max vlaues are selecte near the edges
        max_lat = np.max(lats_geo[np.invert(outside_earth)])
        min_lat = np.min(lats_geo[np.invert(outside_earth)])
        max_lon = np.max(lons_geo[np.invert(outside_earth)])
        min_lon = np.min(lons_geo[np.invert(outside_earth)])
        
        # Global Lat/Lon Index
        lat_index = np.floor(np.abs(lats_geo - 90.0) * int(1 / float(self.resdeg))).astype(int)  # <- This is where the magic happens. Be sure to understand this part.
        lon_index = np.floor((lons_geo + 180.0) * int(1 / float(self.resdeg))).astype(int)
        
        # World outside (north & west) of tile (subtract this from world lat/lon index
        north = np.floor(np.max(lats_geo-90) * int(1 / float(self.resdeg))).astype(int)
        west = np.floor(np.min(lons_geo+180) * int(1 / float(self.resdeg))).astype(int)
        
        # Tile Lat/Lon Index
        lat_index = lat_index - (np.max(lat_index) - np.min(lat_index)) + north
        lon_index = lon_index - (np.max(lon_index) - np.min(lon_index)) - west

        # Initiate Raster to append output results to)
        summ = np.zeros((((max_lat - min_lat) * np.round((1 / self.resdeg), 0)).astype(int)+1, ((max_lon - min_lon) * np.round((1 / self.resdeg), 0)+1).astype(int))) # paste #tdim as another dimension at place 0
        
        # Open the files, one by one
        for filename in monthly_files:
            # Open the input monthly hdf file, read its data and NoDataValue
            ds = gdal.Open('HDF4_EOS:EOS_GRID:' + filename + ':MOD_Grid_Monthly_500m_DB_BA:Burn Date', gdalconst.GA_ReadOnly)
            Data = ds.ReadAsArray().astype(float)
            metadata = ds.GetMetadata()
            #NoDataValue = ds.GetRasterBand(1).GetNoDataValue()
            del ds # close to save memory space
            
            # Let the NoData Pixels (-1) and those without fires (-2) display the value 0
            #Data[Data == NoDataValue] = np.nan
            Data[Data <= 0] = 0
            
            # Make it a boolean raster: fires yes/no
            Data[Data > 0] = 1
            
            # if no burned area happened in the dataset, we can just keep the zeroes
            if np.sum(Data) > 0:
                index = np.where(Data > 0)
                for x, y in zip(index[0], index[1]):
                    summ[lat_index[x, y], lon_index[x, y]] = summ[lat_index[x, y], lon_index[x, y]] + Data[x, y]
            #summ += Data
            
        
        total_time = time.time() - t0
        print(f"Processed tile '{self.tile}' in: {total_time} s")

        return summ, lats_geo, lons_geo, metadata, total_time


#%% ===== MODEL FLOW =====

# Get a list with all tiles to be processed
folderlist = glob.glob(os.path.join(wdir + os.path.sep + 'MODIS_MCD64A1\\*'))
tilelist = [re.search(r'h..v..', f).group() for f in folderlist]  # list of tiles in folder.

for year in range(startyear, endyear+1, 1):
    print(f"Processing year: {year}")
    annual_tilepaths = []
    for t in tilelist:
        tile_tif = os.path.join(wdir + os.path.sep + f'BA_{year}_{t}.tif') # Define export name
        tile_obj = AnnualSumTile(wdir = wdir, year = year, tile = t, total_pix = total_pix) # Calculate the annual BA sum
        tile_obj.WriteTif(tile_tif) # Write output to tif
        annual_tilepaths.append(tile_tif) # Note output location
    
    # Now Merge all annual tile rasters back to one again:
    print(f"Merging all tiles for {year} now.")
    arcpy.management.MosaicToNewRaster(input_rasters = ';'.join(annual_tilepaths[:]),
                                       output_location = wdir,
                                       raster_dataset_name_with_extension = f"BA{year}.tif",
                                       pixel_type = "16_BIT_SIGNED",
                                       number_of_bands = 1)
    
    # Lastly, delete all intermediate tiles
    for i in annual_tilepaths:
        os.remove(i)