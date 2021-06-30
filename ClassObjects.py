# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:22:21 2021

@author: Jasper Dijkstra

This script specifies how data should be imported

"""

import os

import numpy as np
import pandas as pd

# Local Imports
import ReadingData as read

# Import variables from Main Script
from __main__ import wdir


class IndependentVariable:
    """
    This class initiates an Independent Variable Object, containing the following information:
    
    inputs:
        ID -> ID of the variable
        name -> name of the variable
        author -> author of the input dataset
        filename -> name of the (xls) file with NUTS3 zonal data for input dataset. 
            Determined as os.path.join(wdir + os.path.sep +  r'a0Data\b03ExcelCSV') + filename
        source -> link to the source of the dataset (e.g. URL or DOI)
        atrtibute -> zonal statistics attribute to use: ['OBJECTID', 'NUTS_ID', 'ZONE_CODE', 'COUNT', 'AREA', 
                    'MIN', 'MAX', 'RANGE', 'MEAN', 'STD', 'SUM', 'VARIETY', 'MAJORITY', 'MINORITY', 'MEDIAN', 'PCT100'].
                    (The default is "MEAN")
        units -> Units of the variable of interest (The default is None)
        mode -> Only relevant in Ruggedness Index Calculation (ID = 7), and tells the algorithm to use the default computation
                or the "TRI" mode.
        
    """

    
    def __init__(self, ID, name, author, filename, source, attribute="MEAN", units=None, mode=None):
        # Add some general info
        self.ID = ID
        self.name = name
        self.author = author
        self.source = source
        self.units = units
        self.mode = mode
        self.attribute = attribute
        
        # Add the dataset, depending on its extension
        if filename.endswith(".xls"):
            self.__ReadXLS__(filename)
        elif filename.endswith(".csv") or filename.endswith(".tsv"):
            self.__ReadCSV__(filename)
        return
        
    
    def __ReadXLS__(self, filename):
        # Check if the given file exists, else regenerate it through the Arcpy Module
        if os.path.isfile(os.path.join(wdir + os.path.sep +  r'a0Data\b03ExcelCSV' + os.path.sep + filename)):
            self.metadata = pd.read_excel(os.path.join(wdir + os.path.sep +  r'a0Data\b03ExcelCSV' + os.path.sep + filename))
        else:
            self.__GenerateMissingData__(filename)
            self.metadata = pd.read_excel(os.path.join(wdir + os.path.sep +  r'a0Data\b03ExcelCSV' + os.path.sep + filename))
        
        # Keep the xls data as 'metadata' parameter and add the data of interest as 'data' parameter.
        self.data = self.metadata[["NUTS_ID", self.attribute]]
        self.data.rename(columns = {self.attribute : "Mean_{}".format(str(self.ID))}, inplace = True)

        return
    
    
    def __ReadCSV__(self, filename):
        # Import CSV or TSV files, based on independent variable ID
        if self.ID == 3: # ID 3 = population Density Dataset
            tsv_path = os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename)
            years = 2018 # 2018, because it is the most recent year with data for all NUTS3 regions
            fires = DependentVariable(os.path.join(wdir + os.path.sep + r'a0Data\b02Shapes\NUTS_fire2.shp')) # Open the NUTS shapefile, because we need the area of all the NUTS regions!
            self.metadata = read.ReadPopulationDensityTSV(tsv_path = tsv_path, 
                                                      dependent_variable_obj = fires, 
                                                      years = years)
            self.data = self.metadata[["NUTS_ID", "Dens2018 "]]
        if self.ID == 6 and not os.path.isfile(os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename)):
            self.__GenerateMissingData__(filename)           
        if self.ID == 6 and os.path.isfile(os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename)):
            self.metadata = pd.read_csv(os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename))
            self.data = self.metadata[["NUTS_ID", "BA_CV"]]
            
            # We do not want nan's in the data, so we fill them with the 'minimum' ba:
            years = [col for col in self.metadata.columns if "SUM" in col] 
            nandf = self.metadata.dropna()
            fit = np.polyfit(np.log(nandf[years].sum(axis=1)), nandf["BA_CV"], 1)
            minimum = fit[0] * np.log(1e-10) + fit[1]
            # Alternative for above method to determine the minimum, based on excel trendline (2001 - 2019)
            #minimum = -1.336 * np.log(1e-10) + 3.954
            self.data = self.data.fillna(minimum)

        return
    

    def __GenerateMissingData__(self, filename):
        # This function is called if the specified dataset (filename) does not exist.
        # The data is (re-)generated using the given filename at the default path
        
        # Import script inside function, to make model work without Arcpy as well (when correct datasets present)
        import PreparingData as prep
        
        # Input zone shapefile
        in_zones_shp = os.path.join(wdir + os.path.sep + r'a0Data\b02Shapes\NUTS_fire2.shp')
        
        # Check for item ID to call correct function to generate data
        if self.ID == 1: # ID 1 = Jacobsen et al. 2019 (Low Impact Land)
            print("Using the Arcpy Module to generate missing 'Low Impact Land' data")
            Jacobsen = os.path.join(wdir + r"\\a0Data\\b01Rasters\\01_Jacobsen2019_LowImpact.tif") # Input Raster path
            out_xls = os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename) # Output xls file path
            in_value_raster = prep.ReclassLowImpactRaster(wdir, Jacobsen)
            prep.ZonalStatistics(wdir, in_zones_shp, in_value_raster, out_xls)
        if self.ID == 2: # ID 2 = Digital Elevation Model (Altitude data)
            print('Using the Arcpy Module to generate missing Altitude data')
            DEM = os.path.join(wdir + os.path.sep + r"a0Data\b01Rasters\02_Altitude_DEM.tif") # Input Raster path
            out_xls = os.path.join(wdir + "\\a0Data\\b03ExcelCSV\\" + filename) # Output xls file path
            prep.ZonalStatistics(wdir, in_zones_shp, DEM, out_xls)
        if self.ID == 4: # ID 4 = Lightning Data
            print('Using the Arcpy Module to generate missing Lightning data')
            nc_path = os.path.join(wdir + r"\\a0Data\b04NetCDFHDF\LISOTD_HRFC_V2.3.2015.nc") # input nc path
            out_file = "LightningRaster" # Filename of raster to be generated
            out_xls = os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename) # Output xls file path
            lon, lat, data = read.ReadNC(path = nc_path, variable = "HRFC_COM_FR") # Open NC dataset
            data = np.nan_to_num(data, copy = False, nan = 0)
            prep.NumpyToRaster(wdir, out_file, lon, lat, data, cellsize=0.5) # Opened NC to raster
            prep.ZonalStatistics(wdir, in_zones_shp, out_file, out_xls) # Zonal stats on data
        if self.ID == 5: # ID 5 = Tree Cover Density
            print('Using the Arcpy Module to generate missing Tree Cover Density data')
            TCD = os.path.join(wdir + os.path.sep + r"a0Data\b01Rasters\05_TreeCoverDensity.tif") # Input Raster path
            out_xls = os.path.join(wdir + "\\a0Data\\b03ExcelCSV\\" + filename) # Output xls file path
            prep.ZonalStatistics(wdir, in_zones_shp, TCD, out_xls)
        if self.ID == 6: # ID 6 = MODIS BA data
            print("Using the Arcpy Module to generate missing Coefficient of Variation in Burned Area")
            print(f"Looking in {wdir}\a0Data\b01Rasters\06_MODIS_BA for tif files that are output of 'ImportingMODIS.py'")
            prep.BA_CV_FromModis(wdir, in_zones_shp, filename)
        if self.ID == 7: # ID 7 = Ruggedness Index
            print('Using the Arcpy Module to generate missing Ruggedness Index data')
            DEM = os.path.join(wdir + os.path.sep + r"a0Data\b01Rasters\02_Altitude_DEM.tif") # Input Raster path
            out_xls = os.path.join(wdir + "\\a0Data\\b03ExcelCSV\\" + filename) # Output xls file path
            
            # This is the default!
            if self.mode != 'TRI':
                prep.ZonalStatistics(wdir, in_zones_shp, DEM, out_xls)
            
            # This is only used if one does not wish to use the standard deviation as TRI variable
            else:
                out_tif = os.path.join(wdir + os.path.sep + r"a0Data\\b01Rasters\\08_TerrainRuggedness.tif")
                try:
                    array, lowerLeft, cellSize = prep.RasterToNumpy(DEM)
                    tri_array = prep.TerrainRuggednessIndex(array, nodata_value=float('nan'))
                    prep.NumpyToRaster2(wdir, out_tif, tri_array, lowerLeft, cellSize)
                except RuntimeError:
                    prep.HandleLargeRaster(prep.TerrainRuggednessIndex, wdir, in_raster = DEM, 
                                           out_path = out_tif, file_id = "TRI")
                prep.ZonalStatistics(wdir, in_zones_shp, out_tif, out_xls)
                
        else:
            pass
        
        return


class DependentVariable:
    """
    This class initiates a Dependent Variable Object: The European Fire Database Shapefile
    
    The Independent Variable Object consists of:
        - data_with_nan: The original Shapefile with the required fire ratio's, but where no fires were detected, 
          the ratio's return 'nan'
        - data: The same as data_with_nan, but with all 'nan' values set to 0!
        - geometries: the geometries of all shapefiles
        - countries: a list containing the abbreviations of all countries of which data is known.
    """
    
    def __init__(self, filepath):
        try:
            self.data_with_nan, self.geometries = self.__ReadShapefileSpatial__(filepath) # Open the shapefile
        except AttributeError:
            self.data_with_nan = self.__ReadShapefile__(self, filepath)
        
        self.data_with_nan = self.__AppendRatios__(self.data_with_nan) # Calculate the ratio's
        self.data = self.data_with_nan.dropna(subset = ["N_RATIO_Human", "BA_RATIO_Human"])
        #self.data = self.data_with_nan.fillna(0)# Fill the nan's with 0!
        self.countries = list(self.data["CNTR_CODE"].unique())
        self.data_header = list(self.data.columns.values.tolist())

        
    def __ReadShapefileSpatial__(self, filepath):
        # Open a shapefile and convert to a pandas DataFrame, with geometries
        from arcgis.features import GeoAccessor, GeoSeriesAccessor # enables to create spatial dataframe
        
        df = pd.DataFrame.spatial.from_featureclass(filepath)
        sdf = df[["NUTS_ID", "SHAPE"]] # Separate Geomtries
        df = pd.DataFrame(df.drop(columns='SHAPE')) # Keep DataFrame without Geometries
        
        return df, sdf
    
    def __ReadShapefile__(self, filepath):
        # Open a shapefile and convert to a pandas DataFrame, without geometries
        import shapefile
        
        shp = shapefile.Reader(filepath)
        fields = [x[0] for x in shp.fields][1:]
        records = [list(i) for i in shp.records()]
        df = pd.DataFrame(columns = fields, data = records)
        
        return df
    
    def __AppendRatios__(self, df):
        
        # Determine N_Ratio's
        df["N_RATIO_Human"] = df["nhuman"] / (df["nhuman"]+df["nlightning"])
        df["N_RATIO_Lightning"] = 1 - df["N_RATIO_Human"]
        
        # Determine BA_Ratio's
        df["BA_RATIO_Human"] = df["bahuman"] / (df["bahuman"]+df["balightnin"])
        df["BA_RATIO_Lightning"] = 1 - df["BA_RATIO_Human"]

        return df
    
    def ExportSHP(self, DataFrame, out_path):
        """Export Spatial DataFrame as shp file"""
        sdf = pd.merge(DataFrame, self.geometries, on=['NUTS_ID'])
        sdf.spatial.to_featureclass(out_path)
        return
