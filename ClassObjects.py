# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:22:21 2021

@author: Jasper Dijkstra


"""

import os


import pandas as pd
import shapefile

# Local Imports
import ImportingData as read

# Import variables from Main Script
from __main__ import wdir, data_dir


class IndependentVariable:
    """
    This class initiates an Independent Variable Object, containing the following information:
    
    inputs:
        ID -> ID of the variable
        name -> name of the variable
        author -> author of the input dataset
        filename -> name of the (xls) file with NUTS3 zonal data for input dataset. 
            Determined as data_dir + filename
        source -> link to the source of the dataset (e.g. URL or DOI)
        
    """

    
    def __init__(self, ID, name, author, filename, source):
        # Add some general info
        self.ID = ID
        self.name = name
        self.author = author
        self.source = source
                
        # Add the actual dataset
        if filename.endswith(".xls"):
            self.__ReadXLS__(filename)
        elif filename.endswith(".csv") or filename.endswith(".tsv"):
            self.__ReadCSV__(filename)
        return
        
    
    def __ReadXLS__(self, filename):
        # Check if the given file exists, else regenerate it through the Arcpy Module
        if os.path.isfile(os.path.join(data_dir + os.path.sep + filename)):
            self.data = pd.read_excel(os.path.join(data_dir + os.path.sep + filename))
        else:
            self.__FunctionToGenerate__(filename)
            self.data = pd.read_excel(os.path.join(data_dir + os.path.sep + filename))
        return
    
    
    def __ReadCSV__(self, filename):
        # Import CSV or TSV files, based on independent variable ID
        fires = DependentVariable(os.path.join(wdir + os.path.sep + r'a0Data\b02Shapes\NUTS_fire2.shp'))
        
        if self.ID == 3: # ID 3 population Density Dataset
            tsv_path = os.path.join(wdir + r"\\a0Data\\b03ExcelCSV\\" + filename)
            years = 2018 # 2018, because it is the most recent year with data for all NUTS3 regions
            self.data = read.ReadPopulationDensityTSV(tsv_path = tsv_path, 
                                                      dependent_variable_obj = fires, 
                                                      years = years)
        return
    
    
    def __FunctionToGenerate__(self, filename):
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
        else:
            pass
        
        return
    
    
class DependentVariable:
    
    def __init__(self, filepath):
        self.__readshapefile__(filepath)
        
        
        
    def __readshapefile__(self, filepath):
        shp = shapefile.Reader(filepath)
        fields = [x[0] for x in shp.fields][1:]
        records = [list(i) for i in shp.records()]
        shps = [s.points for s in shp.shapes()]
        
        df = pd.DataFrame(columns = fields, data = records)
        df.assign(coords = shps)
        
        self.data = df
        
        return
