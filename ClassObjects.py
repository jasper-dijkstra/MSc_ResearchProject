# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:22:21 2021

@author: Jasper Dijkstra


"""

import os


import pandas as pd
import shapefile

# Import variables from Main Script
from __main__ import *


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
        self.__ReadXLS__(filename)
        
        
    def __ReadXLS__(self, filename):
        # Check if the given file exists, else regenerate it through the Arcpy Module
        if os.path.isfile(os.path.join(data_dir + os.path.sep + filename)):
            self.data = pd.read_excel(os.path.join(data_dir + os.path.sep + filename))
        else:
            # Perform other thing
            self.__FunctionToGenerate__()
            self.data = pd.read_excel(os.path.join(data_dir + os.path.sep + filename))
    
    
    def __FunctionToGenerate__(self):
        # This function is called if the specified dataset (filename) does not exist.
        # The data is (re-)generated using the default filename at the default path
        
        # Import inside function, to make model work without Arcpy as well, when correct datasets present
        import PreparingData as prep
        
        # Check for item ID to call correct function to generate data
        if self.ID == 1: # ID 1 = Jacobsen et al. 2019 (Low Impact Land)
            prep.JacobsenDataset(wdir)          
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
        # Do stuff

