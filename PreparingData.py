# -*- coding: utf-8 -*-
"""
Author: Jasper Dijkstra

This script contains functions that (re-)generate missing data to the desired output (xls file).

NOTE that the functions in this script depend on the Arcpy library!

"""
import os
import numpy as np

# Load the Arcpy Module
import arcpy
arcpy.CheckOutExtension('Spatial')
from arcpy.sa import *



def crs(file):
    # Print CRS of given file to console
    
    arcpy.env.workspace = arcpy.GetParameterAsText(0)
    
    desc = arcpy.Describe(file)
    spatial_ref = desc.SpatialReference
    
    if spatial_ref.name == "Unknown":
        print("{0} has an unknown spatial reference".format(file))

    else:
            print("{0} : {1}".format(file, spatial_ref.name))
    
    return


def ZonalStatistics(wdir, in_zones_shp, in_value_raster, out_xls):
    """

    Parameters
    ----------
    wdir : str
        Path to Working Directory.
    in_zones_shp : str
        Path to shapefile containing the zones to retrieve data from (NUTS regions).
    in_value_raster : str
        Path to raster containing the values from which the zonal statistics have to be calculated.
    out_xls : str
        Path to output xls file.

    Returns
    -------
    XLS file saved at input parameter location.

    """
    # Set Environment Setting
    arcpy.env.workspace = os.path.join(wdir + r'\\a0Data\\a03TempData.gdb')
    arcpy.env.overwriteOutput = True
    arcpy.ImportToolbox(r"c:\users\jaspd\appdata\local\programs\arcgis\pro\Resources\ArcToolbox\toolboxes\Conversion Tools.tbx")

    # Temporary Output paths
    NUTS3_zone_raster = "NUTS3_zone_raster"
    table = "temp_table"
    
    # Check if raster size is at least 0.01 degrees
    x = float(arcpy.management.GetRasterProperties(in_value_raster, "CELLSIZEX").getOutput(0).replace(',', '.'))
    y = float(arcpy.management.GetRasterProperties(in_value_raster, "CELLSIZEY").getOutput(0).replace(',', '.'))
    if x >= 0.01 or y >= 0.01:
        # Resample in_value_raster
        print(f"Spatial resolution of {x} x {y} is too coarse for useful output! Resampling to default 0.01 x 0.01.")
        resampled_in_values = "resampled_in_values_raster"
        arcpy.Resample_management (in_value_raster, resampled_in_values, 0.01, "NEAREST")#, "BILINEAR")
        in_value_raster = resampled_in_values
        
    # Process: Polygon to Raster (Polygon to Raster) (conversion)
    print("Identifying all NUTS zones")
    arcpy.conversion.PolygonToRaster(in_features = in_zones_shp, 
                                     value_field = "NUTS_ID", 
                                     out_rasterdataset = NUTS3_zone_raster, 
                                     cell_assignment = "CELL_CENTER", 
                                     priority_field = "NONE", 
                                     cellsize = in_value_raster)
    
    # Zonal Statistics as a table
    print("Performing a zonal statistics operation")
    zone_raster = arcpy.Raster(NUTS3_zone_raster)
    arcpy.sa.ZonalStatisticsAsTable(in_zone_data = zone_raster, 
                                    zone_field = "NUTS_ID", 
                                    in_value_raster = in_value_raster, 
                                    out_table = table, 
                                    ignore_nodata = "DATA", 
                                    statistics_type = "ALL", 
                                    process_as_multidimensional = "CURRENT_SLICE", 
                                    percentile_values=[100])
    
    # Save output to xls file
    print('Saving the data to an Excel file at: {}'.format(out_xls))
    arcpy.conversion.TableToExcel(Input_Table = table, 
                                  Output_Excel_File = out_xls, 
                                  Use_field_alias_as_column_header = "ALIAS", 
                                  Use_domain_and_subtype_description = "CODE")
    
    # Delete temprorary outputs and in_memory datasets
    arcpy.management.Delete(NUTS3_zone_raster)
    arcpy.management.Delete("in_memory")
    
    return
    

def NumpyToRaster(wdir, out_file, lon, lat, data, cellsize=0.1):
    # Some Environment Settings
    arcpy.env.workspace = os.path.join(wdir + r'\\a0Data\\a03TempData.gdb')
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)
    
    # Define the lower left corner of the map
    lowerLeft = arcpy.Point(float(np.min(lon)), float(np.min(lat)))

    # Convert Raster to Array 
    tif_raster = arcpy.NumPyArrayToRaster(data, lowerLeft, cellsize, cellsize)
    tif_raster.save(out_file)
    
    # Check if the crs is assigned correctly
    crs(os.path.join(wdir + r'\\a0Data\\a03TempData.gdb\\' + out_file))
    
    return


#%% Variable Specific Functions

def ReclassLowImpactRaster(wdir, low_impact_raster_path):
    """
    This function reclassifies the input raster of the Jacobsen et al. Dataset.

    Parameters
    ----------
    wdir : str
        Path to Working Directory.
    low_impact_raster_path : str
        Path to 'Low Impact Land' raster of Jacobsen et al. (2019).

    Returns
    -------
    LIA_Reclassified : str
        Path to reclassified output raster.

    """
    # Environment Settings
    arcpy.env.workspace = os.path.join(wdir + r'\\a0Data\\a03TempData.gdb')
    arcpy.env.overwriteOutput = True
    
    # Output Dataset(s)
    LIA_Reclassified = "LIA_Reclassified"
    
    # Reclassify the data
    print('Reclassifying Low Impact Land Data...')
    LIA_Reclass = arcpy.sa.Reclassify(in_raster = arcpy.Raster(low_impact_raster_path), 
                                      reclass_field = "Value", 
                                      remap = "-128 NODATA;0 0;100 1", 
                                      missing_values = "DATA")
    
    # Save the data
    LIA_Reclass.save(LIA_Reclassified) 

    return LIA_Reclassified

