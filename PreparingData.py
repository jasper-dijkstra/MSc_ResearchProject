# -*- coding: utf-8 -*-
"""
Author: Jasper Dijkstra, adapted from model generated by ArcGIS ModelBuilder.



"""
import os

# Import variables from Main Script
from __main__ import *

# Load the Arcpy Module
import arcpy
arcpy.CheckOutExtension('Spatial')
from arcpy.sa import *

#=====================================
# Set Environment Settings
#=====================================
def EnvironmentSettings(wdir):
    arcpy.env.workspace = os.path.join(wdir + r'\\a0Data\\a03TempData.gdb')

    # Allow overwriting outputs
    arcpy.env.overwriteOutput = True
    
    return 



def JacobsenDataset(wdir):
    
    # Inform that Arcpy Module will be used to generate missing data
    print('Using the Arcpy Module to generate missing data')
    
    # Define Environment Settings
    EnvironmentSettings(wdir)
    arcpy.env.cellSize = 500
    
    # Identify Datasets
    NUTS_fire2 = os.path.join(wdir + os.path.sep + r'a0Data\b02Shapes\NUTS_fire2.shp')
    Low_Impact_tif = os.path.join(wdir + os.path.sep + r'c0Scratch\Low_Impact.tif')
    LowImpactLand_Table = os.path.join(wdir + "\\a0Data\\a02WorkingData.gdb\LowImpactLand_Table")
    LowImpactLand_NUTS3_Stats_xls = os.path.join(wdir + "\\a0Data\\b03ExcelCSV\\LowImpactLand_NUTS3_Stats.xls")
    
    # Importing the Conversion Toolset
    arcpy.ImportToolbox(r"c:\users\jaspd\appdata\local\programs\arcgis\pro\Resources\ArcToolbox\toolboxes\Conversion Tools.tbx")
    
    # ===== Starting the actual process
    # Process: Extract by Mask (Extract by Mask) (sa)
    LIA_clipped = arcpy.sa.ExtractByMask(in_raster=Low_Impact_tif, in_mask_data=NUTS_fire2)
    LIA_clipped.save("LIA_clipped")

    # Process: Reclassify (Reclassify) (sa)
    LIA_reclassed = arcpy.sa.Reclassify(in_raster=LIA_clipped, reclass_field="VALUE", remap="-128 NODATA;0 0;100 1", missing_values="DATA")
    LIA_reclassed.save("LIA_reclassed")

    # Process: Zonal Statistics as Table (Zonal Statistics as Table) (sa)
    print('Performing a Zonal Statistics Analysis to the dataset of Jacobsen')
    LIL_Table = arcpy.sa.ZonalStatisticsAsTable(in_zone_data = NUTS_fire2, zone_field = "NUTS_ID", 
                                    in_value_raster = LIA_reclassed, out_table = LowImpactLand_Table, 
                                    ignore_nodata = "NODATA", statistics_type = "ALL", 
                                    process_as_multidimensional = "CURRENT_SLICE", percentile_values = [100])
                                                          
    # Process: Table To Excel (Table To Excel) (conversion)
    arcpy.conversion.TableToExcel(Input_Table=LIL_Table, Output_Excel_File=LowImpactLand_NUTS3_Stats_xls, 
                                  Use_field_alias_as_column_header="ALIAS", Use_domain_and_subtype_description="CODE")
    
    return

