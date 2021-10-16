# -*- coding: utf-8 -*-
"""
Author: Jasper Dijkstra

This script contains functions that (re-)generate missing data to the desired output (xls file).

NOTE that the functions in this script depend on the Arcpy library!

"""
import os, glob, re
import numpy as np
import pandas as pd

# Load the Arcpy Module
import arcpy
arcpy.CheckOutExtension('Spatial')
from arcgis.features import GeoAccessor, GeoSeriesAccessor # enables to create spatial dataframe
#from arcpy.sa import *


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


def ZonalStatistics(wdir, in_zones_shp, in_value_raster, out_xls, value_field = "NUTS_ID"):
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
    
    # Make sure raster is raster
    if type(in_value_raster) != arcpy.sa.Raster:
        in_value_raster = arcpy.sa.Raster(in_value_raster)
    
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
    print(f"Identifying all {value_field} zones")
    with arcpy.EnvManager(snapRaster = in_value_raster):
        arcpy.conversion.PolygonToRaster(in_features = in_zones_shp, 
                                         value_field = value_field, 
                                         out_rasterdataset = NUTS3_zone_raster, 
                                         cell_assignment = "CELL_CENTER", 
                                         priority_field = "NONE", 
                                         cellsize = in_value_raster)
    
    # Zonal Statistics as a table
    print("Performing a zonal statistics operation")
    zone_raster = arcpy.Raster(NUTS3_zone_raster)
    arcpy.sa.ZonalStatisticsAsTable(in_zone_data = zone_raster, 
                                    zone_field = value_field, 
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


def BA_CV_FromModis(wdir, in_zones_shp, filename, value_field = "NUTS_ID"):

    # Set Environment Setting
    arcpy.env.workspace = os.path.join(wdir + r'\\a0Data\\a03TempData.gdb')
    arcpy.env.overwriteOutput = True
    arcpy.ImportToolbox(r"c:\users\jaspd\appdata\local\programs\arcgis\pro\Resources\ArcToolbox\toolboxes\Conversion Tools.tbx")

    # Temporary Output paths
    NUTS3_zone_raster = "NUTS3_zone_raster"
    table = "temp_table"

    # Open the zones shapefile as a df with all NUTS_ID's
    df = pd.DataFrame.spatial.from_featureclass(in_zones_shp)
    df = df[[value_field]]
    
    # Get a list of all tif files in the MODIS BA folder
    all_files = glob.glob(os.path.join(wdir + os.path.sep + r"a0Data\b01Rasters\06_MODIS_BA\*.tif"))

    # Process: Polygon to Raster (Polygon to Raster) (conversion)
    print("Identifying all NUTS zones")
    with arcpy.EnvManager(snapRaster = all_files[0]):
        arcpy.conversion.PolygonToRaster(in_features = in_zones_shp, 
                                         value_field = value_field, 
                                         out_rasterdataset = NUTS3_zone_raster, 
                                         cell_assignment = "CELL_CENTER", 
                                         priority_field = "NONE", 
                                         cellsize = all_files[0])
    zone_raster = arcpy.Raster(NUTS3_zone_raster)
    
    print("Performing a zonal statistics operation for each MODIS year")
    for file in all_files:
        # Determine the year to which the file belongs
        year = re.search(r'BA2...', file).group()[2:]
        
        # Zonal Statistiscs
        arcpy.sa.ZonalStatisticsAsTable(in_zone_data = zone_raster, 
                                        zone_field = value_field, 
                                        in_value_raster = file, 
                                        out_table = table, 
                                        ignore_nodata = "DATA", 
                                        statistics_type = "SUM", 
                                        process_as_multidimensional = "CURRENT_SLICE")
        
        # Return the created statistics table as a DataFrame
        OIDFieldName = arcpy.Describe(table).OIDFieldName
        input_fields = [value_field, "SUM"]
        final_fields = [OIDFieldName] + input_fields
        np_array = arcpy.da.TableToNumPyArray(table, final_fields)#, query="", skip_nulls=False, null_values=None)
        object_id_index = np_array[OIDFieldName]
        table_df = pd.DataFrame(np_array, index=object_id_index, columns=input_fields)
        
        # Append this to the actual dataframe
        df = pd.merge(df, table_df, on=[value_field])
        df = df.rename(columns = {"SUM" : f"SUM{year}"})
        
    # Calculate the coefficient of variation
    valid_labels = [col for col in df.columns if "SUM" in col]
    df['mean'] = df[valid_labels].mean(axis = 1)
    df['stdev'] = df[valid_labels].std(axis = 1)
    df['BA_CV'] = np.divide(df['stdev'], df['mean'])
    
    # Save dataframe as csv
    df.to_csv(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV" + os.path.sep + filename))
    
    return


def TerrainRuggednessIndex(arr, nodata_value=float('nan')):
    """
    This Function calculates the Terrain Ruggedness Index of an np.array
    (https://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf)
    
    The function is based on the 'CountNeighbors Function' used in my Bachelor Thesis Project:
        https://github.com/jasper-dijkstra/bachelorthesis/blob/master/raster_tools.py
    
    Note that the outer edges of the raster might display inaccuracies as the neighbors are smaller there!
    Parameters
    ----------
    arr : arcpy.sa.raster
        Digital Elevation Model Raster from the arcpy module.
    out_path : str
        Path (incl. filename) where the output raster will be saved.

    Returns
    -------
    TRI: np.array of size arr, with terrain ruggedness values.

    """
    
    # avoid problems/biases due to nodata
    invalid_raster = np.array(arr == nodata_value).astype(int)
    arr[arr == nodata_value] = 0
    
    # Retrieve the x and y size of the array
    x, y = arr.shape
    
    # Initiate array where the sum of differences will be stored
    ssdiff = np.zeros(arr.shape)
    
    # Determine for each gird cell the original grid cell, minus its neighor:
    # We use np.abs, but Riley et al. (1999) use: np.sqrt(np.square()), which should yield the same result
    ssdiff[1:x,:]       += np.abs(arr[1:] - arr[:-1]) # North
    ssdiff[1:x,0:y-1]   += np.abs(arr[1:,:-1] - arr[:-1,1:]) # Northeast
    ssdiff[0:x,0:y-1]   += np.abs(arr[:,:-1] - arr[:,1:]) # East
    ssdiff[0:x-1,0:y-1] += np.abs(arr[:-1,:-1] - arr[1:,1:]) # Southeast
    ssdiff[0:x-1,:]     += np.abs(arr[:-1,:] - arr[1:,:]) # South
    ssdiff[0:x-1,1:]    += np.abs(arr[:-1,1:] - arr[1:,:-1]) # Southwest
    ssdiff[:,1:]        += np.abs(arr[:,1:] - arr[:,:-1]) # West
    ssdiff[1:x,1:]      += np.abs(arr[1:,1:] - arr[:-1,:-1]) # Northwest
    
    # Now take the square root of ssdiff matrix to compute the Ruggedness Index
    TRI = ssdiff
    
    # Lastly, remove the nodata values from the input again:
    TRI[invalid_raster == 1] = nodata_value
    
    return TRI


def HandleLargeRaster(function, wdir, in_raster, out_path, file_id):
    """
    Use this funciton when arcpy is not able to convert a raster to an np.array directly, because of its size.
    This function splits the raster into several smaller 'pixelblocks' that will be merged together again.
    

    Parameters
    ----------
    function : function
        Function that will be performed on each pixelblock.
    wdir : str
        Working Directory.
    in_raster : str / arcpy.sa.raster
        path to, or raster object of the raster on which the function has to be applied.
    out_path : str
        path (with filename and extension) to output raster.
    file_id : TYPE
        cahracteristic of fucntion so temporary files are easier to recognize.

    Returns
    -------
    Raster stored at out_path.

    """
    
    # Environment Settings
    #arcpy.env.workspace = os.path.join(wdir + r'\\c0Scratch\\')
    workspace_path = os.path.join(wdir + os.path.sep + r"a0Data\a03TempData.gdb")
    arcpy.env.workspace = workspace_path
    arcpy.env.outputCoordinateSystem = in_raster
    arcpy.env.cellSize = in_raster
    arcpy.env.overwriteOutput = True
    
    # Notify that the PixelBlock Function will be used
    print("The raster is very large, therefore a Pixelblock function will be used! This might take some time!")
    
    # Make sure input actually is a raster
    in_raster = arcpy.sa.Raster(in_raster)
    
    # The raster is too large to export to an array, so we'll have to use PixelBlocks
    # PC RAM = 7.88 GB -> we want to use max half, so 3.94e9 bytes
    # This means the following blocksize (assuming 1 band is used):
    blocksize = int(np.ceil(np.sqrt(3.94e09 / int(in_raster.pixelType[1:]))))
    #print(f"blocksize = {blocksize}, with type {type(blocksize)}")
    nodata_value = -32768
    filelist = []
    
    for x in range(0, in_raster.width, blocksize):
        for y in range(0, in_raster.height, blocksize):
                
            # Get lower left coordinate of map (in map units)
            mx = in_raster.extent.XMin + x * in_raster.meanCellWidth
            my = in_raster.extent.YMin + y * in_raster.meanCellWidth
            
            # Upper right coordinate of block (in cells)
            lx = min([x + blocksize, in_raster.width])
            ly = min([y + blocksize, in_raster.height])
            
            # Extract data block
            array = arcpy.RasterToNumPyArray(in_raster, arcpy.Point(mx, my), lx-x, ly-y)
            
            # Apply desired funciton upon block
            out_array = function(array, nodata_value)
            
            # Convert data block back to raster
            out_raster = arcpy.NumPyArrayToRaster(out_array, arcpy.Point(mx, my), 
                                                  in_raster.meanCellWidth,
                                                  in_raster.meanCellHeight)
            
            # Save on disk temporarily as 'filename_#.ext'
            temp_file = (f"x{x}_y{y}_tile")
            out_raster.save(temp_file)
            
            # Maintain the blocknumber and append the stored raster to list
            filelist.append(temp_file)
        
        print(f'finished row {(x/blocksize)+1} of {np.ceil(in_raster.height/blocksize)+1}')
            
    # Now Merge all rasters back to one again:
    arcpy.management.MosaicToNewRaster(input_rasters = ';'.join(filelist[:]),
                                       output_location = workspace_path,
                                       raster_dataset_name_with_extension = f"{file_id}_Python",
                                       pixel_type = "16_BIT_SIGNED",
                                       number_of_bands = 1)
    
    # Now set actual nodata to supposed nodata values
    outNull = arcpy.sa.SetNull(f"{file_id}_Python", f"{file_id}_Python", f"VALUE = {nodata_value}")
    outNull.save(out_path)

    # Release raster objects from memory
    del out_raster
    del in_raster
    del outNull
    
    # Remove temporary files
    arcpy.Delete_management("TRI_Python")
    for fileitem in filelist:
        if arcpy.Exists(fileitem):
            arcpy.Delete_management(fileitem)

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


def NumpyToRaster2(wdir, out_path, array, lowerLeft, cellSize):
    # Some Environment Settings
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)

    # Convert Raster to Array 
    tif_raster = arcpy.NumPyArrayToRaster(array, lowerLeft, cellSize, cellSize)
    tif_raster.save(out_path)
    
    # Check if the crs is assigned correctly
    crs(out_path)
    
    return


def RasterToNumpy(raster_path):
    raster = arcpy.sa.Raster(raster_path)
    
    # Extract some metadata of the raster to ease later export back to raster
    lowerLeft = arcpy.Point(raster.extent.XMin, raster.extent.YMin)
    cellSize = raster.meanCellWidth
    
    # Extract the raster values to an np.array
    array = arcpy.RasterToNumPyArray(raster, nodata_to_value = -999)
    
    return array, lowerLeft, cellSize


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


def TreeCoverDensityToFract(wdir, TreeCoverDensity):
    # Environment Settings
    arcpy.env.workspace = os.path.join(wdir + r'\\a0Data\\a03TempData.gdb')
    arcpy.env.overwriteOutput = True
    
    # Output Dataset(s)
    tcd_fract = "TCD_Fract"
    
    # Divide by 100
    tcd = arcpy.sa.Raster(TreeCoverDensity) / 100
    tcd.save(tcd_fract)
    
    return tcd_fract