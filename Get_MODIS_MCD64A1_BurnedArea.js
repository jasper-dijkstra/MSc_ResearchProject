/*
This Small Google Earth Engine Script is Used to import the Required MODIS MCD64A1 data.

The returned data is a .csv file that contains the total burned area (in m^2) per NUTS3 geometry and per year

Author @ Jasper Dijkstra

Link to the Script in Google Earth Engine: https://code.earthengine.google.com/84d4275b0e18ab06912db61f430259ac

*/


// ========== VARIABLES ==========
// Time Window for BA data
var startyear = 2001;
var endyear = 2020;

// The values that will be used to remap the data later
var InRemap = ee.List.sequence(1, 366, 1);
var OutRemap = ee.List.repeat(1, InRemap.length());

// ========== FUNCTIONS ==========
// Remove properties from input featurecollection
var RemoveProperties = function(feature){
  var propertiesList = ['NAME_LATN', 'NUTS_Area', 'NUTS_ID', 'NUTS_NAME'];
  var geom = feature.geometry();
  return ee.Feature(geom).copyProperties(feature, propertiesList);
};

// retrieve the correct crs and resolution of an ee.Image
function GetProjectionAndResolution(eeImage){
  var proj = eeImage.projection();
  var scale = proj.nominalScale();
  scale = scale.multiply(ee.Number(0.00439453125));
  return [proj, scale];
}


// ========== MODEL FLOW ==========
// Import NUTS shapefiles and remove unneccesary properties for later export
var NUTS_shp = ee.FeatureCollection('users/jaspd/NUTS_fire2');
NUTS_shp = NUTS_shp.map(RemoveProperties);


// Import annual MODIS MCD64A1 Data 
var AnnualBA =  ee.FeatureCollection(ee.List.sequence(startyear, endyear).map(function (year){
  var date_start = ee.Date.fromYMD(year, 1, 1);
  var date_end = date_start.advance(1, "year");
  
  // Import and sum annual Burndate data
  var MODIS_raw = ee.ImageCollection('MODIS/006/MCD64A1')
        .select('BurnDate')
        .filterDate(date_start, date_end)
          .sum()
          .set({year: year, 'system:time_start':date_start});
  
  // Clip the data to required extent
  MODIS_raw = MODIS_raw.clip(NUTS_shp);
  
  // Set all BurnDate values to 1 (burned) 
  var burnBool = MODIS_raw.remap(InRemap, OutRemap);
  
  // Determine the area of the pixels classified as 'burned'
  var burnedArea = burnBool.multiply(ee.Image.pixelArea());
  
  // Determine scale and projection of burned Area map
  var meta = GetProjectionAndResolution(burnedArea);
  
  // Apply Zonal Statistics on Annual Image
  var ZonalStats = burnedArea.reduceRegions({
    collection: NUTS_shp, 
    reducer: ee.Reducer.sum(),
    scale: meta[1], //meters 
  });
  
  // Also add a year attribute to the Zonal Statistics Collection
  ZonalStats = ZonalStats.map(function (f) {
    return f.set('year', year);
  });
  
  return ZonalStats;
}));


print('FeatureCollection of FeatureCollections', AnnualBA);

// Create a single FeatureCollection
var AnnualBA = AnnualBA.flatten();

print('Annual Burned Area', AnnualBA);

// Export the FeatureCollection.
Export.table.toDrive({
  collection: AnnualBA,
  description: 'AnnualBurnedAreaPerNUTS',
  fileFormat: 'CSV'
});
