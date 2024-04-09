#imports
import geemap
import ee
import os

# Initialize Earth Engine
ee.Initialize()

# choose variables
lonmin = -8.4352394959010475
lonmax = -8.4058902913773323
latmin = 38.4920591447323588
latmax = 38.5120362266671918
start_date = '2018-09-01'
end_date = '2018-10-30'
cloud_percentage = 20
scale = 10

# create output folder if it doesn't exist
parent_folder = 'NDVI_images'
subfolder = '50445-T001_EG'
output_folder = os.path.join(parent_folder, subfolder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Functions
# Define a function to mask clouds using the Sentinel-2 QA band.
def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

# Compute NDVI.
def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def download_collection(collection, output_folder, scale, region):
    try:
        # Download the collection based on the specified region (or without clipping if the region is not defined)
        geemap.download_ee_image_collection(
            collection,
            output_folder,
            scale=scale,
            region=region)
        print(f'The image was clipped and downloaded successfully in {output_folder}')
    except ValueError as e:
        print(f'Error downloading collection: {e}')

# Main

regiao = ee.Geometry.Polygon(
    [[[lonmin, latmin],
      [lonmax, latmin],
      [lonmax, latmax],
      [lonmin, latmax]]])


# determine centroid LONG and LAT
centroid = regiao.centroid().coordinates()
LONG = ee.Number(centroid.get(0)).getInfo()
LAT = ee.Number(centroid.get(1)).getInfo()
#print(LONG, LAT)

# Load Sentinel-2 surface reflectance data.
dataset = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(regiao) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage)) \
    .map(maskS2clouds)

withNDVI = dataset.map(addNDVI)

download_collection(dataset, output_folder, 10, regiao)
