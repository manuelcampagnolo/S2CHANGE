# PyCCD

## Install dependencies
Para começar o processamento do algoritmo PyCCD, é necessário instalar um ambiente virtual utilizando os ficheiros .yml correspondentes ao sistema operativo.
### Windows (via conda)
1. Clone repository
```
git clone https://github.com/manuelcampagnolo/S2CHANGE.git
```
2. Change directory `cd S2CHANGE`

3. Install dependencies from `yml` file using conda
```
conda env create -f ccdISA_win.yml
```
4. Activate virtual environment
```
conda activate ccdISA
```

### Linux (via conda)
1. Clone repository
```
git clone https://github.com/manuelcampagnolo/S2CHANGE.git
```
2. Change directory `cd S2CHANGE`

3. Install dependencies from `yml` file using conda
```
conda env create -f ccdISA_linux.yml
```
> For the HPC machine, use ccdISA_macc.yml
4. Activate virtual environment
```
conda activate ccdISA
```

## Execution
The program will process a time series of Sentinel-2 images using a mask that determines the region of interest and store the selected pixels in a hdf5 file. Then, it will read the hdf5 file and run the change detection algorithm for each pixel. The result is saved as a dataframe in the parquet format.

### Inputs
The program can be set to process only the hdf5 file creation, the change detection or the entire pipeline. First, we describe the inputs to process the entire pipeline.
- Time series of Sentinel-2 images, on a per tile basis
    - Data source can be GEE or Theia
    - Image names should be in a predefined format containing the image date
- Geometries of the region of interest (`.shp or .gpkg`) for masking Sentinel-2 images

Typical directory structure:
```
data_dir
   |-- ROI_mask.gpkg
   |-- imagens_Theia (if using Theia)
      |-- T29SPB
         ...
      |-- T29TNE
   |-- s2_images (if using GEE)
      |-- T29SPB
         ...
      |-- T29TNE
   |-- outputs_ROI
      |-- hdf5
         |-- T29SPB
            ...
         |-- T29TNE
      |-- plots
      |-- tabular
      |-- shapefiles
```

**Existing hdf5 file**

Alternatively, if a hdf5 already exists, it should be placed along with the `tif_dates_ord.npy` file in the output folder of the corresponding tile, such as:
```
outputs_ROI
  |-- hdf5
    |-- T29SPB
      |-- tif_dates_ord.npy
      |-- s2_images-NDVI_XX999YM1NOBS6LDA2ITER1000_START20170412_END20241229_ROINAV.h5
```

### Configurations
The user needs to define some execution parameters in the [pyccd/config/config.py](https://github.com/manuelcampagnolo/S2CHANGE/blob/main/scripts/pyccd/config/config.py) file.

*Source variables*
- `data_source_folder`: should be set to `GEE` or `THEIA`, according to the input images used
- `s2_tile_folder`: tile to be processed (e.g. `T29SPB`)
- `roi_filename`: a name to identify the region of interest

*Base path*
- Set `data_path` accordingly (for instance, it should be the path to `data_dir` in the example above)

