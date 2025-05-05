# About
Description of the files contained in this folder

## PyCCD Processing
Upon calling `main.py` (local or hpc):
1. `preprocessing.py`
    - intermediate/auxiliary modules: `read_files.py`, `utils.py`
2. `processing.py`

Process the CCD algorithm for a given tile and region of interest.

For more information check this [link](https://github.com/manuelcampagnolo/S2CHANGE/blob/main/scripts/README.md)

## Create GeoTiff from Parquets
- `ccd_results_filter.py`

Merges files in a parquet directory and creates a GeoTiff.

**Usage**

`python ccd_results_filter.py`

Inputs:
- `input_directory`: directory containing the parquet files
  
Outputs:
- `output_raster_file`: path and name used to save the GeoTiff (needs to be set)
- `output_vector_file`: path and name of vector file with used points (needs to be set)

## Accuracy Assessment
- `avaliacao_exatidao_pyccd.py`

Conducts accuracy assessment of the pyccd results.

**Usage**

`python avaliacao_exatidao_pyccd.py`

Inputs:
- `FOLDER_PARQUET`: directory containing the parquet files (pyccd's results)
- `BDR_DGT`: path to the shp/gpkg of the reference dataset used for validation

Outputs:
- creates a `csv` file with the dataframe resulting from the accuracy assessment
    - file is saved in the `accuracy_assessment` folder inside `FOLDER_PARQUET`
- outputs accuracy metrics (F1-score, omission and commission errors) to the console

