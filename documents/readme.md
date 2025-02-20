# Meeting feb 20, 2025

## Sentinel data

Extract from GEE [script](scripts/pyccd_theia/notebooks/download-gee-36-parts-portugal.py)

date_start = '2017-01-01'  # Escolher a data inicial para fazer o download das imagens S2
date_end = '2024-12-31'  # Escolher a data final para fazer o download das imagens S2
bandas = ['B3', 'B4', 'B8', 'B12']  # 12 bandas S2

Current: 
* 8 tiles in tif format are already available im MACC
* the are being downloaded at ISA and will be transfered to MACC
* INCD is not used anymore
* Prevision: all sentinel data imm MACC by the end of February
  
## Reference data

### Navigator data base

### ICNF

### Negative examples

## Auxiliary data

### Potencial areas of vegetation loss


### Other land cover land use maps

# Data preprocessing

## Select ROI and create npy files

What is the ROI?
* We have the potential areas from DGT (António)
* Compare with potential areas from CLC backbone vector 2018 (D)

Steps:
* for each tile, apply mask and create npy (bandsXdatesXpoints + coordinates) files
* there will be duplicate pixels from the tile overlap
* can we keep the tif files and the npy files (enough memory)?

## Running PyCCD

On INCD the estimated computation time with 5 nodes and 96 cores; for 10^6 pixels and DGT time series. [Script](scripts/pyccd_theia/notebooks/main_mpi_incd.py)
* batch size = 50 (stable between 50 and 100)
* core minutes = 16500/ 10^6 pixels
* 1305 sec per 10^6 pixels
  
**ToDo** 
* Test running PyCCD over tile T29TNE, for 10^6 pixels. To do that we need a gpkg with a 10 by 10 km square within T29TN. Goal: do a comparison like in report 2.3_v2
* Re-define function `getTimeSeriesForPoints` so we don't need to use the point gpkg as input







