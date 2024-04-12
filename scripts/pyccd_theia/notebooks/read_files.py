from pyproj import Transformer
import os
import re
from datetime import datetime, timedelta
import geopandas as gpd
#%%
def get_most_recent_file(directory, exclude_string=None):
    try:
        # Get a list of all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # If there are no files, return None
        if not files:
            return None

        # Filter files based on the exclude_string
        if exclude_string:
            files = [f for f in files if exclude_string not in f]

        # Get the full path for each file and its corresponding modification time
        file_times = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files]

        # Find the file with the maximum modification time
        most_recent_file = max(file_times, key=lambda x: x[1])

        return most_recent_file[0]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
#%%
def convertPointToCrs(point, source_crs, target_crs):
    """
    Converts a point from a source crs to a target crs.

    Args:
        point: point (shapely.geometry.poin.Point) as extracted from a gdf.
        source_crs: original crs of the input point. Use int (e.g. 4326) or string (e.g. 'EPSG:4326')
        target_crs: new crs the the point should bear. Use int (e.g. 32629) or string (e.g. 'EPSG:32629')
    Returns:
        point with new crs
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    #create a transformer for the conversion
    x, y = point

    # transform coordinates to new crs
    new_x, new_y = transformer.transform(x, y)

    return new_x, new_y
#%%
def read_tif_files_theia(S2_tile,tiles):
    # DGT
    DGT=False
    # outro
    # Theia_T29TNE_20171007-112058

    list_files=[]
    for i in range(2017, 2022):
        if DGT: 
            if i == 2017:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process\\" + S2_tile
            else:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process_" + str(i + 1) + "\\" + S2_tile
            tiff_pattern = fr"{base_folder}\\S2*.tif"
        else:
            base_folder=tiles
            #print('base_folder',base_folder)
            tiff_pattern=re.compile('^Theia_T29TNE_' + re.escape(str(i)) + '.*tif$')

        tiff_files1=[]
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if tiff_pattern.match(file):
                    tiff_files1.append(file)
        
        # Ordena os arquivos pela data
        tiff_files = sorted(tiff_files1)
        list_files.extend(tiff_files)


    if DGT:
        dates = []
        date_pattern = re.compile(r"S2A_L2A_(\d{8})-\d{6}_"+S2_tile+".tif")
        date_pattern2 = re.compile(r"S2B_L2A_(\d{8})-\d{6}_"+S2_tile+".tif")
        for tiff_file in tiff_files:
            match = date_pattern.search(tiff_file)
            match1 = date_pattern2.search(tiff_file)
            if match:
                date = match.group(1)
                dates.append(date)
            if match1:
                date = match1.group(1)
                dates.append(date)
    else:
        L=len('Theia_T29TNE_')
        dates= [x[L:(L+8)] for x in list_files]

    date_objects = [datetime.strptime(date, '%Y%m%d').date() for date in dates]
    return list_files, date_objects
#%%
def read_tif_files_gee(S2_tile,tiles):
    list_files=[]
    DGT=False
    if DGT:
        for i in range(2017, 2022):
            if i == 2017:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process\\" + S2_tile
            else:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process_" + str(i + 1) + "\\" + S2_tile
            tiff_pattern = fr"{base_folder}\\S2*.tif"
    else:
        base_folder=tiles
        #print('base_folder',base_folder)
        tiff_pattern=re.compile('^S2SR_image_.*tif$')
    
    tiff_files1=[]
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if tiff_pattern.match(file):
                tiff_files1.append(file)
    
    # Ordena os arquivos pela data
    tiff_files = sorted(tiff_files1) #, key=extract_date)
    list_files.extend(tiff_files)

    if DGT:
        dates = []
        date_pattern = re.compile(r"S2A_L2A_(\d{8})-\d{6}_"+S2_tile+".tif")
        date_pattern2 = re.compile(r"S2B_L2A_(\d{8})-\d{6}_"+S2_tile+".tif")
        for tiff_file in tiff_files:
            match = date_pattern.search(tiff_file)
            match1 = date_pattern2.search(tiff_file)
            if match:
                date = match.group(1)
                dates.append(date)
            if match1:
                date = match1.group(1)
                dates.append(date)
    else:
        L=len('S2SR_image_')
        dates= [x[L:(L+13)] for x in list_files]

    date_objects = [datetime.utcfromtimestamp(int(date)/1000).date() for date in dates]
    return list_files, date_objects
#%%
def readPoints(caminho_arquivo, n_samples=None, random_state_value=42):
    dados_geoespaciais_metros = gpd.read_file(caminho_arquivo) # seria melhor ler csv; apenas coordenadas interessam
    if n_samples:
        dados_geoespaciais_metros = dados_geoespaciais_metros.sample(n_samples, random_state=random_state_value).copy()

    return dados_geoespaciais_metros