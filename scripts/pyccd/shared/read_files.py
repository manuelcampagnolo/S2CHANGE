from pyproj import Transformer
import os
import re
from datetime import datetime
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
def read_tif_files_theia(S2_tile, tiles, min_year, max_date):
    """
    Reads and filters Theia Sentinel-2 TIFF files from a specified directory based on a year range and maximum date.

    This function iterates over a range of years (from `min_year` to the year of `max_date`) and searches for TIFF 
    files with filenames that match a specific pattern. It returns the files that match the year range and are 
    earlier or equal to the specified `max_date`.

    Args:
        S2_tile (str) : The Sentinel-2 tile name (not used in the function but could be useful for filtering).
        tiles (str) : Path to the base folder containing the Theia TIFF files.
        min_year (int) : The starting year for filtering the files.
        max_date (datetime) : The maximum date for filtering the files. Only files with timestamps earlier than or equal 
          to this date will be included.

    Returns:
        (list_files, date_objects) (tuple) :
        - list_files (list): A list of the filtered TIFF file names.
        - date_objects (list): A list of the dates corresponding to the filtered TIFF files.
    """
    list_files=[]
    
    end_year = max_date.year
    
    for i in range(min_year, end_year+1):
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
    L=len('Theia_T29TNE_')
    dates= [x[L:(L+8)] for x in list_files]

    date_objects = [datetime.strptime(date, '%Y%m%d').date() for date in dates]
    return list_files, date_objects
#%%
def read_tif_files_gee(S2_tile, tiles, max_date):
    """
    Reads and filters GEE Sentinel-2 TIFF files from a specified directory based on a maximum date.
    It filters the files by their embedded timestamp and returns a list of files and corresponding dates that 
    are earlier or equal to the provided `max_date`.

    Args:
        S2_tile (str) : The Sentinel-2 tile name (not used in the function, but could be relevant for filtering).
        tiles (str) : Path to the base folder containing the TIFF files.
        max_date (datetime) : The maximum date for filtering the files. Only files with a timestamp less than or equal 
          to this date are returned.

    Returns:
        (list_files, date_objects) (tuple) :
        - list_files (list): A list of the filtered TIFF file names.
        - date_objects (list): A list of the dates corresponding to the filtered TIFF files.
    """
    list_files = []
    
    base_folder = tiles
    tiff_pattern = re.compile('^S2SR_image_.*tif$')
    
    tiff_files1 = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if tiff_pattern.match(file):
                tiff_files1.append(file)
    
    # Filter the files by date
    tiff_files = []
    L = len('S2SR_image_')
    for file in sorted(tiff_files1):
        date_str = file[L:(L+13)]
        timestamp_ms = int(date_str)
        timestamp_sec = timestamp_ms / 1000
        date_obj = datetime.utcfromtimestamp(timestamp_sec).date()
        if date_obj <= max_date.date():
            tiff_files.append(file)
    
    list_files.extend(tiff_files)

    dates = [x[L:(L+13)] for x in list_files]

    date_objects = []
    for date_str in dates:
        timestamp_ms = int(date_str)
        timestamp_sec = timestamp_ms / 1000
        date_obj = datetime.utcfromtimestamp(timestamp_sec).date()
        if date_obj <= max_date.date():
            date_objects.append(date_obj)
    
    return list_files, date_objects
