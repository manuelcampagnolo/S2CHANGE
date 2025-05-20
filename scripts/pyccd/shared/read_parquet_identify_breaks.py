def read_parquet_identify_breaks(filepath, extra_cols_to_read):
    """
    Separate segments that corresponds to breaks from terminal segments (not breaks)
    Assumption: all segments in the parquet file for the same pixel are in sequence, from the earlier segment to the last one (the terminal one)

    Inputs: 
    * path to parquet file (output of a pyccd task); each row is a ccd segment; the file has columns 'x_coord' and 'y_coord'
    * extra_cols_to_read: list of columns other then x_coord and y_coord to be read from the parquet file and included in the output dataframe

    Output dataframe with columns ['x_coord', 'y_coord','is_break'] plus columns in extra_cols_to_read ; each row is still a segment
    
    Note: 'is_break' is 0 for a terminal segment (not a break) and 'is_break' is 1 otherwise 
    """
    df=pd.read_parquet(filepath)[['x_coord', 'y_coord']+extra_cols_to_read]
    df['is_break']=0 #is_break==0 terminal segment; is_break==1 for segment with break
    # remove last segment for each pixel # 
    mask = (df['x_coord'] == df['x_coord'].shift(-1)) & (df['y_coord'] == df['y_coord'].shift(-1))
    df.loc[mask, 'is_break'] = 1
    return df 
