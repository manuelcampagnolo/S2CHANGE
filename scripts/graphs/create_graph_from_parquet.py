import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
import sys


def read_filter_parquet(filepath, extra_cols_to_read):
    """
    The script distinguishes segments that corresponds to breaks from terminal segments (not breaks)
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

def create_graph_from_parquet(file_paths,maxdist,theta):
    # file_paths is your list of Parquet file paths
    dfs = [read_filter_parquet(fp, extra_cols_to_read=['tBreak']) for fp in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    #sys.exit(df)
    # convert millis into days
    df['tBreak'] = df['tBreak'] / (1000 * 3600 * 24)  # number of days since 1970/01/01
    # Build KD-tree on spatial coordinates
    coords = df[['x_coord', 'y_coord']].values
    tree = cKDTree(coords)
    # Prepare graph
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx, tBreak=row['tBreak'], x_coord=row['x_coord'], y_coord=row['y_coord'])
    # For each point, find all spatial neighbors within maxdist
    pairs = tree.query_pairs(r=maxdist)  # Returns set of (i, j) with i < j
    # Add edges if temporal condition is also satisfied
    tBreaks = df['tBreak'].values
    is_break = df['is_break'].astype(bool).tolist()
    for i, j in pairs:
        if (is_break[i] and not is_break[j]) or (is_break[j] and not is_break[i]) or (is_break[i] and is_break[j] and abs(tBreaks[i] - tBreaks[j]) < theta):
            G.add_edge(i, j)
    return G, df
