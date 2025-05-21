from pathlib import Path
import pandas as pd
import networkx as nx
import sys
import pickle

from create_graph_from_parquet import *
#from create_geodataframe import *
#from output_tiff import *
from concave_hull import concave_hull
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiPoint


DO_COMPUTE_GRAPH=True
DO_COMPUTE_COMMUNITIES=True
SAVE_COMMUNITIES=True
SAVE_GRAPH_DF=True
SAVE_ADJ=False
END2024=1735430400000 # in millis; 20086 dias desde 1970
NMIN=25 # minimum mumber of events per community; 0.5 ha=5000 m2 corresponds to 50 S2 pixels
MAXDIST=12 # max distance (meters) between two vertices to be connected
THETA=30 # max tBreak diference(days) to be connected (if both is_tBreak)
PARQUET_INPUT_FOLDER=r"C:\temp\s2change\S2CHANGE\scripts\networkx\parquet_files"
PARQUET_INPUT_FOLDER=r"C:\temp\s2change\S2CHANGE\scripts\networkx\BDRDGT300_0999_buffer_100m"
PREFIX=PARQUET_INPUT_FOLDER.split('\\')[-1]+f'_maxdist_{MAXDIST}_theta_{THETA}_nmin_{NMIN}'
FN_GPKG_OUTPUT=PREFIX+'.gpkg'
G_DF_PICKLE_OUTPUT='G_df_'+PREFIX+'.pkl'
COMM_PICKLE_OUTPUT='Comm_'+PREFIX+'.pkl'
print(FN_GPKG_OUTPUT,G_DF_PICKLE_OUTPUT,COMM_PICKLE_OUTPUT)

if DO_COMPUTE_GRAPH:
    folder_path = PARQUET_INPUT_FOLDER # Replace with your folder path
    parquet_files = list(Path(folder_path).glob('*.parquet'))
    # If you want string paths instead of Path objects:
    parquet_file_paths = [str(p) for p in parquet_files]
    # create graph from parquet files
    G,df = create_graph_from_parquet(parquet_file_paths,maxdist=MAXDIST, theta=THETA)
    # Get basic graph statistics
    print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    if SAVE_GRAPH_DF:
       with open(G_DF_PICKLE_OUTPUT, 'wb') as f:
            pickle.dump((G, df), f)
    if SAVE_ADJ:
    # save adj list
        nx.write_adjlist(G, "my_graph.adjlist")
    # To load:
    # G_loaded = nx.read_adjlist("my_graph.adjlist")
else:
    with open(G_DF_PICKLE_OUTPUT, 'rb') as f:
        G, df = pickle.load(f)


# communities
if DO_COMPUTE_COMMUNITIES:
    L=nx.community.louvain_communities(G, seed=123)
    if SAVE_COMMUNITIES:
        with open(COMM_PICKLE_OUTPUT, 'wb') as f:
            pickle.dump(L, f)
else:
    with open(COMM_PICKLE_OUTPUT, 'rb') as f:
        L = pickle.load(f)


print('Number of communities', len(L))
L=[c for c in L if len(c)>=NMIN]
print(f'Number of communities with more than {NMIN} pixels', len(L))

# create dataframe with the communitu index for each pixel
polygons=[]
averages=[]
min_tBreaks=[]
max_tBreaks=[]
for index,c in enumerate(L):
    if index % 10 == 0: print(f'community {index}/{len(L)}')
    cdf=df.iloc[list(c)]
    # average tBreak (days since 1970)
    average = cdf.loc[cdf['is_break'] == 1, 'tBreak'].mean()
    min_tBreak = cdf.loc[cdf['is_break'] == 1, 'tBreak'].min()
    max_tBreak = cdf.loc[cdf['is_break'] == 1, 'tBreak'].max()
    # tried geoseries and concave_hall
    #gs = gpd.GeoSeries.from_xy(cdf['x_coord'], cdf['y_coord'], crs="EPSG:32629")
    #sys.exit(gs.union_all().concave_hull(ratio=0.5))  <<<<<<<<<<<<<<<<<<<<<<<<<<< 'MultiPoint' object has no attribute 'concave_hull'. Did you mean: 'convex_hull'?
    #
    # perplexity claims that concave_hull implements the fast concaveman algorithm, but it could be safer to run python up to 3.10
    # the input is a np.array
    xy=cdf[['x_coord', 'y_coord']].to_numpy()
    hull = concave_hull(xy, concavity=2.0, length_threshold=0.0)
    #print(hull)
    # Suppose 'hull' is your NumPy array of (x, y) coordinates forming the hull
    polygons.append(Polygon(hull))
    averages.append(average)
    min_tBreaks.append(min_tBreak)
    max_tBreaks.append(max_tBreak)
# Create a GeoDataFrame with the correct CRS
gdf = gpd.GeoDataFrame({'geometry': polygons, 'tBreak_mean': averages, 'tBreak_min': min_tBreaks, 'tBreak_max': max_tBreaks}, crs="EPSG:32629")
gdf.to_file(FN_GPKG_OUTPUT, driver="GPKG")
