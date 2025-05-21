May 2025

An attempt to use graphs and communities to classify events (tBreaks) produced by pyccd

Terminology:
1. Segment: segment estimated by CCD. The parameters of the harmonic model for the segment are available in the outputs of CCD (parquet files)
2. Terminal segment: last segment estimated by CCD that spans through the end of the time series. 
3. tBreak: for each segment it is the value of tBreak (in days from 1970/01/01); tBreak is only relevant for non terminal segments
4. is_break: equals True for non terminal segments and equals False for terminal segments
5. Event: tuple (x_ccord, y_coord, tBreak, is_break,...), where x_coord and y_coord are in CRS UTM 29N
6. G=(V,E) is a graph where V is a set of events, and E is a set of edges (i,j) that connect two events is a given condition is True. The condition involves
   - theta = maximum number of days between tBreak(i) and tBreak(j)
   - distmax = maximum distance in the xy space between events (i,j)
8. Community: a community of G returned by the Louvain algorithm for maximum modularity; communities are sets of events and communities do not overlap (i.e. an event belongs to only one community)

The tentative condition for an edge `(i,j)` is:
```
  (is_break[i] and not is_break[j]) or
  (is_break[j] and not is_break[i]) or
  (is_break[i] and is_break[j] and abs(tBreaks[i] - tBreaks[j]) < theta)
```
