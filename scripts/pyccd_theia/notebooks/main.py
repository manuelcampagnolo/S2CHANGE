import concurrent.futures
from processamento import processar_ponto  # Importe a função processar_ponto do arquivo processamento.py
import glob
import re
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime
import ccd 

# Leitura dos dados geoespaciais
caminho_arquivo = "C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_teste_buffer_metros_v2.gpkg"
dados_geoespaciais = gpd.read_file(caminho_arquivo)

tiles ="T29TNE"
number_tile = tiles
#%%
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Use o método submit para agendar a execução da função para cada ponto
    futures = [executor.submit(processar_ponto, k, dados_geoespaciais, number_tile) for k in range(len(dados_geoespaciais))]

    # Aguarde a conclusão de todas as execuções
    concurrent.futures.wait(futures)

    # Obtenha os resultados
    resultados = [future.result() for future in futures]