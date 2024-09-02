# Algoritmo PyCCD

## Instalação do ambiente virtual
Para começar o processamento do algoritmo é necessário a instalação do ambiente virtual a partir dos ficheiros '.yml':

* WINDOWS: https://github.com/manuelcampagnolo/S2CHANGE/blob/main/ccdISA_win.yml
* LINUX: https://github.com/manuelcampagnolo/S2CHANGE/blob/main/ccdISA_linux.yml

1. Deve-se fazer o download e proceder à criação do ambiente virtual, através da linha de comandos:
conda env create -f environment.yml
2. Ativar o ambiente:
conda activate meu_ambiente
3. Verificar se o ambiente foi criado:
conda env list

## Inputs e outputs do algoritmo
Os inputs e outputs do algoritmo estão numa pasta partilhada do máquina do ISA, tendo o seguinte diretório: *C:\Users\Public\Documents*

### Inputs
* Imagens Sentinel-2 (Theia ou s2cloudless)
* Base de dados de referência (BDR-DGT ou BDR-Navigator)
* Nome do tile (T29TNE, T29SNB, ...)

Os inputs tem a seguinte configuração:
 Working directory (DADOS):
 |----FOLDER PUBLIC DOCUMENTS
    |---- SUBFOLDER BDR_300 (DGT)
         |---- file.shp
    |---- SUBFOLDER BDR_Navigator
         |---- file.gpkg
    |---- SUBFOLDER IMAGENS GEE
         |---- folder TILES
              |---- files.tif
    |---- SUBFOLDER IMAGENS THEIA
         |---- folder TILES
              |---- files.tif

### Outputs (outputs_BDR300 & outputs_BDR-NAV)
* Ficheiro numpy dos dados
* Plots (se pedido)
* CSV com os resultados do PyCCD / validação dos resultados (if BDR == DGT)
* Shapefiles com as datas de quebra e resultados do PyCCD

Os outputs tem a seguinte configuração:
 Working directory (DADOS):
 |----FOLDER PUBLIC DOCUMENTS
    |---- SUBFOLDER output_BDR300
         |---- folder numpy
              |---- files.npy
         |---- folder plots
              |---- plots.png
         |---- folder tabular (csv e validação)
              |---- files.csv
    |---- SUBFOLDER output_NAV
         |---- folder numpy
              |---- files.npy
         |---- folder plots
              |---- plots.png
         |---- folder tabular (csv)
              |---- files.csv

### Algoritmo PyCCD
O algoritmo é processado para o tile definido para cada um dos pontos dentro das geometrias dadas pela BDR.
O algoritmo é dividido por duas pastas: notebooks (que são scripts de apoio ao processamento) e ccd (onde contém o algoritmo todo do PyCCD).
Cada uma das pastas contém a seguinte configuração:
 Working directory (PyCCD):
 |----FOLDER CCD_yml_win
    |---- SUBFOLDER scripts
    |---- SUBFOLDER pyccd_theia
         |---- SUBFOLDER ccd
              |---- SUBFOLDER models
                   |---- __init__.py
                   |---- lasso.py
                   |---- robust_fit.py
                   |---- tmask.py
              |---- __init__.py
              |---- app.py
              |---- change.py
              |---- math_utils.py
              |---- parameters.py
              |---- procedures.py
              |---- qa.py
              |---- version.py
         |---- SUBFOLDER notebooks 
              |---- addNewImageToFile.py
              |---- avaliacao_exatidao_pyccd.py
              |---- main.py (** ficheiro principal **)
              |---- plot.py
              |---- processing.py
              |---- read_files.py
              |---- utils.py