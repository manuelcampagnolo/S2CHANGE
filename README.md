# S2CHANGE 

**Desenvolvimento de mapas de perdas recentes de floresta e mato em Portugal derivados de imagens de satélite**

Contrato N.º 3044 de cooperação entre a Direção Geral do Território e o Instituto Superior de Agronomia

## Referências

<details markdown="block">
<summary> Descrição do projeto e referências</summary>

* Gestor do Contrato: Hugo Costa, DGT
* Responsável pela execução: Manuel Campagnolo, CEF, DCEB, ISA
* Procedimento CEXC/2152/2023
* O contrato tem enquadramento no subprojecto «P1.5- Dados de deteção remota para a gestão
florestal» (DRFloresta) do projeto Agenda Transform (Agenda para a transformação digital das
cadeias de valor florestais numa economia portuguesa mais resiliente e hipocarbónica), apoiado
pelo Plano de Recuperação e Resiliência (PRR), sob o cabimento n.º CI42300913 e compromisso n.º
CI52301229
* O contrato tem por objecto a realização de atividades de investigação e
desenvolvimento (I&D) para o desenvolvimento de metodologias eficazes à escala nacional e
eficientes a nível computacional para a criação sistemática de um produto nacional em formato
vetorial de delimitação de manchas superiores a 0.5 ha de perda recente de floresta e mato com
base em análise automática de imagens de satélite
* Data de início: contrato assinado a 20 de outubro de 2023
* Centro de custos do ISA: 5207 (S2CHANGE)
</details>

## Apresentações e outros contributos

<details markdown="block">
<summary> Reunião DGT 20 fevereiro de 2024</summary>

* Apresentação Sara Caetano. Resultados que permitem comparar a aplicação da metodologia de deteção de alterações (CCD), com o algoritmo Python pyccd, a imagens Sentinel-2 obtidas no GEE (com máscara de nuvens produzida pelo algoritmo S2cloudness) com as imagens Sentinel-2 préprocessadas pela Theia: [ficheiro powerpoint](documents/PPT_CCD_20fev.pptx)
* Apresentação Inês Silveira sobre a base de dados de referência Navigator; análise em particular da distribuição de datas de cortes dentro do mesmo talhão e da possibilidade de associar um sub-talhão a cada data de corte; análise preliminar sobre a possibilidade de associar uma alteração de sinal a operações de rechega e outras [ficheiro pdf](documents/Apresentacao_BD_NVG_IS_20fev.pdf)

</details>

<details markdown="block">
<summary> Reunião DGT 23 de maio de 2024</summary>

* Apresentação Sara Caetano [ficheiro pdf](documents/PPT_CCD_23maio2024.pdf)
* Apresentação Inês Silveira [ficheiro pdf](documents/Apresentacao_DatasCorte_DGT_23maio2024.pdf)

</details>

<details markdown="block">
<summary> Reuniões DGT/INCD/LIP 19 de junho e 4 julho de 2024</summary>

* [Apresentação](documents/reuniao_DGT_4_julho_2024.pdf)

</details>



<details markdown="block">
<summary> Call for Advanced Computing Projects (4th ed) - Grant 2024.07034.CPCA.A0</summary>

* [Application](documents/Application_2024.07034.CPCA.pdf)
* [Allocated resources](documents/A0_project_resources.png)

</details>

<details markdown="block">
<summary> Call for Advanced Computing Projects (5th ed) - Grant 2024_10004_CPCA_A1</summary>

* [Application](documents/application_2024_10004_CPCA_A1.pdf)
* [Contract](documents/TA_2024_10004_CPCA_A1.pdf)

</details>

<details markdown="block">
<summary> Call for EuroHPC March 2025 </summary>

* [Application](documents/EHPC-DEV-2025D03-088_Campagnolo_Consolidated_online_forms_submetido.pdf)

</details>

## Tarefas

**Tarefa 1** – Seleção e justificação das metodologias a operacionalizar nas Tarefas 2, 3 e 4, dados de input, especificações técnicas dos outputs, e potenciais adaptações tecnológicas a implementar na cadeia de produção da DGT.

*Duração: Mês 1-6*

Entregáveis:
  * E1.1 – [Relatório com a descrição do problema, condicionantes, dados de input e especificações técnicas dos outputs](documents/Entregavel_1_1.pdf) (10 de dezembro de 2023).
  * E1.2 – [Relatório com seleção e justificação das metodologias a operacionalizar](documents/Entregavel_1_2.pdf) (1 de maio de 2024); [versão revista](documents/Entregavel_1_2_v2.pdf) (14 de maio de 2024)
  * E1.3 – [Relatório sobre potenciais adaptações tecnológicas a implementar na cadeia de produção da DGT](documents/Entregável_1.3_v3.pdf) (5 de julho de 2024).

**Tarefa 2** - Construção da uma base de dados de referência (BDR) para calibração e validação espacial e temporal das metodologias a operacionalizar com base em dados resultantes de interpretação de imagens aéreas e de satélite, do Instituto de Conservação da Natureza e Florestas (ICNF) e outras fontes consideradas relevantes.

*Duração: Meses 2-18*

Entregáveis:  
  * E2.1 – [Relatório com metodologia de criação da BDR](documents/Entregavel_2_1.pdf) (1 de maio de 2024); [versão revista](documents/Entregavel_2_1_v2.pdf) (14 de maio de 2024)
  * E2.2 – [Metadads Base de dados de referência em formato ESRI shapefile ou Geopackage para uma tile Sentinel-2 sobre Portugal Continental](documents/Entregavel_2_2_BDR_navigator_sentinel2_metadados_v4.pdf) (7 de julho de 2024).
  * E2.3 – Extensão da base de dados para outras regiões de Portugal Continental. Ver [BRD_NVG](https://github.com/manuelcampagnolo/S2CHANGE/tree/main/documents/reference_data/NVG) para acesso aos dados (protegido) e à documentação.

**Tarefa 3** – Adaptação e implementação operacional de uma metodologia automática com base em imagens de satélite para a criação sistemática de um produto nacional de delimitação de manchas vetoriais superiores a 0.5 ha de perda recente de floresta e mato, com uma periodicidade de pelo menos dois meses.

*Duração: Meses 9-20*

Entregáveis:
  * E3.1 – [Manual de utilização operacional da metodologia implementada na cadeia de produção da DGT](documents/Entregavel_3_1_manuel_utilizacao_pyccd_v1.pdf) (20 de outubro de 2024).
  * E3.2 – Demonstrador prático: mapas nacionais vetoriais a delimitar manchas de perda recente de floresta e mato superiores a 0.5 ha com uma frequência bimestral relativos a um período contínuo de dois anos entre 2023 e 2025.
  * E3.3 – Relatório de validação dos mapas nacionais.
  * E3.4 – Aplicação informática que possa ser integrada na cadeia de produção da DGT.

**Tarefa 4** – Adaptação e implementação operacional na cadeia de produção da DGT de uma metodologia automática com base em imagens de satélite para a identificação sistemática do agente causador das perdas recentes de floresta e mato delimitadas no produto da tarefa 3, com uma periodicidade de pelo menos dois meses.

*Duração: Meses 13-24*

Entregáveis:
  * E4.1 – Manual de utilização operacional da metodologia implementada na cadeia de produção da DGT.
  * E4.2 – Demonstrador prático: mapas nacionais a identificar o agente causador das perdas recentes de floresta e mato superiores a 0.5 ha produzidas na tarefa 3.
  * E4.3 – Relatório de validação dos mapas nacionais.
  * E4.4 – Aplicação informática que possa ser integrada na cadeia de produção da DGT. 

### Produtos

<details markdown="block">
<summary> Base de dados de referência </summary>

* [BRD_NVG](https://github.com/manuelcampagnolo/S2CHANGE/tree/main/documents/reference_data/NVG)

</details>


