# Generic Buy Now, Pay Later Project

## Group Information
Group number: 16 <br />
Group members and student ID:
- Ke Zhang: 1053318
- Leyao Lyu: 1173989
- Seren Adachi: 1156523
- Xiangyi He: 1166146
- Ziyi Wang: 1166087

## Project Information
**Research Goal:** 
- Generate some suitable features from the given tables. Then establish a ranking system to select the top 100 merchants for partnership with the buy now pay later company. 
- Discover what feature(s) can greatly separate merchants that should and shouldn’t be accepted for partnership.

**Timeline:** The timeline for the research area is 28/02/2021 - 28/08/2022

## External Datasets links
- **Census data:** https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip
- **Postcode and SA2 data:** http://github.com/matthewproctor/australianpostcodes/zipball/master
- **File with 2016 SA2 info and 2021 SA2 info:** https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/correspondences/CG_SA2_2016_SA2_2021.csv

## Pipeline
1. **Please use this `Google Drive shared link` to download these two large file:**
    https://drive.google.com/drive/folders/1hebdc2wFIKHbXpnYWN7jeA_c8mP9-y14?usp=sharing (Accessable for unimelb email address)
    1. The `prediction` file needs to be downloaded under the `curated` file
    2. The `census` file needs to be downloaded under the `outer` file
2. **Please visit the `scripts` directory and run the files in order:**
    1. `requirements.txt`: install all the packages needed for the rest of the code
    2. `ETL_script.py`: 
        - read the files from the `data/tables` directory
        - preprocess the data, such as detect NULL values, delete outliers
        - generated some features and exported some tables to the `data/curated` directory
3. **Please visit the `models` directory and run the files in order:**
    1. `prediction.ipynb`: use multi-layer perceptron model to predict the next monthly sales for each merchant, then export the table to the `data/curated` directory 
4. **Please visit the `notebooks` directory and run the files in order:**
    1. `read_external_data.ipynb`: download the external datasets to `data/external` and `data/outer` directories
    2. `combine_table.ipynb`: read the files from the `data/curated` and `data/external` directory to curate a final dataset for the ranking system, export the final dataset (named `merchant_info.parquet`) to the `data/curated` directory
    3. `geovisualisation.ipynb`: draw some geographical maps from the external datasets in the `data/external` and `data/outer` directories
    4. `visualization.ipynb`: draw some plots from the final dataset and save them to the `plots` directory <br />
5. **Please visit the `models` directory again and run the files in order:**
    1. `fitting_model.ipynb`: 
        - use two different models and combined their ranking results to get the final ranking on the merchants, then select the top 100 merchants
        - grouped the merchants into 5 segments, select the top 10 merchants in each segment
        - export the files to the `data/curated` directory
        - draw the correlation graph between the merchants' final ranking and the individual ranking of the other features, and save the figures to the `plots` directory
    2. `findings.ipynb`:
        - Do more visualization and analysis, and save the figures to the `plots` directory
6. **Please visit the `notebooks` again directory and run the file:**
    1. `summary_notebook.ipynb`:
        - The final summary notebook which summaries all the processes and results

