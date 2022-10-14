# Generic Buy Now, Pay Later Project

## Group Information
Group number: 16 <br />
Group members and student ID:
- Ke Zhang: 
- Leyao Lyu: 1173989
- Seren Adachi: 
- Xiangyi He: 
- Ziyi Wang: 

## Project Information
**Research Goal:** 
- Generate some suitable features from the given tables. Then develop a ranking systme to select the top 100 merchants for partership with buy now pay later company. 
- Discover what feature(s) can greatly separated merchants that should and shouldn’t be accepted for partnership.

**Timeline:** The timeline for the research area is 28/02/2021 - 28/08/2022

## External Datasets links
- **Census data:** https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip
- **Postcode and SA2 data:** http://github.com/matthewproctor/australianpostcodes/zipball/master
- **File with 2016 SA2 info and 2021 SA2 info:** https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/correspondences/CG_SA2_2016_SA2_2021.csv

## Pipeline
1. **Please visit the `scripts` directory and run the files in order:**
    1. `requirements.txt`: install all the package needed for the rest of code
    2. `ETL script.py`: 
        - read the files from `data/tables` directory
        - preprocess the data, such as detect NULL values, delete outliers
        - generated some features and export some tables to `data/curated` directory
        - download the external datasets to `data/external` and `data/outer` directories
2. **Please visit the `models` directory and run the files in order:**
    1. `prediction.ipynb`: use multi-layer perceptron model to predict the next monthly sales for each merchant, then export the table to `data/curated` directory 
3. **Please visit the `models` directory and run the files in order:**
    1. `Combine_table.ipynb`: read the files from `data/curated` directory to curate a final dataset for the ranking system, export the final dataset (named 'merchant_info') to `data/curated` directory
    2. `geovisualisation.ipynb`: draw some geographical maps from the external datasets in `data/external` and `data/outer` directories
    3. `Visualization.ipynb`: draw some images from the final dataset and save them to `plots` directory <br />
4. **Please visit the `models` directory again and run the files in order:**
    1. `Fitting_model.ipynb`: 
        - use two different models and combined their ranking result to get the final ranking on the merchants, then select the top 100 merchants
        - grouped the merchants into 5 segments, select the top 10 merchants in each segments
        - export the files to `data/curated` directory
        - draw the correlation graph between the merchants' final ranking and the individual ranking of the other features, save the figure to `plots` directory
