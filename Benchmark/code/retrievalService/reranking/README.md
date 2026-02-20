### Setup
1) The setup has been tested on Python version 3.9
(a) Setup a conda environment using environment.yml file 
or 
(b) Install LAVIS from source alond with its dependencies as given in https://github.com/salesforce/LAVIS/tree/main?tab=readme-ov-file#installation. 

2) Activate environment

3) Start the (FAST API) service
``` uvicorn query_rerank_service:app --reload --port 8083 ```
