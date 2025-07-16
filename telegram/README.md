# TGDataset creation

## How to download the data
1. Clone this github repo: https://github.com/SystemsLab-Sapienza/TGDataset
2. Do the following using this repo:
- Download the dataset as json files from here: https://zenodo.org/records/7640712#.Y-9PjNLMKXI
- Extract files in the folder public_db
- Save all json files as documents in mongoDB using **db_utilities.py**
3. Clone this repo: https://github.com/TomSOWI/thesis-datasets
4. Store all messages from english-speaking channels as parquet using **write_mongodb_to_parquet.py**
- If you can have enough memory on your device you can concate all batch files and **merge them with channel_topic_mappings.csv** and move one
- Else you can:
    - upload these file kaggle 
    - download these files on a remote-server
    - concatenate these files and add topic labels using **combine-batches.py**
--> The base version of the dataset is stored under **datasets/TG/TG_base.parquet**

## How to process the data
1. Explore the character length distribution of messages
2. Remove messages extending 280 characters
3. Unify messages and remove duplicates
4. Pre-Classify the dataset to get weak labels to obtain an overview of class distribution
5. Save the current dataset to HF as TG-all-topics
6. Filter messages from core topics and save dataset as TG-core-topics


