# 📦 TG Dataset Creation

## 📥 How to Download the Data

- **Clone the repository**  
  https://github.com/TomSOWI/thesis-datasets

- **Download the dataset**  
  From Zenodo: https://zenodo.org/records/7640712

- **Save the downloaded `.json` files**  
  Place them under:  
  `telegram/mongodb/public_db/`

- **Upload the `.json` documents to MongoDB**  
  Use:  
  `mongodb/db_utilities.py`

- **Store English messages as `.parquet`**  
  Use:  
  `mongodb/write_mongodb_to_parquet.py`

- **Combine batches or upload to a remote server (if memory is an issue)**  
  - Use: `combine_batches.py`  
  - Appends topic labels from: `channel_topic_mappings.csv`

- **Final dataset location**  
  Saved under:  
  `datasets/TG/TG_base.parquet`

---

## 🧹 How to Process the Data

- **Explore message length distribution**  
  → `explore_character_len.py`

- **Filter out messages longer than 280 characters**  
  → `explore_character_len.py`

- **Unify and deduplicate messages**  
  → `unify_messages.py`  
  - Evaluate unification: `test/unify_verify.py`

- **Pre-classify dataset to assign weak labels**  
  → See `pre-classify/` folder

- **Select and filter by topics**  
  → `topic_filter.py`

- **Save full dataset to Hugging Face Hub**  
  → Name: `TG-all-topics`

- **Filter for core topics only**  
  → Save as: `TG-core-topics`

- **Pre-classify for weak labels** 
  → Save as: `labels/TG-gender`
  → Save as: `labels/TG-hate`
  → Save as: `labels/TG-sentiment`
  → Save as: `labels/TG-lexbias`

- **Combine labels** 
 Using pre-classify/summary.ipynb
  → All topics: `TG-weak-label-all`
  → Core topics: `TG-weak-label-core`

- **Downsample based on weak labels tasks for each topic**  
Using task_downsampling.ipynb
  → Save as: `TG-weak-gender-core`
  → Save as: `TG-weak-hate-core`
  → Save as: `TG-weak-sentiment-core`
  → Save as: `TG-weak-lexbias-core`


### Running order
1. combine_batches.py (internet required for kaggle)
2. explore_character_len.py
3. unify_messages.py
4. topic_filter.py
5. pre-classify/pre-classify.sbatch
6. task_downsampling.ipynb
--> tg-preprocess.sbatch


