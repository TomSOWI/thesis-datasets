# 📦 TGDataset Creation

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


### Running order
1. combine_batches.py (internet required for kaggle)
2. explore_character_len.py
3. unify_messages.py
4. topic_filter.py

--> tg-preprocess.sbatch


