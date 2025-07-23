# 📊 Annolex Processing

## 🛠️ Preparing the Dataset

1. **Download the dataset from Hugging Face**  
   Run: `download_annolex.py`  
   → Saves to: `datasets/annolex/base.parquet`

2. **Add weak labels for classification tasks**  
   Run: `process.py`

3. **Explore and analyze the weak labels**  
   Run: `explore.py`

---

## 🧾 Recommended Execution Order

```bash
# 1. Download the dataset (requires internet)
python download_annolex.py

# 2. Add weak supervision labels
python process.py

# 3. Analyze label distributions and results
python explore.py
