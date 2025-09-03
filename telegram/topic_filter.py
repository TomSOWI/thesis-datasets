import pandas as pd

#base_path = "/scratch/usr/nimtsspi"
base_path = "/mnt/vast-kisski/projects/kisski_tegami"
INPUT_PATH = f"{base_path}/datasets/TG"
OUTPUT_PATH = f"{base_path}/datasets/TG"

df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
relevant_topics = ['Religion','US news','Covid','World news','Extremists and radicals','Social']
print(df.columns)
print(df.topic.unique())

df = df[df.topic.isin(relevant_topics)]
print("N topic focussed:", len(df))
df.to_parquet(f"{OUTPUT_PATH}/TG_topic_augmented.parquet")



