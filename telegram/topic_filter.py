import pandas as pd


INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"

df = pd.read_parquet(f"{INPUT_PATH}/TG_unified.parquet")
relevant_topics = ['Religion','US news','Covid','World news','Extremists and radicals','Social']
print(df.columns)
print(df.topic.unique())

df = df[df.topic.isin(relevant_topics)]
print("N topic focussed:", len(df))
df.to_parquet(f"{OUTPUT_PATH}/TG_topic_augmented.parquet")



