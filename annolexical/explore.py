import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "/scratch/usr/nimtsspi/datasets/annolex"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/annolex"

def weak_label_distribution(df, label):

    print(f"Analyzing {label} distribution")
    
    # Show distribution
    counts = df[label].value_counts()
    print(counts)
    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"Distribution of {label}")
    plt.axis('equal') 
    # Save figure
    plt.savefig(f"src/{label}_dist.png")
    plt.close()

    # Show examples for each category
    categories = df[label].unique()
    n_excamples = 3
    for category in categories:
        print(f"Show {n_excamples} for each {category}")
        examples = df[df[label] == category].sample(n_excamples).text.to_list()
        # Print sentences
        [print(f"{idx+1}: {example}") for idx, example in enumerate(examples)]

def main():
    df = pd.read_parquet(f"{INPUT_PATH}/preclassify.parquet")
    labels = ["gender", "hate", "lexbias", "senti"]
    for label in labels:
        # Highlight distribution of each class
        weak_label_distribution(df, label)
        print("---------------------------")

if __name__ == "__main__":
    main()
    