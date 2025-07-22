import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.utils import to_parquet


INPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
OUTPUT_PATH = "/scratch/usr/nimtsspi/datasets/TG"
tqdm.pandas()


def plot_distribution(df, output_dir="src"):
    # KDE Plot
    kde = sns.displot(df, x="char_len", kind="kde")
    kde.figure.suptitle("Character Length Distribution (KDE)")
    kde.set_axis_labels("Character length", "Density")
    # Adjust axes
    for ax in kde.axes.flat:
        ax.axvline(x=280, color='red', linestyle='dotted', linewidth=1.5)
        ax.set_xlim(0, 2000)
        
    kde.figure.tight_layout()
    kde.savefig(f"{output_dir}/character_length_kde.png")
    plt.close()

    # Histogram Plot
    hist = sns.displot(df, x="char_len")
    hist.figure.suptitle("Character Length Distribution (Histogram)")
    hist.set_axis_labels("Character length", "Count")
    hist.figure.tight_layout()
    hist.savefig(f"{output_dir}/character_length_hist.png")
    plt.close()
    

@to_parquet(f"{OUTPUT_PATH}/TG_280limit.parquet")
def main():
    df = pd.read_parquet(f"{INPUT_PATH}/TG_base.parquet")

    # Calculate charater length per message
    df["char_len"] = df["message"].progress_apply(len)

    # Print summary statistics of message length
    summary = df["char_len"].describe()
    print("Summary statistics for message's character length")
    print(summary)

    plot_distribution(df)

    # Apply charater length limit
    df_280len = df[df.char_len <= 280]

    # Highlight discarded messages
    print(f"N messages base: {len(df)}")
    print(f"N messages characater-limit of 280 applied: {len(df_280len)}")
  
    return df_280len

if __name__ == "__main__":
    main()