import matplotlib.pyplot as plt
import html

#The problem is if there are 2 special characters in a row the functions doesn't really work 
def ultimately_unescape(s: str) -> str:
    unescaped = ""
    while unescaped != s:
        s = html.unescape(s)
        unescaped = html.unescape(s)
    
    return s


def plot_distribution(df, label_column, class_name=None):
    """
    Plots a pie chart showing the distribution of values in the given label column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the label column to analyze.
        class_name (str, optional): A human-readable label for the title. Defaults to the column name.
    """
    counts = df[label_column].value_counts()
    print("Label counts:\n", counts)

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    title = f"Distribution of {class_name or label_column} Labels"
    plt.title(title)
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def show_examples(df, label_column, n_examples=3, text_column="text"):
    """
    Displays n examples from each category in the label column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the column containing labels.
        n_examples (int): Number of examples to show per category.
        text_column (str): Name of the text column.
    """
    categories = df[label_column].unique()
    for category in categories:
        print(f"\nShowing {n_examples} example(s) for label: {category}")
        examples = (
            df[df[label_column] == category]
            .sample(n=n_examples, random_state=42)[text_column]
            .tolist()
        )
        for idx, example in enumerate(examples):
            print(f"  {idx + 1}: {example}")


def investigate_patterns(df, text_column="text", n_examples=3):
    patterns = {
        "HTTP": "http",
        "DOMAINS": "www",
        "BITLY URL": "bit.ly",
        "MARKDOWN": r"\*\*",
        "RETWEET": "RT",
        "MENTION": "@",
        "HTML": "&amp"
    }

    for pattern, regex in patterns.items():
        print(f"Investigating {pattern}...")
        pattern_df = df[df[text_column].str.contains(regex, regex=True, na=False)]
        share = round((len(pattern_df) / len(df)) * 100, 2)
        print(f"Occurrence of pattern: {len(pattern_df)}")
        print(f"That's {share}% of messages")
        examples = pattern_df.sample(n=min(n_examples, len(pattern_df)))[text_column].tolist()
        print(f"\nShowing {n_examples} example(s) for pattern: {pattern}")
        for idx, example in enumerate(examples):
            print(f"  {idx + 1}: {example}")
        
        print("\n-----------------")
