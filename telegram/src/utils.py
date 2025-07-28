import functools
import pandas as pd
import html

def to_parquet(filename):
    def decorator_to_parquet(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            result.to_parquet(filename)

            return result

        return wrapper

    return decorator_to_parquet

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


#The problem is if there are 2 special characters in a row the functions doesn't really work 
def ultimately_unescape(s: str) -> str:
    unescaped = ""
    while unescaped != s:
        s = html.unescape(s)
        unescaped = html.unescape(s)
    
    return s
