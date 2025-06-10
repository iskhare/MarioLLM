import re
import pandas as pd


# Function to parse the entire log text and save to CSV
def parse_log_to_csv(log_text, output_filename="episode_reward_epsilon_full_test_10000_lv2.csv"):
    pattern = r"Episode:\s*(\d+)\s*Total reward:\s*([\d\.]+)\s*Epsilon:\s*([\d\.]+)"
    matches = re.findall(pattern, log_text)

    df = pd.DataFrame(matches, columns=["Episode", "Reward", "Epsilon"])
    df = df.astype({"Episode": int, "Reward": float, "Epsilon": float})
    df.to_csv(output_filename, index=False)
    return df


# Example usage:
# 1. Paste your full console output into a file named "log.txt" in the current directory.
# 2. Read that file and pass its content to the function.
with open("log_test_10000_lv2.txt", "r") as f:
    log_content = f.read()

# Parse and save to CSV
df_full = parse_log_to_csv(log_content)
df_full.head(10)  # Show first 10 rows as confirmation
