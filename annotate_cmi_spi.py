import pandas as pd
import numpy as np

# split the [0, 0.5] interval into 3 buckets.
cmi_dict = [0.17, 0.33]
input_file = "./data/input.txt"
output_file = "./data/input_cmi_spi.tsv"
np.random.seed(1)

def assign_discrete_cmi(sentence):
    num_tokens = len(sentence.split(' '))
    half_tokens = int(num_tokens/2)
    possible_cmi = [np.round(a/num_tokens, 3) for a in range(1, half_tokens+1)]
    if len(possible_cmi) > 0:
    	return np.random.choice(possible_cmi)
    else:
        return 0.0

def assign_spi(cmi_score):
    if cmi_score <= cmi_dict[0]:
        return np.random.uniform(0.0001, 0.4)
    elif cmi_score <= cmi_dict[-1]:
        return np.random.uniform(0.0001, 0.6)
    else:
        return np.random.uniform(0.01, 1.0)

if __name__ == "__main__":
    df = pd.read_csv(input_file, sep="\t", names=["source"])
    df["cmi"] = df["source"].apply(lambda x : assign_discrete_cmi(x))
    df["spi"] = df["cmi"].apply(lambda x : np.round(assign_spi(x), 3))
    df.to_csv(output_file, sep="\t", header=False, index=False)

