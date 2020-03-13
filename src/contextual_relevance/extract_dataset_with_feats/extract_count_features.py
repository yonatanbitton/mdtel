import os
import pandas as pd
from collections import Counter
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir

wiki_data_dir = data_dir + r"contextual_relevance\wiki_data.txt"

input_dir = data_dir + r"contextual_relevance\initialized_training_dataset"

output_dir = data_dir + r"contextual_relevance\count_features"

def prepare_wiki_data_counter():
    with open(wiki_data_dir, encoding='utf-8') as f:
        lines = [x.rstrip('\n') for x in f.readlines()]

    all_words = []
    for line in lines:
        line_words = line.split(' ')
        all_words += line_words

    all_words_counter = Counter(all_words)

    print(f"got {len(all_words)} words, of which {len(set(all_words))} unique.")
    return all_words_counter


def handle_community(community):
    print(f"community: {community}")
    comm_df = pd.read_csv(input_dir + os.sep + community + ".csv")

    all_match_counts = []
    all_match_freqs = []
    for row_idx, row in comm_df.iterrows():
        match_count = all_words_counter[row['umls_match']]
        match_freq = match_count / number_of_unique_tokens
        all_match_counts.append(match_count)
        all_match_freqs.append(match_freq)
        if row_idx % 100 == 0:
            print(f"match: {row['umls_match']}, count: {match_count}, freq: {match_freq}")
    comm_df['match_count'] = all_match_counts
    comm_df['match_freq'] = all_match_freqs
    comm_df.to_csv(output_dir + os.sep + community + "_output.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    all_words_counter = prepare_wiki_data_counter()
    number_of_unique_tokens = len(all_words_counter)

    handle_community("sclerosis")
    handle_community("diabetes")
    handle_community("depression")

    print("Done")