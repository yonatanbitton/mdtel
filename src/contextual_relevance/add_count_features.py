import os
import pandas as pd
from collections import Counter

wiki_data_dir = r"E:\mdtel_data\data\contextual_relevance\wiki_data.txt"

input_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\dataset_for_lm_probs'
output_dir = r'D:\ThesisResources\OHCsProject_Resources\Camoni\medical_terms\translation\detected_umls\dataset_for_lm_probs\with_count_features'

def prepare_wiki_data_counter():
    train_data_path = wiki_data_dir + os.sep + 'train' + os.sep + 'train.txt'
    valid_data_path = wiki_data_dir + os.sep + 'valid' + os.sep + 'valid.txt'
    with open(train_data_path, encoding='utf-8') as f:
        train_lines = [x.rstrip('\n') for x in f.readlines()]
    with open(valid_data_path, encoding='utf-8') as f:
        valid_lines = [x.rstrip('\n') for x in f.readlines()]
    wiki_lines = train_lines + valid_lines

    all_words = []
    for line in wiki_lines:
        line_words = line.split(' ')
        all_words += line_words

    all_words_counter = Counter(all_words)

    print(f"got {len(all_words)} words, of which {len(set(all_words))} unique.")
    return all_words_counter

all_words_counter = prepare_wiki_data_counter()

number_of_unique_tokens = len(all_words_counter)

def handle_community(community):
    print(f"community: {community}")
    comm_df = pd.read_excel(input_dir + os.sep + community + ".xlsx")
    all_match_counts = []
    all_match_freqs = []
    for row_idx, row in comm_df.iterrows():
        match_count = all_words_counter[row['match']]
        match_freq = match_count / number_of_unique_tokens
        all_match_counts.append(match_count)
        all_match_freqs.append(match_freq)
        if row_idx % 10 == 0:
            print(row['match'], match_count, match_freq)
    comm_df['match_count'] = all_match_counts
    comm_df['match_freq'] = all_match_freqs
    comm_df.to_excel(output_dir + os.sep + community + ".xlsx", index=False)

# handle_community("diabetes")
# handle_community("sclerosis")
handle_community("depression")

print("Done")