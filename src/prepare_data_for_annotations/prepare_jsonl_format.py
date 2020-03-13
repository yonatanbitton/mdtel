
import pandas as pd
import json
from config import data_dir
import os
input_dir = data_dir + r"high_recall_matcher\posts\lemlda"
output_dir = input_dir + os.sep + "jsonl"

def handle_comm(comm):
    comm_df = pd.read_excel(input_dir + os.sep + comm + "_posts.xlsx")
    with open(output_dir + os.sep + comm + "_dicts.txt", 'w', encoding='utf-8') as f:
        for row_idx, row in comm_df.iterrows():
            row_d = {'text': row['post_txt'], 'file_name': row['file_name'], 'post_words_and_lemmas': row['post_words_and_lemmas']}
            f.write(json.dumps(row_d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    handle_comm('sclerosis')
    handle_comm('diabetes')
    handle_comm('depression')