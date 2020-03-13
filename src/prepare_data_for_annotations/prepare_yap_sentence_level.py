from config import data_dir
from contextual_relevance.extract_dataset_with_feats.extract_relatedness_features import get_words_and_lemmas
from contextual_relevance.extract_dataset_with_feats.yap.yap_api import YapApi
import os
import pandas as pd

raw_tokenized_dir = data_dir + r"high_recall_matcher\posts\lemlda\raw_tokenized"
output_dir = data_dir + r"high_recall_matcher\posts\lemlda"

yap = YapApi()
ip = '127.0.0.1:8000'

general_count = 0

def parse_community(community):
    comm_dir = raw_tokenized_dir + os.sep + community
    all_df_rows = []
    for fname in os.listdir(comm_dir):
        fpath = comm_dir + os.sep + fname
        handle_fpath(fpath, fname.split(".txt")[0], all_df_rows)

    columns = ['file_name', 'sent_idx', 'sent_words_and_lemmas', 'sent_txt']
    comm_df = pd.DataFrame(all_df_rows, columns=columns)
    comm_df.to_excel(output_dir + os.sep + community + "_sentences.xlsx", index=False)
    print(f"Finished with community {community}")


def handle_fpath(fpath, file_name, all_df_rows):
    global general_count
    with open(fpath, encoding='utf-8') as f:
        post_txt_lines = [x.rstrip('\n') for x in f.readlines()]
    post_txt_lines = [l for l in post_txt_lines if l != '']
    post_txt_dots = " .".join(post_txt_lines)
    if '\xa0' in post_txt_dots:
        post_txt_dots = post_txt_dots.replace('\xa0', ' ')
    post_txt_sentences = [x for x in post_txt_dots.split(".") if x != '' and x != ' ' and x != '  ']
    for sent_idx, sent_txt in enumerate(post_txt_sentences):
        sent_words_and_lemmas = get_words_and_lemmas(sent_txt)
        sent_d = {'file_name': file_name, 'sent_idx': sent_idx, 'sent_words_and_lemmas': sent_words_and_lemmas, 'sent_txt': sent_txt}
        general_count += 1
        if general_count % 50 == 0 or "שלום שושי" in post_txt_dots:
            if "שלום שושי" in post_txt_dots:
                print(f"שלום שושי!!!")
            print(sent_d)
        all_df_rows.append(sent_d)


def main():
    # fpath = r"E:\mdtel_data\data\high_recall_matcher\posts\lemlda\raw_tokenized\sclerosis\15.txt"
    # handle_fpath_newlines(fpath)
    parse_community('sclerosis')
    parse_community('diabetes')
    parse_community('depression')

if __name__ == '__main__':
    main()
