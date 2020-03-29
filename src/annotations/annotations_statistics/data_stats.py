import json
import os
import string

import pandas as pd
import numpy

from config import DISORDER, CHEMICAL_OR_DRUG

bdir = r'E:\mdtel_data\data\manual_labeled_v2\doccano\merged_output'


label_numbers_for_each_community = {"diabetes": {DISORDER: 9, CHEMICAL_OR_DRUG: 10},
                                    "sclerosis": {DISORDER: 7, CHEMICAL_OR_DRUG: 8},
                                    "depression": {DISORDER: 11, CHEMICAL_OR_DRUG: 12}}

def get_stats_for_community(community):
    df = pd.read_csv(bdir + os.sep + community + "_labels.csv")
    df['merged_inner_and_outer'] = df['merged_inner_and_outer'].apply(json.loads)
    df = df[['text', 'merged_inner_and_outer']]
    df['words_lst'] = df['text'].apply(lambda post_txt: [x for x in post_txt.split(" ") if x not in string.punctuation and x not in ['', ' ']])
    df['word_num_in_each_lst'] = df['words_lst'].apply(lambda x: len(x))
    df['annotated_terms'] = df['merged_inner_and_outer'].apply(lambda lst: len([x['term'] for x in lst]))
    df['disorders_terms'] = df['merged_inner_and_outer'].apply(lambda lst: len([x['term'] for x in lst if x['label'] == label_numbers_for_each_community[community][DISORDER]]))
    df['chemical_terms'] = df['merged_inner_and_outer'].apply(lambda lst: len([x['term'] for x in lst if x['label'] == label_numbers_for_each_community[community][CHEMICAL_OR_DRUG]]))

    # for lst in df['annotated_terms']:
    #     all_annotated_terms += lst
    #
    # number_of_annotated_terms = len(all_annotated_terms)
    # number_of_unique_annotated_terms = len(set(all_annotated_terms))

    all_words = []
    for x in df['words_lst'].values:
        all_words += x
    all_unique_words = len(set(all_words))
    all_total_words = len(all_words) # sanity

    number_of_posts = len(df)
    total_of_words = df['word_num_in_each_lst'].sum()
    print(f'all_total_words: {all_total_words}, total_of_words: {total_of_words}')
    average_words_in_each_post = round(df['word_num_in_each_lst'].mean(), 2)

    d = {'number_of_posts': number_of_posts, 'all_total_words': all_total_words,
         'all_unique_words': all_unique_words,
         'average_words_in_each_post': average_words_in_each_post,
         'number_of_annotated_terms': df['annotated_terms'].sum(),
         'number_of_disorders_terms': df['disorders_terms'].sum(),
         'number_of_chemical_terms': df['chemical_terms'].sum(),
         }
    ser = pd.Series(d, name=community)
    return ser


def main():
    sclerosis_stats = get_stats_for_community('sclerosis')
    depression_stats = get_stats_for_community('depression')
    diabetes_stats = get_stats_for_community('diabetes')

    stats_df = pd.DataFrame([sclerosis_stats, depression_stats, diabetes_stats], index=['sclerosis', 'depression', 'diabtes'])
    print(stats_df)

    total_number_of_posts = stats_df['number_of_posts'].sum()
    total_number_of_words = stats_df['all_total_words'].sum()
    total_number_of_unique_words = stats_df['all_unique_words'].sum()
    average_number_of_words_in_post = stats_df['average_words_in_each_post'].mean()
    total_number_of_annotated_terms = stats_df['number_of_annotated_terms'].sum()
    total_number_of_disorders_terms = stats_df['number_of_disorders_terms'].sum()
    total_number_of_chemical_terms = stats_df['number_of_chemical_terms'].sum()

    total_d = {'number_of_posts': total_number_of_posts,
               'all_total_words': total_number_of_words,
               'all_unique_words': total_number_of_unique_words,
               'average_words_in_each_post': average_number_of_words_in_post,
               'number_of_annotated_terms': total_number_of_annotated_terms,
               'number_of_disorders_terms': total_number_of_disorders_terms,
               'number_of_chemical_terms': total_number_of_chemical_terms,
               }
    total_ser = pd.Series(total_d)
    stats_df.loc['Total'] = total_ser

    print(f"total_number_of_posts: {total_number_of_posts}, "
          f"total_number_of_words: {total_number_of_words}, "
          f"total_number_of_unique_words: {total_number_of_unique_words}, "
          f"average_number_of_words_in_post: {average_number_of_words_in_post},"
          f"total_number_of_annotated_terms: {total_number_of_annotated_terms},"
          f"total_number_of_disorders_terms: {total_number_of_disorders_terms},"
          f"total_number_of_chemical_terms: {total_number_of_chemical_terms}")

    stats_df.to_excel('stats.xlsx')
    print("Done")
if __name__ == '__main__':
    main()