import difflib
import re
import string
from collections import Counter

from config import DISORDER, CHEMICAL_OR_DRUG


def word_is_english(word):
    for c in word:
        if 'a' <= c <= 'z' or 'A' <= c <= 'C':
            return True
    return False


def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


def replace_puncs(post_txt):
    puncs = list(string.punctuation)
    puncs += ['\n', ]
    for p in puncs:
        post_txt = post_txt.replace(p, ' ')
    return post_txt


class SequenceTagger:
    def __init__(self, community, predictor=False):
        self.community = community
        self.predictor = predictor
        self.SIMILARITY_THRESHOLD = 0.72

    def get_bio_tags(self, row, label_col):
        file_name = row['file_name']

        labels = row[label_col]
        text = replace_puncs(row['text'])

        text_words = [w for w in text.split(" ")]

        all_post_d = self.start_with_naive_tag(text_words)

        tokenization_problems = []
        all_words_indices_labels = {}
        for l in labels:
            l['term'] = l['term'].strip().replace("-", " ")
            semantic_type = self.get_semantic_type(l)
            labeled_words_indices = self.get_words_and_indices_at_txt_words(l, text, text_words, semantic_type)
            if labeled_words_indices == {0: {}}:
                tokenization_problems.append(l)
                continue
            all_words_indices_labels = {**all_words_indices_labels, **labeled_words_indices}

        for word_idx, word_w_and_label in all_words_indices_labels.items():
            all_post_d[word_idx] = word_w_and_label

        words_and_tags = list(zip([x['w'] for x in all_post_d.values()], [x['tag'] for x in all_post_d.values()]))
        return {'words_and_tags': words_and_tags, 'tokenization_problems': tokenization_problems}

    def start_with_naive_tag(self, text_words):
        all_post_d = {}
        all_indices = list(range(len(text_words)))
        all_tags = ['O' for _ in all_indices]
        for idx, w, tag in zip(all_indices, text_words, all_tags):
            all_post_d[idx] = {'w': w, 'tag': tag}
        return all_post_d

    def get_words_and_indices_at_txt_words(self, l, text, text_words, semantic_type):
        start_offset = l['start_offset']

        term = l['term']

        post_prefix = text[:start_offset + 1]

        indices_of_spaces = [m.start() for m in re.finditer(' ', post_prefix)]
        if len(indices_of_spaces) > 0:
            pairs = list(zip(indices_of_spaces, indices_of_spaces[1:]))
            pairs.insert(0, (-1, indices_of_spaces[0]))
            all_preceding_ws = [text[i1 + 1:i2] for i1, i2 in pairs]
            last_w_before_t = all_preceding_ws[-1]
        else:
            all_preceding_ws = []
        last_w_before_t_idx = len(all_preceding_ws) - 1

        term_indices, term_join_words_lst, term_joint_words = self.get_indices(l, last_w_before_t_idx, text_words)
        sim = words_similarity(term, term_joint_words)
        if not sim >= self.SIMILARITY_THRESHOLD:
            # print(f'SIM not passed - {sim}')
            # print(term)
            # print(term_joint_words)
            return {0: {}}

        words_indices_labels = self.add_tags(semantic_type, term_indices, term_join_words_lst)

        return words_indices_labels

    def add_tags(self, semantic_type, term_indices, term_join_words_lst):
        words_and_indices = dict(zip(term_indices, term_join_words_lst))
        for idx, word_idx in enumerate(words_and_indices):
            if idx == 0:
                label_for_w = 'B-' + semantic_type[0]
            else:
                label_for_w = 'I-' + semantic_type[0]
            words_and_indices[word_idx] = {'w': words_and_indices[word_idx], 'tag': label_for_w}
        return words_and_indices

    def get_indices(self, l, last_w_before_t_idx, text_words):
        term_num_words = len([x for x in l['term'].split(" ") if x != ''])
        start_idx_of_l = last_w_before_t_idx + 1
        end_idx_of_l = start_idx_of_l + term_num_words
        term_indices = range(start_idx_of_l, end_idx_of_l)  # inclusive
        term_join_words_lst = [text_words[i] for i in term_indices]
        term_joint_words = " ".join(term_join_words_lst)
        for seperator in ['-', '.']:
            if Counter(term_joint_words)[seperator] == 1:
                term_joint_words = term_joint_words.split(seperator)[0]
        return term_indices, term_join_words_lst, term_joint_words

    def get_semantic_type(self, l):
        semantic_type = l['label']
        return semantic_type

