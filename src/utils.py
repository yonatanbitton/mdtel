import difflib
import string


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
