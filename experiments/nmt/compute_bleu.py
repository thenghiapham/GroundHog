'''
Created on Jun 30, 2015

@author: nghia
'''

import sys
from nltk.align import bleu

def get_bleu_score(out_file, gold_file):
    weights = [0.25, 0.25, 0.25, 0.25]
    sum_bleu = 0
    sent_num = 0
    with open(out_file) as o_stream, open(gold_file) as gold_stream:
        for out_sentence, gold_sentence in zip(o_stream, gold_stream):
            sum_bleu += bleu(out_sentence, [gold_sentence], weights)
            sent_num += 1
    return sum_bleu / sent_num

if __name__ == "__main__":
    argv = sys.argv[1:]
    out_file = argv[0]
    gold_file = argv[1]
    print get_bleu_score(out_file, gold_file)