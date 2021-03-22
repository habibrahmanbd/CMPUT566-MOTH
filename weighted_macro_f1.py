from tensorflow import keras
import math

def weight(t):
    return 1

def weighted_true_positive(true_positives):
    wtp = 0
    for t in true_positives:
        wtp += weight(t)
    return wtp

def weighted_false_negative(false_negative):
    wfn = 0
    for t in false_negative:
        wfn += weight(t)
    return wfn

def weighted_recall(weighted_true_positive, weighted_false_negative):
    weighted_recall_value = (weighted_true_positive / (weighted_true_positive + weighted_false_negative))
    return weighted_recall_value

def weighted_f1(s):
    return 1

def weighted_macro_f1(S):
    wmf1 = 0
    for s in S:
        wmf1 += weighted_f1(s)
    wmf1 /= abs(S)
    return wmf1



