"""Correct codes can appear multple times.
AP_all is Average Precision (AP) calculated over all retrieved correct codes.
AP is Average Precision (AP) calculated over the first retrieved of each correct code only.
    We do bounded AP by letting the denominator be min(k, number of correct codes).
Average Reciprocal Rank of Correct codes (ARR)
Binary Normalised Discounted Cumulative Gain (NDCG)
    Demoninator for NDCG will be the ideal DCG, which is the DCG value when
    all correct codes are ranked at the top. This is defined per example and is
    IDCG.
"""

# pylint: disable=C0103

import math

####################
# Single Clerical Code
####################
correct_code = ["code1"]
"""
All retrieved codes are of length 8
AP_demonimator = min(len(correct_code), len(retrieved codes)) = 1


IDCG = (1/math.log2(2)) = 1
"""
IDCG = 1 / math.log2(2)

retrieved_codes_1 = [
    "code1",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]

"""
AP_all = (1/1)/1 = 1
AP = (1/1)/1 = 1
ARR = 1/1 = 1
NDCG = (1/math.log2(1 + 1))/IDCG = 1
"""
AP_all = 1
AP = 1
ARR = 1
NDCG = 1


retrieved_codes_2 = [
    "incorrect_code",
    "incorrect_code",
    "code1",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]

"""
AP_all = (1/3)/1 = 0.3333333333333333
AP = (1/3)/1 = 0.3333333333333333
ARR = (1/3)/1 = 0.3333333333333333
NDCG = (1/math.log2(1 + 3))/IDCG = 0.23463936301137822
"""
AP_all = 0.3333333333333333
AP = 0.3333333333333333
ARR = 0.3333333333333333
NDCG = 0.23463936301137822


####################
# Multiple Clerical Codes
####################
correct_codes = ["code1", "code2", "code3"]
"""
All retrieved codes are of length 8
AP_demonimator = min(len(correct_code), len(retrieved codes)) = 3
AP_all_denominator

IDCG = (1/math.log2(1 + 1) + 1/math.log2(1 + 2) + 1/math.log2(1 + 3))
     = 2.13093
"""
IDCG2 = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)

retrieved_codes_1 = [
    "code1",
    "code2",
    "code3",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]

"""
AP_all = (1/1 + 2/2 + 3/3)/3 = 1
AP = (1/1 + 2/2 + 3/3)/3 = 1
ARR = 1/1 + 1/2 + 1/3)/3 = 0.6111111111111111
NDCG = (1/math.log2(1 + 1) + 1/math.log2(1 + 2) + 1/math.log2(1 + 3))/IDCG2 = 1
"""
AP_all = 1
AP = 1
ARR = 0.6111111111111111
NDCG = 1


retrieved_codes_2 = [
    "code1",
    "code1",
    "code2",
    "code3",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]

"""
AP_all = (1/1 + 2/2 + 3/3 + 4/4)/4 = 1
AP = (1/1 + 2/3 + 3/4)/3 = 0.8055555555555555
ARR = (1/1 + 1/3 + 1/4)/3 = 0.5277777777777778
NDCG = (1/math.log2(1 + 1) + 1/math.log2(1 + 3) + 1/math.log2(1 + 4))/IDCG2 = 0.9060254355346823
"""
AP_all = 1
AP = 0.8055555555555555
ARR = 0.5277777777777778
NDCG = 0.9060254355346823


retrieved_codes_3 = [
    "code1",
    "code1",
    "code2",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]

"""
AP_all = (1/1 + 2/2 + 3/3)/3 = 1
AP = (1/1 + 2/3)/3 = 0.5555555555555555
ARR = (1/1 + 1/3 + 0)/3 = 0.4444444444444444
NDCG = (1/math.log2(1 + 1) + 1/math.log2(1 + 3))/IDCG2 = 0.7039180890341347
"""
AP_all = 1
AP = 0.5555555555555555
ARR = 0.4444444444444444
NDCG = 0.7039180890341347


retrieved_codes_4 = [
    "code1",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "code2",
    "incorrect_code",
]

"""
AP_all = (1/1 + 2/8)/2 = 0.625
AP = (1/1 + 2/8)/3 = 0.4166666666666667
ARR = (1/1 + 1/8 + 0)/3 = 0.375
NDCG = (1/math.log2(1 + 1) + 1/math.log2(1 + 8))/IDCG2 = 0.617319681505689
"""
AP_all = 0.625
AP = 0.4166666666666667
ARR = 0.375
NDCG = 0.617319681505689

retrieved_codes_5 = [
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "code1",
    "incorrect_code",
    "code2",
    "incorrect_code",
]
"""
AP_all = (1/6 + 2/8)/2 = 0.20833333333333331
AP = (1/6 + 2/8)/3 = 0.13888888888888887
ARR = (1/6 + 1/8 + 0)/3 = 0.09722222222222221
NDCG = (1/math.log2(1 + 6) + 1/math.log2(1 + 8))/IDCG2 = 0.31520141044913487
"""
AP_all = 0.20833333333333331
AP = 0.13888888888888887
ARR = 0.09722222222222221
NDCG = 0.31520141044913487

retrieved_codes_6 = [
    "incorrect_code",
    "code1",
    "code2",
    "code3",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]
"""
AP_all = (1/2 + 2/3 + 3/4)/3 = 0.6388888888888888
AP = (1/2 + 2/3 + 3/4)/3 = 0.6388888888888888
ARR = (1/2 + 1/3 + 1/4)/3 = 0.3611111111111111
NDCG = (1/math.log2(1 + 2) + 1/math.log2(1 + 3) + 1/math.log2(1 + 4))/IDCG2 = 0.7328286204777911
"""
AP_all = 0.6388888888888888
AP = 0.6388888888888888
ARR = 0.3611111111111111
NDCG = 0.7328286204777911

retrieved_codes_7 = [
    "incorrect_code",
    "code1",
    "incorrect_code",
    "incorrect_code",
    "code2",
    "incorrect_code",
    "code3",
    "incorrect_code",
    "incorrect_code",
]
"""
AP_all = (1/2 + 2/5 + 3/7)/3 = 0.44285714285714284
AP = (1/2 + 2/5 + 3/7)/3 = 0.44285714285714284
ARR = (1/2 + 1/5 + 1/7)/3 = 0.2809523809523809
NDCG = (1/math.log2(1 + 2) + 1/math.log2(1 + 5) + 1/math.log2(1 + 7))/IDCG2 = 0.6131471927654584
"""
AP_all = 0.44285714285714284
AP = 0.44285714285714284
ARR = 0.2809523809523809
NDCG = 0.6131471927654584

retrieved_codes_8 = [
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "code2",
    "incorrect_code",
    "code3",
    "incorrect_code",
    "incorrect_code",
]
"""
AP_all = (1/5 + 2/7)/2 = 0.24285714285714285
AP = (1/5 + 2/7)/3 = 0.24285714285714285
ARR = (0 + 1/5 + 1/7)/3 = 0.11428571428571428
NDCG = (1/math.log2(1 + 5) + 1/math.log2(1 + 7))/IDCG2 = 0.3379680345449381
"""
AP_all = 0.24285714285714285
AP = 0.24285714285714285
ARR = 0.11428571428571428
NDCG = 0.3379680345449381


retrieved_codes_9 = [
    "incorrect_code",
    "code1",
    "incorrect_code",
    "incorrect_code",
    "code2",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
    "incorrect_code",
]
"""
AP_all = (1/2 + 2/5)/2 = 0.45
AP = (1/2 + 2/5)/3 = 0.3
ARR = (1/2 + 1/5 + 0)/3 = 0.2333333333333333
NDCG = (1/math.log2(1 + 2) + 1/math.log2(1 + 5))/IDCG2 = 0.4776237035032179
"""
AP_all = 0.45
AP = 0.45
ARR = 0.2333333333333333
NDCG = 0.4776237035032179


"""
Main note is that AP_all will either be higher or the same as AP and will
perform better when correct codes appear
multiple times, while MAP focuses only on the first occurrence of each
correct code.

Should we segment the groups to 1 relevant item and multiple?
MAP scales linearly.
NDCG a missing item at the bottom of the list hurt significantly less
than a missing item at the top.
"""
