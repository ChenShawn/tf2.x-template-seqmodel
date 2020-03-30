
# feature dimensionality
SEQLEN = 7
EACH_FLIGHT_DIM = 1
FLIGHT_GLOBAL_DIM = 1
GLOBAL_FEATURE_DIM = 1
EACH_DAY_DIM = 3 * EACH_FLIGHT_DIM + FLIGHT_GLOBAL_DIM
SEQ_DIM = SEQLEN * EACH_DAY_DIM
TOTAL_DIM = SEQ_DIM + GLOBAL_FEATURE_DIM
# split dimensionality
# GLOBAL_SPLITS = [EACH_DAY_DIM] * SEQLEN + [GLOBAL_FEATURE_DIM]
# SEQ_SPLITS = [EACH_FLIGHT_DIM] * 3 + [FLIGHT_GLOBAL_DIM]
EACH_DAY_DIM = 14
GLOBAL_SPLITS = [EACH_DAY_DIM] * SEQLEN + [12]

# LABEL_NAME = 'amount'
# # Define categorical features here, first line is an example
# COLUMNS = [
#     'cityid', 'diff', 'cancel_num', 'flight_num',
#     'bkd_acc_c_sum', 'bkd_acc_c_real_sum', 'bkd_acc_q_sum',
#     'bkd_acc_q_real_sum', 'income_acc_c_real_avg', 'all_seats', 'direction',
#     'uv', 'pv_1', 'pv_2', 'amount'
# ]
# CATEGORIES = {
#     'cityid': list(range(12)),
#     'direction': [u'进港', u'出港'],
# }
# MEANVARS = {}
# for col in COLUMNS:
#     # label must not be in the preprocessing list
#     if col != 'cityid' and col != 'direction' and col != 'amount':
#         MEANVARS[col] = [500.0, 500.0]


LABEL_NAME = 'amount'
# Define categorical features here, first line is an example
oldcols = [
    'diff', 'cancel_num', 'flight_num',
    'bkd_acc_c_sum', 'bkd_acc_c_real_sum', 'bkd_acc_q_sum',
    'bkd_acc_q_real_sum', 'income_acc_c_real_avg', 'all_seats', 'direction',
    'uv', 'pv_1', 'pv_2'
]
COLUMNS = ['cityid', 'amount']
for day in range(1, 8):
    for col in oldcols:
        COLUMNS.append('day_{}_{}'.format(day, col))

CATEGORIES = {'cityid': list(range(12))}
for day in range(1, 8):
    CATEGORIES['day_{}_direction'.format(day)] = [u'进港', u'出港']
MEANVARS = {}
for col in COLUMNS:
    if 'cityid' not in col and 'direction' not in col and 'amount' not in col:
        MEANVARS[col] = [500.0, 500.0]