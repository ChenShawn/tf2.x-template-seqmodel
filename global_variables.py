import json


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


class GlobalVariables(object):
    def __init__(self, json_dir='./data/info.json', label_name='amount'):
        with open(json_dir, 'r') as fd:
            self.jsoninfo = json.load(fd)
        self.cols = list(self.jsoninfo['catcols'].keys()) + list(self.jsoninfo['numcols'].keys())
        # self.cols.remove('year')
        if 'avg_loadfactor' in self.cols:
            self.cols.remove('avg_loadfactor')
        if 'amount' in self.cols:
            self.cols.remove('amount')
        self.cols.append(label_name)


# use this???
LABEL_NAME = 'amount'

GVARS = GlobalVariables()
CATEGORIES = GVARS.jsoninfo['catcols']
MEANVARS = GVARS.jsoninfo['numcols']
MEANVARS.pop(LABEL_NAME)
COLUMNS = GVARS.cols