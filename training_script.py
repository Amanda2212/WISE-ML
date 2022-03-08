#!/usr/bin/env python

# general imports:
from astropy.table import Column, Table, join
from matplotlib import pyplot as plt
from astropy.io import fits
import pandas as pd
import numpy as np

from wise_ml.models import predictors
from wise_ml.models import training
from wise_ml.models import misc_functions

DESC='''Script to train and save models, by first using full redshift range, excluding predicted values above 2 and 1.5, then training two new predictors using these new subsets. The train_standard_architecture function performs both the training and the model saving.'''

# 4 band training photometric data from r90:
d4_full = pd.read_pickle('/Users/amandascbeck/Desktop/RESEARCH/AGN_Project/WISE-ML/wise_ml/data/train_test_W1W2W3W4.pkl')
# 2 band training photometric data from r90:
d2_full = pd.read_pickle('/Users/amandascbeck/Desktop/RESEARCH/AGN_Project/WISE-ML/wise_ml/data/train_test_W1W2.pkl')

# excluding all redshifts above 2 and 1.5:
d4_2 = d4_full[d4_full['REDSHIFT']<=2.0]
d4_15 = d4_full[d4_full['REDSHIFT']<=1.5]
d2_2 = d2_full[d2_full['REDSHIFT']<=2.0]
d2_15 = d2_full[d2_full['REDSHIFT']<=1.5]

# # specifying max redshift range:
# redshift_max_range = [0, 8.0]
# specifying 2 redshift range:
redshift_2_range = [0, 2.0]
# specifying 1.5 redshift range:
redshift_15_range = [0, 1.5]

# # training model for 4 band predictions:
# predict_redshift = training.train_standard_architecture(d4, redshift_max_range, n_inputs=4)
# # training model for 2 band predictions:
# predict_redshift = training.train_standard_architecture(d2, redshift_max_range, n_inputs=2)

# 4 band:

# # predict redshift for 4 bands using max range:
# r4_max = predictors.predict_redshift(d4.iloc[:, 2:9:2],
#                                      redshift_range = redshift_max_range,
#                                      n_inputs = 4)
# # putting it all together:
# d4['PREDICTED_MAX_REDSHIFT'] = r4_max
# # excluding values above z=2:
# d4_2 = d4[d4['PREDICTED_MAX_REDSHIFT']<=2.0]
# # excluding values above z=1.5:
# d4_15 = d4[d4['PREDICTED_MAX_REDSHIFT']<=1.5]
# training 2 predictor:
predict_redshift = training.train_standard_architecture(d4_2, redshift_2_range, n_inputs=4)
# training 1.5 predictor:
predict_redshift = training.train_standard_architecture(d4_15, redshift_15_range, n_inputs=4)
# final predictions and plots:
r4_2 = predictors.predict_redshift(d4_2.iloc[:, 2:9:2],
                                     redshift_range = redshift_2_range,
                                     n_inputs = 4)
r4_15 = predictors.predict_redshift(d4_15.iloc[:, 2:9:2],
                                     redshift_range = redshift_15_range,
                                     n_inputs = 4)
d4_2['PREDICTED_2_REDSHIFT'] = r4_2 
d4_15['PREDICTED_15_REDSHIFT'] = r4_15

d4_2.to_csv('/Users/amandascbeck/Desktop/RESEARCH/AGN_Project/AGN_BBH_overlap/data/r90_d4_2.csv')
d4_15.to_csv('/Users/amandascbeck/Desktop/RESEARCH/AGN_Project/AGN_BBH_overlap/data/r90_d4_15.csv')

# 2 band:

# # predict redshift for 2 bands using max range:
# r2_max = predictors.predict_redshift(d2.iloc[:, 2:6:2],
#                                      redshift_range = redshift_max_range,
#                                      n_inputs = 2)
# # putting it all together:
# d2['PREDICTED_MAX_REDSHIFT'] = r2_max
# # excluding values above z=2:
# d2_2 = d2[d2['PREDICTED_MAX_REDSHIFT']<=2.0]
# # excluding values above z=1.5:
# d2_15 = d2[d2['PREDICTED_MAX_REDSHIFT']<=1.5]
# training 2 predictor:
predict_redshift = training.train_standard_architecture(d2_2, redshift_2_range, n_inputs=2)
# training 1.5 predictor:
predict_redshift = training.train_standard_architecture(d2_15, redshift_15_range, n_inputs=2)
# final predictions and plots:
r2_2 = predictors.predict_redshift(d2_2.iloc[:, 2:6:2],
                                     redshift_range = redshift_2_range,
                                     n_inputs = 2)
r2_15 = predictors.predict_redshift(d2_15.iloc[:, 2:6:2],
                                     redshift_range = redshift_15_range,
                                     n_inputs = 2)
d2_2['PREDICTED_2_REDSHIFT'] = r2_2
d2_15['PREDICTED_15_REDSHIFT'] = r2_15

d2_2.to_csv('/Users/amandascbeck/Desktop/RESEARCH/AGN_Project/AGN_BBH_overlap/data/r90_d2_2.csv')
d2_15.to_csv('/Users/amandascbeck/Desktop/RESEARCH/AGN_Project/AGN_BBH_overlap/data/r90_d2_15.csv')