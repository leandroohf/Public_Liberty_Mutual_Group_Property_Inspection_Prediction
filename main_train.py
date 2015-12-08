# * ****************************************************************
#   Programmer[s]: Leandro Fernandes
#   Company/Institution:
#   email: leandroohf@gmail.com
#   Program: Property_Inspection_Prediction
#   Commentary: 1st Kaggle Competition
#   Date: September 30, 2015
#
#   The author believes that share code and knowledge is awesome.
#   Feel free to share and modify this piece of code. But don't be
#   impolite and remember to cite the author and give him his credits.
# * ****************************************************************

import libs.utils as utl
import pandas as pd

if __name__ == '__main__':
    train_pre = pd.read_pickle("data/pre/train_pre.pkl")
    test_pre = pd.read_pickle("data/pre/test_pre.pkl")

    print("Model training")
    parameters_dict = \
        utl.get_model_parameter_from_json_file('NOT implemented yet')

    response_str = "Hazard"
    xgb_model = utl.build_xgb_model(train_pre, test_pre, parameters_dict, response_str)

    print ('model was trained')
    print("Printing Model report")
    utl.build_xgb_model_report(xgb_model, train_pre, test_pre, response_str)