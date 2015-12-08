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

if __name__ == '__main__':

    print("Loading data ...")
    raw_data = utl.load_zipped_data('data/raw/train.csv.zip')
    submission_data = utl.load_zipped_data("data/raw/test.csv.zip")

    print ("Data firewall: checking raw data ...")
    utl.check_data(raw_data, submission_data)

    print("Data munging")
    train_size = 40000  # 70% for train
    train_pre, test_pre, submission_pre = utl.preprocess_data(raw_data,
                                                              submission_data,
                                                              train_size)

    print("Saving pre-processing data")
    train_pre.to_pickle("data/pre/train_pre.pkl")
    test_pre.to_pickle("data/pre/test_pre.pkl")

    print("Model training")
    parameters_dict = \
        utl.get_model_parameter_from_json_file('NOT implemented yet')

    response_str = "Hazard"
    xgb_model = utl.build_xgb_model(train_pre, test_pre, parameters_dict, response_str)
    print ('model was trained')
    print("Printing Model report")
    utl.build_xgb_model_report(xgb_model, train_pre, test_pre, response_str)

    print("Save Submission file")
    xgb_model_name = 'xgb_model'
    utl.write_submission_file(submission_pre, xgb_model, xgb_model_name, parameters_dict)
