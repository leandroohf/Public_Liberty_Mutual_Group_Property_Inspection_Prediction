import numpy as np
import pandas as pd
import xgboost as xgb
import json as jsn
import datetime as dt
import os
import zipfile
import copy
import hashlib as hsh


def load_zipped_data(data_file_name):

    zipped_basename = os.path.basename(data_file_name)
    assert(os.path.splitext(zipped_basename)[1] == '.zip')

    csv_basename = os.path.splitext(zipped_basename)[0]
    assert(os.path.splitext(csv_basename)[1] == '.csv')

    zf = zipfile.ZipFile(data_file_name)
    data = pd.read_csv(zf.open(csv_basename), index_col=0)

    return data

def check_data(train, test):

    # print hsh.md5(str(train)).hexdigest()
    assert(hsh.md5(str(train)).hexdigest() == '89d680c9647fa4c25032924abd8ff5ce')
    # print hsh.md5(str(test)).hexdigest()
    assert(hsh.md5(str(test)).hexdigest() == '0ee1c41d578093ef477f44a06f5a7535')
    assert (train.shape[0] == 50999)
    assert (train.shape[1] == 33)
    assert ("Hazard" in train.columns.values)
    assert (test.shape[1] == 32)
    for cols in test.columns.values:
        assert (cols in train.columns.values)


def preprocess_data(raw_data, submission_data, train_size):

    columns_to_factorize = [
                            'T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8',
                            'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16',
                            'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12',
                            'T2_V13'
                            ]

    data_pre = factorize_data(raw_data, columns_to_factorize)
    submission_pre = factorize_data(submission_data, columns_to_factorize)
    train_data, test_data = split_data_randomly(data_pre, train_size)

    print('Saving a copy to data exploratory')
    print('NOT Implemented yet')

    print("Dropping unimportant columns")
    columns_to_drop = ['T1_V13', 'T2_V7', 'T2_V10',
                       'T1_V10']

    train_pre = drop_columns(train_data, columns_to_drop)
    test_pre = drop_columns(test_data, columns_to_drop)
    submission_pre = drop_columns(submission_pre, columns_to_drop)

    return train_pre, test_pre, submission_pre


def factorize_data(data_raw, columns_factor):

    data_factorized = data_raw.copy()

    for c in columns_factor:
        # print('Factorizing ' + c)
        data_factorized[c] = pd.factorize(data_factorized[c])[0]

    return data_factorized


def split_data_randomly(raw_data, train_size, sampling_seed=3):

    raw_data_size = raw_data.shape[0]
    test_size = raw_data_size - train_size
    assert(train_size < raw_data_size)

    print('Sampling test set')
    np.random.seed(sampling_seed)
    test_rows_samples = np.random.choice(raw_data.index, test_size, replace=False)
    test_data = raw_data.ix[test_rows_samples].copy()

    print('Sampling train set')
    train_data = raw_data.drop(test_rows_samples).copy()

    return train_data, test_data


def drop_columns(data, columns_to_drop):
    for c in columns_to_drop:
        # print 'dropping ' + c
        data.drop(c, axis=1, inplace=True)

    return data


def get_model_parameter_from_json_file(json_file_name):

    params = {}
    params["objective"] = "count:poisson" # "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 5
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.80
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["booster"] = "gbtree"
    params["seed"] = 0
    params["max_depth"] = 7
    params["num_rounds"] = 2500
    params["early_stopping"] = None  # 51
    params['nthread'] = 4
    params["eval_metric"] = "rmse"
    params["print.every.n"] = 100
    print(json_file_name)

    return params


def build_xgb_model(train_pre, test_pre, parameters_dict, response_str):

    xgb_train = get_xgb_dmatrix(train_pre, response_str)
    xgb_test = get_xgb_dmatrix(test_pre, response_str)

    # setting parameters to xgboost
    num_rounds = parameters_dict["num_rounds"]
    early_stopping = parameters_dict["early_stopping"]

    parameters_dict_copy = copy.deepcopy(parameters_dict)
    del parameters_dict_copy["early_stopping"]
    del parameters_dict_copy["num_rounds"]

    plst = list(parameters_dict_copy.items())
    print("Prameters list")
    print(plst)

    print('num_rounds: ' + str(num_rounds))
    print('early_stopping_rounds: ' + str(early_stopping))

    res = {'test': [], 'train': []}
    watch_list = [(xgb_test, 'test'), (xgb_train, 'train')]

    # train model
    xgb_model = xgb.train(parameters_dict, xgb_train, num_rounds, evals=watch_list,
                          early_stopping_rounds=early_stopping,
                          evals_result=res, verbose_eval=False)

    return xgb_model


def get_xgb_dmatrix(df, response_str):

    # convert to numpy
    xgb_response = np.array(df[response_str].copy())
    np_arr = df.drop(response_str, axis=1)
    train_arr = np.array(np_arr)

    # convert to xgb.DMatrix
    xgb_dmatrix = xgb.DMatrix(train_arr, label=xgb_response)

    return xgb_dmatrix


def write_submission_file(submission_pre, xgb_model, xgb_model_name,
                          xgb_param_dict, description_str="None"):

    # setting files
    submission_dir = './submissions'
    today_str = dt.datetime.today().strftime("%Y%m%d")
    date_dir = os.path.join(submission_dir, today_str)

    # create date_dir if It does not exist
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)

    submission_basename = xgb_model_name + '.csv'
    submission_filename = os.path.join(date_dir, submission_basename)

    xgb_dict_basename = xgb_model_name + '_param.json'
    xgb_dict_filename = os.path.join(date_dir, xgb_dict_basename)

    xgb_bin_basename = xgb_model_name + '.bin'
    xgb_bin_filename = os.path.join(date_dir, xgb_bin_basename)

    # Get predictions
    submission_ind = submission_pre.index

    submit_arr = np.array(submission_pre)
    xgb_submit = xgb.DMatrix(submit_arr)
    xgb_pred = xgb_model.predict(xgb_submit)

    # generate submission file
    preds = pd.DataFrame({"Id": submission_ind, "Hazard": xgb_pred})
    preds = preds.set_index('Id')
    print(preds.head(7))
    preds.to_csv(submission_filename)

    # generate param_dict file
    xgb_param_dict_copy = copy.deepcopy(xgb_param_dict)
    xgb_param_dict_copy["date"] = dt.datetime.today().strftime("%Y-%m-%d %X")
    xgb_param_dict_copy["description"] = description_str
    jsn.dump(xgb_param_dict_copy, open(xgb_dict_filename, 'wb'))

    # Save model
    xgb_model.save_model(xgb_bin_filename)

    # xgb_model.save_model(os.path.join(date_dir, '0001.model'))
    # dump model
    # xgb_model.dump_model(os.path.join(date_dir, 'dump.raw.txt'))
    # dump model with feature map
    # xgb_model.dump_model(os.path.join(date_dir,'dump.nice.txt'), os.path.join('./data/raw/','train.csv'),
    #                     with_stats=True)


def kaggle_normalized_gini(y_true, y_pred):
    # Gini: Shameless stolen from jpopham91's script
    # https://www.kaggle.com/jpopham91/liberty-mutual-group-property-inspection-prediction/gini-scoring-simple-and-efficient
    # https://www.kaggle.com/wiki/Gini
    assert (len(y_true) == len(y_pred))
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()

    # perfect model
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    # model prediction
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    lorenz_true = np.cumsum(true_order) / float(np.sum(true_order))
    lorenz_pred = np.cumsum(pred_order) / float(np.sum(pred_order))
    line_45 = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    gini_true = np.sum(line_45 - lorenz_true)
    gini_pred = np.sum(line_45 - lorenz_pred)

    # normalize to true Gini coefficient
    return gini_pred / gini_true


def build_xgb_model_report(xgb_model, train_pre, test_pre, response_str):

    #print('best iterations n score')
    #print('iteration: ' + str(xgb_model.best_iteration) + "\t score: "\
    #      + str(xgb_model.best_score))

    xgb_train = get_xgb_dmatrix(train_pre, response_str)
    xgb_test = get_xgb_dmatrix(test_pre, response_str)

    pred_train = xgb_model.predict(xgb_train)
    rmse_train = np.sqrt(np.mean((train_pre[response_str] - pred_train)**2))
    ngini_train = kaggle_normalized_gini(train_pre[response_str], pred_train)

    pred_test = xgb_model.predict(xgb_test)
    rmse_test = np.sqrt(np.mean((test_pre[response_str] - pred_test)**2))
    ngini_test = kaggle_normalized_gini(test_pre[response_str], pred_test)

    print("train-rmse: " + str(rmse_train) + "\t test-rmse: " + str(rmse_test))

    print('PS: Gini (Doubtful)')
    print("train-kaggle-normalized-gini: " + str(ngini_train) + \
          "\t test-kaggle-normalized-gini: " + str(ngini_test))


def rebalance_train_data(train_pre, sample_size, y_var):
    h = 1
    print(y_var + ": " + str(h))
    train_aux = train_pre[train_pre[y_var] == h]
    rows_sampled = np.random.choice(train_aux.index, sample_size)
    train_balanced_pre = train_pre.ix[rows_sampled]

    for h in train_pre[y_var].unique():
        print(y_var + ": " + str(h))
        train_aux = train_pre[train_pre[y_var] == h]
        rows_sampled = np.random.choice(train_aux.index, sample_size)
        train_balanced_pre = train_balanced_pre.append(train_pre.ix[rows_sampled])

    return train_balanced_pre
