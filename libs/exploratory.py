import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import seaborn as sns
from pandas.tools.plotting import parallel_coordinates


def build_histogram_dashboard(train_pre):
    # Shameless stolen (inspired):
    # https://www.kaggle.com/rshah4/liberty-mutual-group-property-inspection-prediction/histogram-of-all-fields-with-labels/code

    plt.style.use('ggplot')
    plt.figure(figsize=(15, 12))

    # response histogram
    n_bins = np.size(train_pre.Hazard.unique())
    plt.hist(np.array(train_pre.Hazard), bins=n_bins, color='#3F5D7D')
    print 'Saving response histogram in figures/.'
    plt.savefig("figures/hazard_histogram.png")

    # predictors histograms
    for i in range(1, train_pre.shape[1]):
        # print 'i: ' + str(i) + ' col: ' + df.columns.values[i]
        plt.subplot(6, 6, i)
        f = plt.gca()
        f.axes.get_yaxis().set_visible(False)
        f.set_title(train_pre.columns.values[i])

        n_bins = np.size(train_pre.iloc[:, i].unique())
        # print 'n_bins: ' + str(n_bins)

        # print 'numpy conversion'
        plt.hist(np.array(train_pre.iloc[:, i]), bins=n_bins, color='#3F5D7D')
        # df.iloc[:, i].hist(bins=n_bins, color='#3F5D7D')

    print 'Saving predictors dashboard in figures/.'
    plt.tight_layout()
    plt.savefig("figures/predictors_dashboards.png")

def build_corrmatrix_dashboard(train_pre):


    plt = sns.corrplot(train_pre,annot=False)
    #sns.corrplot(train_pre)

    print 'Saving correlation matrix in figures/.'
    plt.savefig("figures/corr_matrix.png")


def build_xgb_features_importance_dashboard(xgb_model, train_pre):
    # Shameless stolen (inspired):
    # https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

    # Get features map to y legend
    feature_map_file = 'figures/xgb_features_map.fmap'
    features = train_pre.columns[1:]

    create_feature_map(features, feature_map_file)

    # Compute features importance
    importance = xgb_model.get_fscore(fmap=feature_map_file)
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    # Convert to data frame
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    # Plot dashboard
    plt.style.use('ggplot')
    df.plot(kind='barh', x='feature', y='fscore', legend=False,
            color='#3F5D7D', figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    print 'Saving features importance in figures/.'
    plt.gcf().savefig('figures/feature_importance_xgb.png')


def create_feature_map(features, feature_map_file):
    outfile = open(feature_map_file, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def build_parallel_coordinate_dashboard(train_pre):

    features = ['range', 'T2_V1', 'T1_V2', 'T2_V2', 'T1_V1']

    df = train_pre.copy()
    df['range'] = pd.cut(df['Hazard'], bins=[0,3,5,10,70], labels=["(1-3]","(3-6]","(6-10]","(10-70]"])
    df = df[df['Hazard'] < 11]

    # df = df[df['Hazard'].isin([1, 5, 10])]

    parallel_coordinates(df[features],'range')

def build_data_panel_dashboard(train_pre):

    features = ['range', 'T2_V1', 'T1_V2', 'T2_V2', 'T1_V1']

    df = train_pre.copy()
    df['range'] = pd.cut(df['Hazard'], bins=[0,3,5,10,70], labels=["(1-3]","(3-6]","(6-10]","(10-70]"])
    df = df[df['Hazard'] < 11]

    df = df[features]

    print 'df shape'
    print df.shape
    # predictors histograms
    k = 1
    for rng in df['range'].unique():
        print 'rng: ' + str(rng)
        df_cut = df[df['range'] == rng]

        print '\tdf_cut shape'
        print df_cut.shape

        for i in range(1, df_cut.shape[1]):
            print '\t\tk: ' + str(k) + ' col: ' + df_cut.columns.values[i]
            plt.subplot(3, 4, k)
            f = plt.gca()
            f.axes.get_yaxis().set_visible(False)
            f.set_title(df_cut.columns.values[i])

            n_bins = np.size(df_cut.iloc[:, i].unique())
            print '\t\tn_bins: ' + str(n_bins)

            # print 'numpy conversion'
            plt.ylabel(rng)
            plt.hist(np.array(df_cut.iloc[:, i]), bins=n_bins, color='#3F5D7D')
            k = k + 1