import unittest
import libs.utils as utl
import pandas as pd

class GiniUnitTest(unittest.TestCase):

    def test_normalized_gini_results(self):

        actual_values = pd.Series([1,2,1,3])
        pred_values = pd.Series([1,2,2,3])

        print actual_values
        print pred_values

        normalized_gini = utl.kaggle_normalized_gini(actual_values, actual_values)
        print '{:10.4f}'.format(normalized_gini)
        self.assertAlmostEqual(1.0000,normalized_gini,3)

        print 'Test prediction: [1,2,2,3]'
        normalized_gini = utl.kaggle_normalized_gini(actual_values, pred_values)
        print '{:10.4f}'.format(normalized_gini)
        self.assertAlmostEqual(0.8571,normalized_gini,3)

        print 'Test worst prediction: [2,3,2,1]'
        pred_values = pd.Series([2,3,2,1])
        print pred_values

        normalized_gini = utl.kaggle_normalized_gini(actual_values, pred_values)
        print '{:10.4f}'.format(normalized_gini)
        self.assertAlmostEqual(0.2857,normalized_gini,3)