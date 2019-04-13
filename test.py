from decision_tree import DT
import unittest
import math
from sklearn.feature_selection import mutual_info_classif
import numpy as np

class TestStringMethods(unittest.TestCase):

    def test_check_IG_with_theo(self):
        obj = DT()
        X=[1,1,1,1,1,1,0,0,0,0,0,0,0,0]
        y=[1,1,1,0,0,0,1,1,1,1,1,1,0,0]
        
        f = lambda x: math.log(x)
        H_parent = -((9/14)*f(9/14))-((5/14)*f(5/14))
        H_left = -((3/6)*f(3/6))-((3/6)*f(3/6))
        H_right = -((6/8)*f(6/8))-((2/8)*f(2/8))
        wt_left = 6/14
        wt_right = 8/14
        theo = H_parent - (wt_left*H_left) - (wt_right*H_right)

        self.assertEqual(obj.find_IG(X,y), theo)


    def test_check_IG_with_sklearn(self):
        obj = DT()
        X=[1,1,1,1,1,1,0,0,0,0,0,0,0,0]
        y=[1,1,1,0,0,0,1,1,1,1,1,1,0,0]
        
        sklearn_res = mutual_info_classif(np.array(X).reshape(-1,1),np.array(y),discrete_features=True)[0]
        self.assertEqual(round(obj.find_IG(X,y),6), round(sklearn_res, 6))

if __name__ == '__main__':
    unittest.main()