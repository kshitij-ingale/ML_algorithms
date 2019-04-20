
# Decision Trees

Decision Tree implementation using argparse, csv and math libraries in python. The performance was validated with sklearn implementation and the sklearn implementation was explored further using various parameters available in the Jupyter notebook.

## Repository Structure:
- **data** : Directory for storing training and test datasets (example datasets included)
- **decision_tree.py** : Script for the decision tree implementation
 - **utils.py** : Script for the supplementary functions used in decision tree implementation
 - **test.py** : Script for the unit test functions used in decision tree implementation
 - **Decision_Tree_sklearn.ipynb** : Jupyter notebook demonstrating decision tree classifier using sklearn and validating the from-scratch implementation

Following command can be used to execute the decision tree script
```
python decision_tree.py --train *path to training dataset* --test *path to training dataset*
```

Dataset included: Politicians attributes with the goal to predict the party of the politician, education dataset with grades corresponding to different courses with aim to predict final grade
