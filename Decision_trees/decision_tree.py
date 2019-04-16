"""
This module implements the decision tree from scratch using information gain as split criterion
"""

import argparse
import csv
from utils import find_key_to_maxval, find_entropy, accuracy

class Node:
    """
    Node class to store info about each split point
    """
    def __init__(self, name):
        # Store feature name
        self.name = name

        # Distribution of labels after split to make decision
        self.split_dist = None


class Dtree:
    """
    Class for decision tree storage structure to store head node and sequence of nodes
    """
    def __init__(self):
        # Store head node so that tree can be traversed
        self.head = None

        # Store tree as nested dictionary in which next nodes are stored as
        # values with parent node as key
        self.node_path = {}


class DT:
    """
    Class for instantiating decision tree object
    """
    def __init__(self, max_depth=None):
        # Hyperparameter for max depth of tree
        self.max_depth = max_depth

        # Store tree in this attribute
        self.dtreeobj = None

    def train(self, train_x, train_y, headers):
        """
        Function to train the decision tree model
        Input:
        train_x: Feature columns for training dataset
        train_y: Label column for training dataset
        header: Feature names
        """
        # Generate tree by BFS strategy
        Q = [(train_x, train_y, headers, None, None, 0)]

        # Instantiate decision tree storage structure
        self.dtreeobj = Dtree()

        while Q:
            features, labels, header, parent, parent_val, level = Q.pop(0)
            # Find best split for the features and labels available at this
            # point in tree
            if self.max_depth and level > self.max_depth:
                continue
            arg_val, indices_split, dist_at_split = self.find_best_split(
                features, labels)

            if arg_val == -1:
                continue

            # Find feature name corresponding to the best split
            name = header[arg_val]

            # Node corresponding to current split and add label distribution
            # for this split
            curr = Node(name)
            curr.split_dist = dist_at_split
            if not parent:
                # Save head node
                self.dtreeobj.head = curr
            else:
                # Link current node to parent in the node path used for tree
                # traversal
                self.dtreeobj.node_path[parent][parent_val] = curr

            # Add current node in the node path used for tree traversal
            self.dtreeobj.node_path[curr] = {}

            # Split features, labels and header into number of splits for the current split (2 in this case)

            for num_split in indices_split:
                # If just 1 entry in dictionary for this value of splitting feature, this is leaf node
                # and so dont propagate tree creation beyond
                if len(dist_at_split[num_split]) > 1:
                    # Create features, labels and headers for the splits
                    features_new = []
                    header_new = []
                    for i in range(len(features)):
                        if i == arg_val:
                            continue
                        features_new.append([features[i][j]
                                             for j in indices_split[num_split]])
                        header_new.append(header[i])

                    labels_new = [labels[i] for i in indices_split[num_split]]
                    # If no features are remaining, leaf node
                    if features_new:
                        Q.append((features_new, labels_new,
                                  header_new, curr, num_split, level + 1))
        self.print_tree(level_wise=True)

    def print_tree(self, level_wise=False):
        """
        Function to display the decision tree
        Input:
        level_wise: Print tree with level wise (BFS) or depth wise (DFS)
        """
        stack = [(self.dtreeobj.head, None, None, 0)]

        while stack:
            if level_wise:
                # For level wise plot switch to BFS
                node, val, parent, level = stack.pop(0)
            else:
                # For normal tree diagram, use DFS
                node, val, parent, level = stack.pop()

            if parent:
                print(("|") * level + node.name, node.split_dist,
                      " when ", parent.name, " is ", val)
            else:
                print(("|") * level + node.name,
                      node.split_dist, "  head_node", val)

            for parent_val, next_node in self.dtreeobj.node_path[node].items():
                stack.append((next_node, parent_val, node, level + 1))

    def predict(self, dataset, header):
        """
        Function to perform predictions on the trained model
        Input:
        dataset: dataset for which predictions are to be obtained
        header: feature names
        Output:
        predictions for the input features
        """
        predictions = []
        for i in range(len(dataset[0])):
            update = False
            node = self.dtreeobj.head
            while self.dtreeobj.node_path[node]:

                feat = header.index(node.name)
                val = dataset[feat][i]

                if val in self.dtreeobj.node_path[node]:
                    node = self.dtreeobj.node_path[node][val]
                    update = True
                else:
                    break
            if update:
                feat = header.index(node.name)
                val = dataset[feat][i]
                predictions.append(find_key_to_maxval(node.split_dist[val]))
            else:
                predictions.append(find_key_to_maxval(node.split_dist[val]))

        return predictions

    def find_best_split(self, features, labels):
        """
        Function to find best feature to be used to split by determining maximum information gain corresponding to split in that feature
        Input:
        features: Features column out of which split is to be determined
        labels: Labels column for the dataset
        header: Feature names
        Output:
        arg_val: index corresponding to best feature for the split
        index_val: indices for different splits as per the best feature
        dist_at_split: Distribution at the split point
        header[arg_val]: Name of feature which provides maximum information gain
        """
        max_val, arg_val = -1, -1
        for ind, feature in enumerate(features):
            val, children_indices, dist_at_node = self.find_IG(
                feature, labels)

            if val > max_val:
                max_val = val
                arg_val = ind
                index_val = children_indices
                dist_at_split = dist_at_node
        # If maximum information gain is 0, split should not be made
        if not max_val > 0:
            return -1, -1, -1
        return arg_val, index_val, dist_at_split

    @staticmethod
    def find_IG(X, y):
        """
        Function to find Information Gain for a feature column
        Input:
        X: feature column
        y: label column
        Output:
        Information gain for split at the input feature, indices corresponding to data points in each split and distribution of label values after split
        """

        # Generate y values array when this feature = Y and N (used to find entropy of children nodes)
        children_y_val = {}

        # Counter for No.of data points with this feature = Y and No.of data points with this feature = N (used to find weight for weighted avg children entropy)
        count_children = {}

        # Store indices of the data points with this feature = Y and indices of data points with this feature = N (used for further splits at next level of tree)
        children_index = {}

        for ind, val in enumerate(X):
            if val in children_y_val:
                children_y_val[val].append(y[ind])
                children_index[val].append(ind)
            else:
                children_y_val[val] = [y[ind]]
                children_index[val] = [ind]
            count_children[val] = count_children.get(val, 0) + 1

        parent_entropy, _ = find_entropy(y)
        avg_children_entropy = 0

        # Distribution of label values for children nodes (number of values for y=1 and y=0 for each child so that majority vote can be done)
        dist_after_split = {}

        for child in children_y_val:

            entropy, label_dist_for_child = find_entropy(children_y_val[child])
            avg_children_entropy += (count_children[child] / len(X)) * entropy

            dist_after_split[child] = label_dist_for_child

        return parent_entropy - avg_children_entropy, children_index, dist_after_split


def read_data(path, vocab=None, test_mode=False):
    """
    Function to read input csv file and encode the categorical variables present in the data
    Input:
    path: csv file location
    vocab: vocabulary corresponding to training dataset (required for test dataset)
    test_mode: Boolean variable to indicate that test dataset is being read and so generated vocab is used for encoding
    Output:
    feat: Encoded features
    lab: Encoded labels
    vocab: vocabulary of categorical training dataset
    """
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # First row is header
        header = csv_reader.__next__()

        # Store data points as list
        data = []

        # Generate vocabulary for categories for encoding or use generated for test dataset
        if test_mode:
            # Use existing vocab generated using training dataset
            if not vocab:
                raise AssertionError("Training dataset encoding not provided")
        else:
            # Generate vocab corresponding to training dataset
            vocab = {}
            for i in range(len(header)):
                vocab[i] = []
        for row in csv_reader:
            if test_mode:
                for i, v in enumerate(row):
                    row[i] = vocab[i].index(row[i])
            else:
                for i, v in enumerate(row):
                    if v not in vocab[i]:
                        vocab[i].append(v)
                    # Ordinal encocding for categorical data
                    row[i] = vocab[i].index(v)
            data.append(row)

    # Structure the dataset lists in required format
    feat = []
    for l in zip(*data):
        feat.append(list(l))
    lab = feat[-1]
    feat = feat[:-1]

    if test_mode:
        return feat, lab
    return feat, lab, vocab, header


def argument_parser():
    """
    Function to parse argparse arguments
    """
    parser = argparse.ArgumentParser(description='Read input parameters')
    parser.add_argument('--train', dest='train_dir', default=None,
                        help='Training dataset directory location')
    parser.add_argument('--test', dest='test_dir', default=None,
                        help='Test dataset directory location')
    parser.add_argument('--max_d', dest='max_d', default=None, type=int,
                        help='Maximum depth for the tree')
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    # Read and encode data from csv files
    train_x, train_y, vocab, header = read_data(args.train_dir)
    test_x, test_y = read_data(args.test_dir, vocab, test_mode=True)

    header = header[:-1]
    # Create decision tree instance
    DTObj = DT(args.max_d)
    DTObj.train(train_x, train_y, header)

    train_predictions = DTObj.predict(train_x, header)
    print(f"Training dataset accuracy is {accuracy(train_y,train_predictions)}")

    predictions = DTObj.predict(test_x, header)
    print(f"Test dataset accuracy is {accuracy(test_y,predictions)}")
