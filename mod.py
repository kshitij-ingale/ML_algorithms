import argparse
import csv
import math

class Node:
    def __init__(self, name):
        self.name = name
        self.next = None
        self.split_dist = None

class Dtree:
    def __init__(self):
        self.head = None
        self.node_path = {}


class DT:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        self.tree = None
        self.dtreeobj = None

    def train(self, train_x, train_y, header, vocab_map_header):
        """
        Function to train the decision tree model
        Input:
        train_x: Feature columns for training dataset
        train_y: Label column for training dataset
        header: Feature names
        """

        stack = [(train_x, train_y, header, None, None,0)]
        Tree = []

#####################################################################################################################################################################
        dtobj = Dtree()
#####################################################################################################################################################################
        
        while stack:
            features, labels, header, parent, parent_val,level = stack.pop()
            arg_val, indices_split, dist_at_split, name = self.find_best_split(features, labels, header)
            Tree.append((dist_at_split, name))
            
#####################################################################################################################################################################
            curr = Node(name)
            curr.split_dist = dist_at_split
            if not parent:
                dtobj.head = curr
            else:
                dtobj.node_path[parent][parent_val] = curr
            dtobj.node_path[curr.name] = {}
#####################################################################################################################################################################
            
            for num_split in indices_split.keys():
                features_new = []
                header_new = []
                for i in range(len(features)):
                    if i==arg_val:
                        continue
                    features_new.append([features[i][j] for j in indices_split[num_split]])
                    header_new.append(header[i])

                labels_new = [labels[i] for i in indices_split[num_split]]
                if features_new and level+1 < self.depth:
                    stack.append((features_new, labels_new, header_new, name, num_split,level+1))

#####################################################################################################################################################################
                    dtobj.node_path[curr.name][num_split] = {}
#####################################################################################################################################################################

        self.tree = Tree
        self.dtreeobj = dtobj
        self.print_tree(Tree, header, vocab_map_header)


    def print_tree(self, Tree, header, vocab_map_header):
        stack = [(self.dtreeobj.head, None, None, 0)]

        while stack:
            node, val, parent, level = stack.pop()
            print(node.name+"\t"+str(level))
            # print(("|")*level+node.name)
            if self.dtreeobj.node_path[node.name]:    
                for parent_val, next_node in self.dtreeobj.node_path[node.name].items():
                    stack.append((next_node, parent_val, node, level+1))


    def predict(self, test_x, header, vocab_map_header):
        
        predictions = []
        for i in range(len(test_x[0])):

            node = self.dtreeobj.head
            # while node:
            while self.dtreeobj.node_path[node.name]:
                feat = header.index(node.name)
                val = test_x[feat][i]
                node = self.dtreeobj.node_path[node.name][val]

            feat = header.index(node.name)
            val = test_x[feat][i]
            max_val = -1
            decision = 0
            for key,value in node.split_dist[val].items():
                if val > max_val:
                    decision = key
            predictions.append(decision)

        return predictions
        
    def find_best_split(self, features, labels, header):
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
        for i in range(len(features)):

            val, children_indices, dist_at_node = self.find_IG(features[i], labels)
            if val > max_val:
                max_val = val
                arg_val = i
                index_val = children_indices
                dist_at_split = dist_at_node
        return arg_val, index_val, dist_at_split, header[arg_val]
    
    def find_IG(self, X, y):
        """
        Function to find Information Gain for a feature column
        Input:
        X: feature column
        y: label column
        Output:
        Information gain for split at the input feature 
        """
        def find_entropy(arr):
            """
            Function to find entropy of input array
            Input:
            X: input column
            Output:
            Entropy corresponding to the input column
            """
            count_store = {}
            total = 0
            for i in arr:
                count_store[i] = count_store.get(i,0) + 1
                total += 1
            entropy = 0
            for value in count_store.values():
                # Entropy calculation assuming base of e, base of 2 can be specified in the math.log function 
                # Since, sklearn uses base of e, this function uses the default base to validate results against sklearn
                entropy -= (value/total)*math.log(value/total)
            return entropy, count_store
        
        children_y_val = {}
        count_children = {}
        children_index = {}
        for i in range(len(X)):
            if X[i] in children_y_val:
                children_y_val[X[i]].append(y[i])
                children_index[X[i]].append(i)
            else:
                children_y_val[X[i]] = [y[i]]
                children_index[X[i]] = [i]
            
            count_children[X[i]] = count_children.get(X[i],0) + 1

        parent_entropy, count_parent = find_entropy(y)
        avg_children_entropy = 0
        dist_after_split = {}

        for child in children_y_val.keys():
            
            entropy, label_dist_after_split = find_entropy(children_y_val[child])
            avg_children_entropy += (count_children[child]/len(X))*entropy

            dist_after_split[child] = label_dist_after_split

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
                raise ValueError("Training dataset encoding not provided")
        else:
            # Generate vocab corresponding to training dataset
            vocab = {}
            for i in range(len(header)):
                vocab[i] = []

        for row in csv_reader:
            if test_mode:
                for i,v in enumerate(row):
                    row[i] = vocab[i].index(row[i])
            else:
                for i,v in enumerate(row):
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
    else:
        return feat, lab, vocab, header

def accuracy(actual,predictions):
    assert len(actual)==len(predictions), "Mismatch in number of actual data points and predicted data points"
    correct = 0.0
    for i in range(len(actual)):
        if actual[i]== predictions[i]:
            correct += 1

    return correct/len(actual)



def argument_parser():
    parser = argparse.ArgumentParser(description='Read input parameters')
    parser.add_argument('--train', dest='train_dir', default = None,
                    help='Training dataset directory location')
    parser.add_argument('--test', dest='test_dir', default = None,
                    help='Test dataset directory location')
    parser.add_argument('--max_d', dest='max_d', default = None,
                    help='Maximum depth for the tree')
    return parser.parse_args()

if __name__ == '__main__':
    args = argument_parser()
    # Read and encode data from csv files
    train_x, train_y, vocab, header = read_data(args.train_dir)
    test_x, test_y = read_data(args.test_dir, vocab, test_mode = True)
    
    # Change vocab keys as per headers
    vocab_map_header = vocab.copy()
    for i in range(len(header)):
        vocab_map_header[header[i]] = vocab_map_header.pop(i)

    # Create decision tree instance
    DT_obj = DT(args.max_d)

    DT_obj.train(train_x, train_y, header, vocab_map_header)


    train_predictions = DT_obj.predict(train_x, header, vocab_map_header)

    print(f"Training dataset accuracy is {accuracy(train_y,train_predictions)}")


    predictions = DT_obj.predict(test_x, header, vocab_map_header)

    print(f"Test dataset accuracy is {accuracy(test_y,predictions)}")

    
