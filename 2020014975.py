import sys
import numpy as np
from collections import Counter


def entropy(y):
    counts = np.bincount(y)
    prob = counts / len(y)
    return -np.sum([p * np.log2(p) for p in prob if p>0])


def split(x, y, feature_index, threshold):
    left_idx = [i for i in range(len(x)) if x[i][feature_index] == threshold]
    right_idx = [i for i in range(len(x)) if x[i][feature_index] != threshold]
    x_left = [x[i] for i in left_idx]
    y_left = [y[i] for i in left_idx]
    x_right = [x[i] for i in right_idx]
    y_right = [y[i] for i in right_idx]

    return x_left, y_left, x_right, y_right


def best_split(X, y):
    best_gain = -1
    best_feature, best_value = None, None
    base_entropy = entropy(y)
    n_features = len(X[0])

    for feature_index in range(n_features):
        values = set(row[feature_index] for row in X)
        for val in values:
            X_left, y_left, X_right, y_right = split(X, y, feature_index, val)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gain = base_entropy - (
                len(y_left)/len(y) * entropy(y_left) +
                len(y_right)/len(y) * entropy(y_right)
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_value = val
    return best_feature, best_value


class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, *, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        return self.label is not None
    

def build_tree(X, y):
    if len(set(y)) == 1:
        return Node(label=y[0])
    if len(X[0]) == 0:
        most_common = Counter(y).most_common(1)[0][0]
        return Node(label=most_common)

    feature, value = best_split(X, y)
    if feature is None:
        most_common = Counter(y).most_common(1)[0][0]
        return Node(label=most_common)

    X_left, y_left, X_right, y_right = split(X, y, feature, value)
    left = build_tree(X_left, y_left)
    right = build_tree(X_right, y_right)
    return Node(feature=feature, value=value, left=left, right=right)


def predict(node, sample):
    while not node.is_leaf():
        if sample[node.feature] == node.value:
            node = node.left
        else:
            node = node.right
    return node.label


def read_train_file(path):
    with open(path, 'r') as f:
        lines = [line.strip().split('\t') for line in f if line.strip()]
    header = lines[0]
    data = lines[1:]
    X = [row[:-1] for row in data]
    label_names = list(sorted(set(row[-1] for row in data)))
    y = [label_names.index(row[-1]) for row in data]
    return header, X, y, label_names


def read_test_file(path):
    with open(path, 'r') as f:
        lines = [line.strip().split('\t') for line in f if line.strip()]
    header = lines[0]
    data = lines[1:]
    return header, data


def write_result_file(path, header, data, predictions, label_names):
    with open(path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for row, label in zip(data, predictions):
            f.write('\t'.join(row + [label_names[label]]) + '\n')


def main():
    train_path, test_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]

    train_header, X_train, y_train, label_names = read_train_file(train_path)

    test_header, X_test = read_test_file(test_path)

    tree = build_tree(X_train, y_train)

    predictions = [predict(tree, row) for row in X_test]

    write_result_file(result_path, test_header + [train_header[-1]], X_test, predictions, label_names)

if __name__ == "__main__":
    main()