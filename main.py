#type: ignore

import numpy as np
import pandas as pd
from pyparsing import Optional

class InternalNode:
    def __init__(self, attribute, branches: dict):
        self.attribute = attribute
        self.branches = branches

class LeafNode:
    def __init__(self, value):
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root: Optional[InternalNode] = None

    def entropy(self, target_attribute):
        value_counts = target_attribute.value_counts()
        probabilities = value_counts / len(target_attribute)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def information_gain(self, S, A, target_attribute):
        return self.entropy(target_attribute) - sum((len(S[S[A] == v]) / len(S)) * self.entropy(target_attribute[S[A] == v]) for v in S[A].unique())

    def choose_best_attribute(self, examples, target_attribute, attributes):
        best_gain = -1
        best_attribute = None
        for i in attributes:
            gain = self.information_gain(examples, i, examples[target_attribute])
            if gain > best_gain:
                best_gain = gain
                best_attribute = i
        return best_attribute

    def ID3(self, examples, target_attribute, attributes, depth=0):
        if len(set(examples[target_attribute])) == 1:
            return LeafNode(examples[target_attribute].iloc[0])
        if not attributes or (self.max_depth is not None and depth >= self.max_depth):
            majority_class = examples[target_attribute].mode()[0]
            return LeafNode(majority_class)
        else:
            A = self.choose_best_attribute(examples, target_attribute, attributes)
            decision_node = InternalNode(A, {})
            for value in examples[A].unique():
                examples_v = examples[examples[A] == value]
                decision_node.branches[value] = self.ID3(examples_v, target_attribute, attributes - {A}, depth + 1)
        return decision_node

from visualize_tree import visualize_tree
import pandas as pd

df = pd.read_csv("tennis.csv")
target = "PlayTennis"
attributes = set(df.columns) - {target, "Day"}

tree = DecisionTree()
tree.root = tree.ID3(df, target, attributes)

visualize_tree(tree.root, title="PLACEHOLDER", save_path="mi_tree.png")
