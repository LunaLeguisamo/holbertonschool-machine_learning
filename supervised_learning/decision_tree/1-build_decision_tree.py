#!/usr/bin/env python3
"""
Simple implementation of a Decision Tree classifier with
support for custom split criteria, tree depth control, and
population threshold at nodes.

Includes `Node` and `Leaf` classes for tree structure, and
a `Decision_Tree` class for model initialization and structure management.

Dependencies:
- numpy
"""

import numpy as np


class Node:
    """
    Represents an internal decision node in a decision tree.

    Attributes:
        feature (int): Index of the feature to split on.
        threshold (float): Threshold value to split the data.
        left_child (Node): Left child node.
        right_child (Node): Right child node.
        is_leaf (bool): Indicates if the node is a leaf.
        is_root (bool): Indicates if the node is the root of the tree.
        sub_population (list): Subset of data at this node (optional).
        depth (int): Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively calculates the maximum depth from this node downward.

        Returns:
            int: Maximum depth of the subtree rooted at this node.
        """
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()

        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below this node, including itself.
        If `only_leaves=True`, it counts only leaf nodes.
        """
        count = 0
        if self.is_leaf:
            return 1 if only_leaves else 1

        if not only_leaves:
            count += 1

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count


class Leaf(Node):
    """
    Represents a leaf node in the decision tree, holding the predicted value.
    Attributes:
        value (any): The prediction or value held by the leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf node.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree:
    """
    A basic Decision Tree classifier.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum number of samples to allow a split.
        seed (int): Seed for random number generator.
        split_criterion (str): Criterion used to split nodes (e.g., "random").
        root (Node): Root node of the decision tree.
        explanatory (ndarray): Feature data (to be assigned later).
        target (ndarray): Target labels (to be assigned later).
        predict (callable): Prediction method (to be implemented).
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the depth of the tree.

        Returns:
            int: Maximum depth from root to deepest leaf.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)
