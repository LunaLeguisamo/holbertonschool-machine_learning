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
        left_child (Node or Leaf): Left child node.
        right_child (Node or Leaf): Right child node.
        is_leaf (bool): Indicates if the node is a leaf.
        is_root (bool): Indicates if the node is the root of the tree.
        sub_population (list): Subset of data at this node (optional).
        depth (int): Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialize a new decision node.

        Args:
            feature (int): Feature index to split on.
            threshold (float): Threshold value for the split.
            left_child (Node or Leaf): Left child node.
            right_child (Node or Leaf): Right child node.
            is_root (bool): Flag indicating if this is the root node.
            depth (int): Depth level of the node.
        """
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

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves in the subtree.
        """
        count = 0
        if not only_leaves:
            count += 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def right_child_add_prefix(self, text):
        """
        Adds a prefix for formatting the right child when printing the tree.

        Args:
            text (str): String representation of the right child.

        Returns:
            str: Formatted string with right child prefix.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        Adds a prefix for formatting the left child when printing the tree.

        Args:
            text (str): String representation of the left child.

        Returns:
            str: Formatted string with left child prefix.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        Returns a string representation of the node and its children
        in a tree-like format.

        Returns:
            str: Visual representation of the subtree rooted at this node.
        """
        result = (
            f"{'root' if self.is_root else '-> node'} "
            f"[feature={self.feature}, threshold={self.threshold}]\n"
        )
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__().strip()
            )
        if self.right_child:
            result += self.right_child_add_prefix(
                self.right_child.__str__().strip()
            )
        return result

    def get_leaves_below(self):
        """
        Recursively collects all leaf nodes under this node.

        Returns:
            list: List of Leaf objects under this node.
        """
        if self.is_leaf:
            return Leaf.get_leaves_below(self)
        right = self.right_child.get_leaves_below()
        left = self.left_child.get_leaves_below()
        return left + right

    def update_bounds_below(self):
        """
        Recursively computes and sets the upper/lower bounds of each node
        based on feature splits, for potential use in visualization or pruning.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        if self.left_child:
            self.left_child.upper = self.upper.copy()
            self.left_child.lower = self.lower.copy()
            self.left_child.upper[self.feature] = self.threshold
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.upper = self.upper.copy()
            self.right_child.lower = self.lower.copy()
            self.right_child.upper[self.feature] = self.threshold
            self.right_child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf(Node):
    """
    Represents a leaf node in the decision tree, holding the predicted value.

    Attributes:
        value (any): The prediction or value held by the leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf node.

        Args:
            value (any): The value or class prediction of the leaf.
            depth (int): The depth of the leaf node.
        """
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
        """
        Returns 1, as a leaf node counts as one node (or one leaf).

        Args:
            only_leaves (bool): Ignored for leaf nodes.

        Returns:
            int: 1
        """
        return 1

    def update_bounds_below(self):
        """
        Leaves do not have bounds to update, so this is a no-op.
        """
        pass

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
            str: Description of the leaf node.
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Returns itself as it is a leaf node.

        Returns:
            list: A list containing this Leaf object.
        """
        return [self]


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
        """
        Initialize a new Decision Tree.

        Args:
            max_depth (int): Maximum allowed depth of the tree.
            min_pop (int): Minimum population to allow a split.
            seed (int): Seed for reproducibility.
            split_criterion (str): Strategy for splitting ("random", etc.).
            root (Node): Optionally provide a custom root node.
        """
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
        """
        Counts the number of nodes in the tree.

        Args:
            only_leaves (bool): If True, only count leaf nodes.

        Returns:
            int: Total number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
            str: Formatted tree structure.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Retrieves all the leaf nodes of the tree.

        Returns:
            list: List of Leaf objects in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Recursively updates bounds for all nodes in the tree.
        """
        self.root.update_bounds_below()
