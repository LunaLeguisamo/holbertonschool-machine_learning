#!/usr/bin/env python3
"""
Simple implementation of a Decision Tree classifier with
support for custom split criteria, tree depth control, and
minimum sample threshold at nodes.

Includes:
- `Node` and `Leaf` classes to structure the tree
- `Decision_Tree` class to manage the tree building and structure

Dependencies:
- numpy
"""

import numpy as np


class Node:
    """
    Represents an internal node in the decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Threshold value for splitting the feature.
        left_child (Node or Leaf): Left subtree.
        right_child (Node or Leaf): Right subtree.
        is_leaf (bool): True if this node is a leaf.
        is_root (bool): True if this is the root node.
        sub_population (list): Subset of data at this node (optional).
        depth (int): Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialize a new internal node.

        Args:
            feature (int): Index of the splitting feature.
            threshold (float): Split threshold.
            left_child (Node or Leaf): Left child node.
            right_child (Node or Leaf): Right child node.
            is_root (bool): Flag to indicate if this is the root.
            depth (int): Depth of the node in the tree.
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
        """Recursively compute maximum depth below this node."""
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes in the subtree.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: Number of nodes or leaves below this node.
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
        Format the right child branch for tree string representation.

        Args:
            text (str): Child node text.

        Returns:
            str: Formatted right child string.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        Format the left child branch for tree string representation.

        Args:
            text (str): Child node text.

        Returns:
            str: Formatted left child string.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        Generate tree-like string representation from this node down.

        Returns:
            str: Tree visualization string.
        """
        result = (
            f"{'root' if self.is_root else '-> node'} "
            f"[feature={self.feature}, threshold={self.threshold}]\n"
        )
        if self.left_child:
            result +=\
                self.left_child_add_prefix(str(self.left_child).strip())
        if self.right_child:
            result +=\
                self.right_child_add_prefix(str(self.right_child).strip())
        return result

    def get_leaves_below(self):
        """
        Collect all leaves in the subtree.

        Returns:
            list: Leaf nodes under this node.
        """
        if self.is_leaf:
            return Leaf.get_leaves_below(self)
        right = self.right_child.get_leaves_below()
        left = self.left_child.get_leaves_below()
        return left + right

    def update_bounds_below(self):
        """
        Recursively calculate and assign feature bounds for each subtree node.
        Useful for pruning or visualization.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        if self.left_child:
            self.left_child.upper = self.upper.copy()
            self.left_child.lower = self.lower.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.upper = self.upper.copy()
            self.right_child.lower = self.lower.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Computes the indicator function from the Node.lower and
        Node.upper dictionaries and stores it in an attribute
        Node.indicator
        """
        def is_large_enough(x):
            return np.all(np.array([
                x[:, key] >= self.lower[key]
                for key in self.lower
            ]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([
                x[:, key] <= self.upper[key]
                for key in self.upper
            ]), axis=0)

        self.indicator =\
            lambda x: np.logical_and(is_large_enough(x), is_small_enough(x))

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Represents a terminal leaf node containing a prediction.

    Attributes:
        value (any): Predicted class or value.
        depth (int): Tree depth of the leaf.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf node.

        Args:
            value (any): The output value or prediction.
            depth (int): Depth in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns leaf depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 since leaf is a single node."""
        return 1

    def update_bounds_below(self):
        """Leaf has no children, no bounds to update."""
        pass

    def __str__(self):
        """Returns string for leaf node."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns list containing this leaf."""
        return [self]

    def pred(self, x):
        return self.value


class Decision_Tree:
    """
    Basic decision tree classifier with optional splitting strategy.

    Attributes:
        max_depth (int): Tree depth limit.
        min_pop (int): Minimum samples to allow a split.
        seed (int): RNG seed.
        split_criterion (str): Splitting method ("random", etc.).
        root (Node): Tree's root node.
        explanatory (ndarray): Feature matrix.
        target (ndarray): Target labels.
        predict (callable): Method to make predictions.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """
        Initialize the decision tree.

        Args:
            max_depth (int): Maximum allowed depth.
            min_pop (int): Minimum population for splitting.
            seed (int): RNG seed.
            split_criterion (str): Criterion for splitting nodes.
            root (Node): Optionally use a pre-defined root.
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
        """Returns the max depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts all nodes in the tree.

        Args:
            only_leaves (bool): Count only leaves if True.

        Returns:
            int: Number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Visualize the full tree as a string."""
        return str(self.root)

    def get_leaves(self):
        """Return all leaf nodes in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds on all nodes."""
        self.root.update_bounds_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        return self.root.pred(x)
