#!/usr/bin/env python3
"""
Implementación simple de un clasificador de Árbol de Decisión con
soporte para criterios personalizados de división, control de la profundidad del árbol,
y un umbral mínimo de muestras en los nodos.

Incluye:
- Clases `Node` y `Leaf` para estructurar el árbol.
- Clase `Decision_Tree` para gestionar la construcción y estructura del árbol.

Dependencias:
- numpy
"""

import numpy as np


class Node:
    """
    Representa un nodo interno en el árbol de decisión.

    Atributos:
        feature (int): Índice de la característica utilizada para la división.
        threshold (float): Valor umbral para la división de la característica.
        left_child (Node o Leaf): Subárbol izquierdo.
        right_child (Node o Leaf): Subárbol derecho.
        is_leaf (bool): Verdadero si este nodo es una hoja.
        is_root (bool): Verdadero si este nodo es la raíz.
        sub_population (list): Subconjunto de datos en este nodo (opcional).
        depth (int): Profundidad del nodo en el árbol.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Inicializa un nuevo nodo interno.

        Args:
            feature (int): Índice de la característica de división.
            threshold (float): Umbral de división.
            left_child (Node o Leaf): Nodo hijo izquierdo.
            right_child (Node o Leaf): Nodo hijo derecho.
            is_root (bool): Indica si este es el nodo raíz.
            depth (int): Profundidad del nodo en el árbol.
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
        """Calcula recursivamente la profundidad máxima debajo de este nodo."""
        left = self.left_child.max_depth_below() if self.left_child else 0
        right = self.right_child.max_depth_below() if self.right_child else 0
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        Cuenta los nodos en el subárbol.

        Args:
            only_leaves (bool): Si es verdadero, cuenta solo las hojas.

        Returns:
            int: Número de nodos o de hojas en el subárbol.
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
        Formatea la rama del hijo derecho para la representación en cadena del árbol.

        Args:
            text (str): Texto del nodo hijo.

        Returns:
            str: Cadena con formato de la rama del hijo derecho.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        Formatea la rama del hijo izquierdo para la representación en cadena del árbol.

        Args:
            text (str): Texto del nodo hijo.

        Returns:
            str: Cadena con formato de la rama del hijo izquierdo.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        Genera una representación en cadena del árbol desde este nodo hacia abajo.

        Returns:
            str: Cadena que representa el árbol.
        """
        result = (
            f"{'root' if self.is_root else '-> node'} "
            f"[feature={self.feature}, threshold={self.threshold}]\n"
        )
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child).strip())
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child).strip())
        return result

    def get_leaves_below(self):
        """
        Recoge todas las hojas en el subárbol.

        Returns:
            list: Nodos hoja bajo este nodo.
        """
        if self.is_leaf:
            return Leaf.get_leaves_below(self)
        right = self.right_child.get_leaves_below()
        left = self.left_child.get_leaves_below()
        return left + right

    def update_bounds_below(self):
        """
        Calcula recursivamente y asigna los límites de las características para cada nodo en el subárbol.
        Útil para la poda o visualización.
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
        Calcula la función indicadora a partir de los diccionarios Node.lower y
        Node.upper y la almacena en un atributo Node.indicator.
        """
        def is_large_enough(x):
            return np.all(np.array([x[:, key] >= self.lower[key] for key in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, key] <= self.upper[key] for key in self.upper]), axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x), is_small_enough(x))

    def pred(self, x):
        """Realiza una predicción a partir de la característica y el umbral."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Representa un nodo terminal de hoja que contiene una predicción.

    Atributos:
        value (any): Clase o valor predicho.
        depth (int): Profundidad en el árbol de la hoja.
    """

    def __init__(self, value, depth=None):
        """
        Inicializa un nodo hoja.

        Args:
            value (any): El valor de salida o predicción.
            depth (int): Profundidad en el árbol.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Devuelve la profundidad de la hoja."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Devuelve 1 ya que la hoja es un solo nodo."""
        return 1

    def update_bounds_below(self):
        """La hoja no tiene hijos, no hay límites que actualizar."""
        pass

    def __str__(self):
        """Devuelve la cadena de texto para un nodo hoja."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Devuelve una lista que contiene esta hoja."""
        return [self]

    def pred(self, x):
        return self.value


class Decision_Tree:
    """
    Árbol de decisión básico para clasificación con estrategia de división opcional.

    Atributos:
        max_depth (int): Profundidad máxima permitida.
        min_pop (int): Mínimo de muestras para permitir una división.
        seed (int): Semilla para el generador de números aleatorios.
        split_criterion (str): Método de división ("random", etc.).
        root (Node): Nodo raíz del árbol.
        explanatory (ndarray): Matriz de características.
        target (ndarray): Etiquetas objetivo.
        predict (callable): Método para realizar predicciones.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """
        Inicializa el árbol de decisión.

        Args:
            max_depth (int): Profundidad máxima permitida.
            min_pop (int): Mínimo de muestras para permitir una división.
            seed (int): Semilla para el generador de números aleatorios.
            split_criterion (str): Criterio para dividir los nodos.
            root (Node): Nodo raíz opcional.
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
        """Devuelve la profundidad máxima del árbol."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Cuenta todos los nodos en el árbol.

        Args:
            only_leaves (bool): Si es verdadero, cuenta solo las hojas.

        Returns:
            int: Número de nodos o de hojas.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Visualiza el árbol completo como una cadena."""
        return str(self.root)

    def get_leaves(self):
        """Devuelve todos los nodos hoja en el árbol."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Actualiza los límites en todos los nodos."""
        self.root.update_bounds_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        return self.root.pred(x)
