#type: ignore

import numpy as np
import pandas as pd
from pyparsing import Optional

class InternalNode:
    def __init__(self, attribute, branches: dict):
        self.attribute = attribute
        #Se utiliza un diccionario para almacenar las ramas del nodo interno, donde cada clave es un valor del atributo y el valor correspondiente es el subárbol que se sigue para ese valor.
        self.branches = branches

class LeafNode:
    def __init__(self, value):
        self.value = value

class DecisionTree:
    #Inicializo la clase DecisionTree con un parámetro opcional max_depth que define la profundidad máxima del árbol. Si no se proporciona, el árbol puede crecer indefinidamente.
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root: Optional[InternalNode | LeafNode] = None

    #La función entropy calcula la entropía de un atributo objetivo dado utilizando la formula vista en clase
    def entropy(self, target_attribute):
        value_counts = target_attribute.value_counts()
        probabilities = value_counts / len(target_attribute)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def information_gain(self, S, A, target_attribute):
        #Obtiene la entropía del atributo objetivo antes de dividir el conjunto de datos S por el atributo A
        parent_entropy = self.entropy(target_attribute)
        weighted_entropy = 0

        #Por cada valor unico del atributo A
        for value in S[A].unique():
            #Obtiene el subconjunto de S donde el atributo A tiene el valor actual
            subset = S[S[A] == value]
            #Se realiza el cálculo de la entropía ponderada del subconjunto y se suma a la entropía ponderada total
            weight = len(subset) / len(S)
            weighted_entropy += weight * self.entropy(target_attribute[S[A] == value])

        #Finalmente, la ganancia de información se calcula restando la entropía ponderada del subconjunto de la entropía del atributo objetivo antes de la división
        gain = parent_entropy - weighted_entropy
        return gain

    #La función choose_best_attribute selecciona el atributo con la mayor ganancia de información.
    def choose_best_attribute(self, examples, target_attribute, attributes):
        best_gain = -1
        best_attribute = None
        #Por cada atributo en el conjunto de atributos, calcula la ganancia de información y actualiza el mejor atributo si la ganancia es mayor que la mejor ganancia encontrada hasta ahora.
        for i in attributes:
            gain = self.information_gain(examples, i, examples[target_attribute])
            if gain > best_gain:
                best_gain = gain
                best_attribute = i
        return best_attribute

    def ID3(self, examples, target_attribute, attributes, depth=0):
        #Si todos los ejemplos tienen la misma clase, se crea un nodo hoja con esa clase.
        if len(set(examples[target_attribute])) == 1:
            return LeafNode(examples[target_attribute].iloc[0])
        #Si no hay más atributos para dividir o se ha alcanzado la profundidad máxima, se crea un nodo hoja con la clase mayoritaria.
        if not attributes or (self.max_depth is not None and depth >= self.max_depth):
            majority_class = examples[target_attribute].mode()[0]
            return LeafNode(majority_class)
        else:
            #Si no se cumple ninguna de las condiciones anteriores, se elige el mejor atributo para dividir los ejemplos y se crea un nodo interno
            A = self.choose_best_attribute(examples, target_attribute, attributes)
            decision_node = InternalNode(A, {})
            #Por cada valor único del atributo elegido, se crea un subconjunto de ejemplos y se llama recursivamente a la función ID3 para construir el subárbol correspondiente. El nodo interno se devuelve al final.
            for value in examples[A].unique():
                examples_v = examples[examples[A] == value]
                decision_node.branches[value] = self.ID3(examples_v, target_attribute, attributes - {A}, depth + 1)
        return decision_node

    #La funcion de prediccion fue realizada por Claude
    def predict_one(self, node, row):
        if isinstance(node, LeafNode):
            return node.value
        
        value = row[node.attribute]
        if value not in node.branches:
            # Unseen attribute value at test time — fall back gracefully
            # Simplest fix: return the most common leaf value under this node
            return self._majority_leaf(node)
        
        return self.predict_one(node.branches[value], row)

    def _majority_leaf(self, node):
        # Walk all branches, collect leaf values, return the most common
        leaves = []
        def collect(n):
            if isinstance(n, LeafNode):
                leaves.append(n.value)
            else:
                for child in n.branches.values():
                    collect(child)
        collect(node)
        return pd.Series(leaves).mode()[0]

    def predict(self, X: pd.DataFrame):
        return X.apply(lambda row: self.predict_one(self.root, row), axis=1)