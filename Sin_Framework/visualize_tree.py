"""
REALIZADO POR CLAUDE
Visualizador de árboles de decisión.

No depende de tu implementación de DecisionTree/ID3 -- solo espera que
los nodos tengan esta forma (la misma que ya definiste):

    InternalNode:  .attribute  (str)
                   .branches   (dict: {valor_del_atributo: nodo_hijo})

    LeafNode:      .value      (la clase predicha)
    REALIZADO POR CLAUDE
"""

import matplotlib.pyplot as plt


def visualize_tree(root, title="Decision Tree", save_path=None, figsize=(12, 7)):
    positions = {}   # id(node) -> (x, y)
    labels = {}      # id(node) -> texto a mostrar
    is_leaf = {}      # id(node) -> bool
    edges = []       # (id(parent), id(child), texto_de_la_rama)
    leaf_counter = [0]

    def layout(node, depth):
        node_id = id(node)
        if hasattr(node, "branches"):
            # Nodo interno: se posiciona en el promedio de sus hijos
            child_xs = []
            for branch_value, child in node.branches.items():
                child_x = layout(child, depth + 1)
                edges.append((node_id, id(child), str(branch_value)))
                child_xs.append(child_x)
            x = sum(child_xs) / len(child_xs) if child_xs else leaf_counter[0]
            labels[node_id] = str(node.attribute)
            is_leaf[node_id] = False
        else:
            # Hoja: se coloca en la siguiente posición horizontal libre
            x = leaf_counter[0]
            leaf_counter[0] += 1
            labels[node_id] = str(node.value)
            is_leaf[node_id] = True

        positions[node_id] = (x, -depth)
        return x

    layout(root, 0)

    fig, ax = plt.subplots(figsize=figsize)

    # Líneas y etiquetas de las ramas
    for parent_id, child_id, edge_label in edges:
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        ax.plot([x1, x2], [y1, y2], color="#999999", linewidth=1.3, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, edge_label, fontsize=9, color="#444444",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=2)

    # Cajas de los nodos
    for node_id, (x, y) in positions.items():
        leaf = is_leaf[node_id]
        color = "#dff0d8" if leaf else "#d9edf7"
        border = "#3c763d" if leaf else "#31708f"
        ax.text(x, y, labels[node_id], fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec=border, linewidth=1.3),
                zorder=3)

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Árbol guardado en: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # Demo con clases mínimas locales, solo para probar que el visualizador
    # funciona. NO son tus clases reales -- usa las tuyas al importar arriba.
    class LeafNode:
        def __init__(self, value):
            self.value = value

    class InternalNode:
        def __init__(self, attribute, branches):
            self.attribute = attribute
            self.branches = branches

    # Árbol de ejemplo: el mismo del diagrama Outlook/Humidity/Wind
    demo_root = InternalNode("Outlook", {
        "Sunny": InternalNode("Humidity", {
            "High": LeafNode("No"),
            "Normal": LeafNode("Yes"),
        }),
        "Overcast": LeafNode("Yes"),
        "Rain": InternalNode("Wind", {
            "Strong": LeafNode("No"),
            "Weak": LeafNode("Yes"),
        }),
    })

    visualize_tree(demo_root, title="Demo - PlayTennis", save_path="/mnt/user-data/outputs/tree_demo.png")
