#type: ignore
from visualize_tree import visualize_tree
import pandas as pd
from decisiontree import DecisionTree
import matplotlib.pyplot as plt

df = pd.read_csv("car.csv", index_col=0) #Le digo que la primera columna es el índice, no es una característica.

target = "Class"
attributes = set(df.columns) - {target}

train = df.sample(frac=0.7, random_state=42)
test = df.drop(train.index)

tree = DecisionTree(max_depth=5)
tree.root = tree.ID3(train, target, attributes)

predictions = tree.predict(test)
accuracy = (predictions.values == test[target].values).mean()
print(f"Overall Accuracy: {accuracy:.2%}")

#Bloque de codigo hecho por copilot para calcular la precisión por clase y mostrar la matriz de confusión.
#Acurracy by class
class_accuracies = {}
for cls in test[target].unique():
    cls_mask = test[target] == cls
    class_accuracy = (predictions[cls_mask].values == test[target][cls_mask].values).mean()
    class_accuracies[cls] = class_accuracy

for cls, acc in class_accuracies.items():
    print(f"Accuracy for {cls}: {acc:.2%}")

visualize_tree(tree.root, title="PLACEHOLDER", save_path="mi_tree.png")

#Plot the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# predictions and test[target] from your existing code
labels = sorted(df[target].unique())  # consistent class order, e.g. ['acc', 'good', 'unacc', 'vgood']

cm = confusion_matrix(test[target].values, predictions.values, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Decision Tree Confusion Matrix — Car Evaluation")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()