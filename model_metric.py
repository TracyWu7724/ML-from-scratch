# 1. Classification Metrics
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 0, 1]

TP = sum((1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1))
FP = sum((1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1))
FN = sum((1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0))
TN = sum((1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0))

Accuracy = (TP + TN) / len(y_true)
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

print(f"Accuracy: {Accuracy}")
print(f"Precision: {Precision}")
print(f"Recall: {Recall}")
print(f"F1 Score: {F1}")

# 2. Regression Metrics
