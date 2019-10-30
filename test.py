from sklearn.metrics import f1_score, accuracy_score

y_true = [-1, -1, -1, 1, 1, 1]
y_pred = [0, -1, -1, 0, 1, -1]
y_old = [-1, -1, -1, -1, 1, 1]

print("f1:", f1_score(y_true, y_pred, average='weighted'))
print("ac:", accuracy_score(y_true, y_pred))
print("f1:", f1_score(y_true, y_old, average='weighted'))
print("ac:", accuracy_score(y_true, y_old))
