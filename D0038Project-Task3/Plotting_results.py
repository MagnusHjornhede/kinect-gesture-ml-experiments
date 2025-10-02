import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["MLP", "PCA MLP", "SVM-linear", "PCA SVM-linear", "SVM-poly-3", "PCA SVM-poly-3", "SVM-poly-7", "PCA SVM-poly-7"]

# Metrics
accuracy = [0.7222, 0.7180, 0.7315, 0.7726, 0.3241, 0.3289, 0.1019, 0.1316]
precision = [0.80432, 0.7462, 0.8297, 0.8010, 0.8917, 0.8687, 0.9568, 0.9281]
f1_score = [0.73159, 0.7195, 0.7405, 0.7730, 0.3461, 0.3757, 0.0908, 0.1594]

barWidth = 0.25
r1 = np.arange(len(accuracy))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(figsize=(14, 7))

# Create bars, labels and plot
plt.bar(r1, accuracy, width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, precision, width=barWidth, edgecolor='grey', label='Precision')
plt.bar(r3, f1_score, width=barWidth, edgecolor='grey', label='F1-Score')
plt.title('PCA on non Ensemble Methods')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(accuracy))], models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()