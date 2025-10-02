import matplotlib.pyplot as plt
import numpy as np

# Ensemble Models
models = ["Random Forest", 
          "PCA Random Forest", 
          "AdaBoost", 
          "PCA AdaBoost",
          "Bagging", 
          "PCA Bagging", 
          "Extra Trees", 
          "PCA Extra Trees",
          "Gradient Boosting", 
          "PCA Gradient Boosting"]

# Hard coded results for speed
accuracy = [0.7963, 0.6466, 0.1019, 0.0545, 0.6944, 0.7293, 0.8241, 0.6410, 0.6296, 0.2500]
precision = [0.8878, 0.7598, 0.8597, 0.9346, 0.7871, 0.7722, 0.8918, 0.7230, 0.7028, 0.3320]
f1_score = [0.7977, 0.6567, 0.0365, 0.0185, 0.6943, 0.7339, 0.8232, 0.6443, 0.6251, 0.2616]

barWidth = 0.25
r1 = np.arange(len(accuracy))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(figsize=(16, 8))

# Create bars
plt.bar(r1, accuracy, width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, precision, width=barWidth, edgecolor='grey', label='Precision')
plt.bar(r3, f1_score, width=barWidth, edgecolor='grey', label='F1-Score')
plt.title('PCA on Ensemble Methods')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(accuracy))], models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
