import matplotlib.pyplot as plt


# Ensemble Models
models = ["Random Forest", "PCA Random Forest", "AdaBoost", "PCA AdaBoost",
          "Bagging", "PCA Bagging", "Extra Trees", "PCA Extra Trees",
          "Gradient Boosting", "PCA Gradient Boosting"]

# Hard coded values
accuracy = [0.7963, 0.6466, 0.1019, 0.0545, 0.6944, 0.7293, 0.8241, 0.6410, 0.6296, 0.2500]

plt.figure(figsize=(16, 8))
plt.bar(models, accuracy, edgecolor='grey', label='Accuracy')
plt.title('PCA on Ensemble Methods')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()