# Model Evaluation on whole_data_StandardScaler

The performance metrics of the models on the test set were as follows:

| Classifier        | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Gradient Boosting | 0.0282   | 0.8243    | 0.0282 | 0.0126   |
| k-NN  (6)         | 0.4117   | 0.5224    | 0.4117 | 0.4158   |
| SVM-linear        | 0.7876   | 0.8155    | 0.7876 | 0.7890   |
| SVM-poly-3        | 0.6523   | 0.7576    | 0.6523 | 0.6728   |
| SVM-poly-20       | 0.6523   | 0.7576    | 0.6523 | 0.6728   |
| SVM-sigmoid       | 0.7049   | 0.7679    | 0.7049 | 0.7164   |
| SVM-rbf           | 0.7519   | 0.7968    | 0.7519 | 0.7535   |
| AdaBoost          | 0.1692   | 0.7630    | 0.1692 | 0.1156   |
| Bagging           | 0.7462   | 0.7965    | 0.7462 | 0.7507   |
| MLP               | 0.7293   | 0.7613    | 0.7293 | 0.7281   |
| Decision Tree     | 0.5846   | 0.6335    | 0.5846 | 0.5865   |
| Random Forest     | 0.8553   | 0.8833    | 0.8553 | 0.8512   |
| Extra Trees       | 0.8553   | 0.8714    | 0.8553 | 0.8506   |

## Performance Overview

Random forest and Extra Trees outperforms the other models both in accuracy and F1-score. SVM with a linear kernel is
the best among SVM variants, securing a high accuracy at 79%. If we look at Neural networks represented here by the MLP
model
also provide decent results with an accuracy of 73.5%.On the contrary, SVM with a polynomial degree 20 shows a very poor
performance at 5.45%, indicating it's very likely overfitting or perhaps
inappropriate for this type of data.




