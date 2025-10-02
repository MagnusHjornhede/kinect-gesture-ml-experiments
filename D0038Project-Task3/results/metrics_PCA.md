## Classifier Evaluation Metrics with and without PCA

pca = PCA(n_components=0.95)

| Algorithm                     | Accuracy | Precision | Recall | F1-Score |
|-------------------------------|----------|-----------|--------|----------|
| Without PCA k-NN              | 0.4643   | 0.5459    | 0.4643 | 0.4664   |
| With PCA k-NN                 | 0.4474   | 0.5285    | 0.4474 | 0.4531   |
| Without PCA Decision Tree     | 0.6598   | 0.6996    | 0.6598 | 0.6679   |
| With PCA Decision Tree        | 0.2594   | 0.2897    | 0.2594 | 0.2665   |
| Without PCA MLP               | 0.7895   | 0.8095    | 0.7895 | 0.7895   |
| With PCA MLP                  | 0.7820   | 0.7995    | 0.7820 | 0.7818   |
| Without PCA SVM-linear        | 0.8365   | 0.8592    | 0.8365 | 0.8409   |
| With PCA SVM-linear           | 0.8252   | 0.8441    | 0.8252 | 0.8271   |
| Without PCA SVM-poly-3        | 0.4135   | 0.8914    | 0.4135 | 0.4439   |
| With PCA SVM-poly-3           | 0.4173   | 0.8722    | 0.4173 | 0.4442   |
| Without PCA SVM-poly-5        | 0.2218   | 0.9221    | 0.2218 | 0.2626   |
| With PCA SVM-poly-5           | 0.2350   | 0.8932    | 0.2350 | 0.2563   |
| Without PCA SVM-poly-7        | 0.1297   | 0.9347    | 0.1297 | 0.1645   |
| With PCA SVM-poly-7           | 0.1485   | 0.9238    | 0.1485 | 0.1863   |
| Without PCA Random Forest     | 0.9342   | 0.9434    | 0.9342 | 0.9343   |
| With PCA Random Forest        | 0.7519   | 0.7986    | 0.7519 | 0.7508   |
| Without PCA AdaBoost          | 0.0752   | 0.8061    | 0.0752 | 0.0410   |
| With PCA AdaBoost             | 0.0489   | 0.9466    | 0.0489 | 0.0264   |
| Without PCA Bagging           | 0.8158   | 0.8415    | 0.8158 | 0.8175   |
| With PCA Bagging              | 0.7857   | 0.8194    | 0.7857 | 0.7912   |
| Without PCA Extra Trees       | 0.9079   | 0.9173    | 0.9079 | 0.9066   |
| With PCA Extra Trees          | 0.7030   | 0.7447    | 0.7030 | 0.6998   |
| Without PCA Gradient Boosting | 0.6335   | 0.7127    | 0.6335 | 0.6489   |
| With PCA Gradient Boosting    | 0.3421   | 0.4149    | 0.3421 | 0.3454   |

Done 2023-10-11
{'Without PCA k-NN': {'Accuracy': 0.4642857142857143, 'Precision': 0.5458963557778354, 'Recall': 0.4642857142857143, '
F1-Score': 0.4663579896814242}, 'With PCA k-NN': {'Accuracy': 0.4473684210526316, 'Precision': 0.5284991397857618, '
Recall': 0.4473684210526316, 'F1-Score': 0.4531158503002192}, 'Without PCA Decision Tree': {'Accuracy':
0.6597744360902256, 'Precision': 0.6995994806366524, 'Recall': 0.6597744360902256, 'F1-Score': 0.6678697851052058}, '
With PCA Decision Tree': {'Accuracy': 0.2593984962406015, 'Precision': 0.28968415820677124, 'Recall':
0.2593984962406015, 'F1-Score': 0.26654164096046196}, 'Without PCA MLP': {'Accuracy': 0.7894736842105263, 'Precision':
0.8094910347769366, 'Recall': 0.7894736842105263, 'F1-Score': 0.7894625764890076}, 'With PCA MLP': {'Accuracy':
0.7819548872180451, 'Precision': 0.7995154024746485, 'Recall': 0.7819548872180451, 'F1-Score': 0.781818038534024}, '
Without PCA SVM-linear': {'Accuracy': 0.8364661654135338, 'Precision': 0.8591501464189193, 'Recall':
0.8364661654135338, 'F1-Score': 0.8408757490291476}, 'With PCA SVM-linear': {'Accuracy': 0.825187969924812, 'Precision':
0.8441384747355996, 'Recall': 0.825187969924812, 'F1-Score': 0.8271051064246956}, 'Without PCA SVM-poly-3': {'Accuracy':
0.41353383458646614, 'Precision': 0.8914143304137884, 'Recall': 0.41353383458646614, 'F1-Score': 0.4439356942458944}, '
With PCA SVM-poly-3': {'Accuracy': 0.41729323308270677, 'Precision': 0.8722419978991559, 'Recall':
0.41729323308270677, 'F1-Score': 0.44418625366803477}, 'Without PCA SVM-poly-5': {'Accuracy': 0.22180451127819548, '
Precision': 0.9220793964016156, 'Recall': 0.22180451127819548, 'F1-Score': 0.2626143271723238}, 'With PCA SVM-poly-5':
{'Accuracy': 0.2349624060150376, 'Precision': 0.8932298987562146, 'Recall': 0.2349624060150376, 'F1-Score':
0.2563356689868129}, 'Without PCA SVM-poly-7': {'Accuracy': 0.12969924812030076, 'Precision': 0.9346954807060295, '
Recall': 0.12969924812030076, 'F1-Score': 0.16454662926632974}, 'With PCA SVM-poly-7': {'Accuracy':
0.14849624060150377, 'Precision': 0.9238477301934632, 'Recall': 0.14849624060150377, 'F1-Score': 0.18632165505917223}, '
Without PCA Random Forest': {'Accuracy': 0.9342105263157895, 'Precision': 0.9434083652573959, 'Recall':
0.9342105263157895, 'F1-Score': 0.9342767685409316}, 'With PCA Random Forest': {'Accuracy': 0.7518796992481203, '
Precision': 0.7986022405527049, 'Recall': 0.7518796992481203, 'F1-Score': 0.750799794292077}, 'Without PCA AdaBoost':
{'Accuracy': 0.07518796992481203, 'Precision': 0.8061103749801194, 'Recall': 0.07518796992481203, 'F1-Score':
0.040995072091637025}, 'With PCA AdaBoost': {'Accuracy': 0.04887218045112782, 'Precision': 0.9465974174566852, 'Recall':
0.04887218045112782, 'F1-Score': 0.026446870412520838}, 'Without PCA Bagging': {'Accuracy': 0.8157894736842105, '
Precision': 0.8415206916899508, 'Recall': 0.8157894736842105, 'F1-Score': 0.8174932376198107}, 'With PCA Bagging':
{'Accuracy': 0.7857142857142857, 'Precision': 0.8194025847634046, 'Recall': 0.7857142857142857, 'F1-Score':
0.7911712503057539}, 'Without PCA Extra Trees': {'Accuracy': 0.9078947368421053, 'Precision': 0.9173338630363945, '
Recall': 0.9078947368421053, 'F1-Score': 0.9065897596405739}, 'With PCA Extra Trees': {'Accuracy': 0.7030075187969925, '
Precision': 0.744654119438371, 'Recall': 0.7030075187969925, 'F1-Score': 0.6997945579208893}, 'Without PCA Gradient
Boosting': {'Accuracy': 0.6334586466165414, 'Precision': 0.7127466117299448, 'Recall': 0.6334586466165414, 'F1-Score':
0.6488605971118879}, 'With PCA Gradient Boosting': {'Accuracy': 0.34210526315789475, 'Precision': 0.4149410203325166, '
Recall': 0.34210526315789475, 'F1-Score': 0.3454082535031377}}

Done 2023-10-12
{'Without PCA k-NN': {'Accuracy': 0.4642857142857143, 'Precision': 0.5458963557778354, 'Recall': 0.4642857142857143, '
F1-Score': 0.4663579896814242}, 'With PCA k-NN': {'Accuracy': 0.4473684210526316, 'Precision': 0.5284991397857618, '
Recall': 0.4473684210526316, 'F1-Score': 0.4531158503002192}, 'Without PCA Decision Tree': {'Accuracy':
0.6635338345864662, 'Precision': 0.7011797678712645, 'Recall': 0.6635338345864662, 'F1-Score': 0.6694531178891436}, '
With PCA Decision Tree': {'Accuracy': 0.2650375939849624, 'Precision': 0.3011142672808326, 'Recall':
0.2650375939849624, 'F1-Score': 0.2736528159607591}, 'Without PCA MLP': {'Accuracy': 0.7650375939849624, 'Precision':
0.7912706594836744, 'Recall': 0.7650375939849624, 'F1-Score': 0.7665228321231532}, 'With PCA MLP': {'Accuracy':
0.7593984962406015, 'Precision': 0.7960922109127375, 'Recall': 0.7593984962406015, 'F1-Score': 0.7620745378642406}, '
Without PCA SVM-linear': {'Accuracy': 0.8364661654135338, 'Precision': 0.8591501464189193, 'Recall':
0.8364661654135338, 'F1-Score': 0.8408757490291476}, 'With PCA SVM-linear': {'Accuracy': 0.825187969924812, 'Precision':
0.8441384747355996, 'Recall': 0.825187969924812, 'F1-Score': 0.8271051064246956}, 'Without PCA SVM-poly-3': {'Accuracy':
0.41353383458646614, 'Precision': 0.8914143304137884, 'Recall': 0.41353383458646614, 'F1-Score': 0.4439356942458944}, '
With PCA SVM-poly-3': {'Accuracy': 0.41729323308270677, 'Precision': 0.8722419978991559, 'Recall':
0.41729323308270677, 'F1-Score': 0.44418625366803477}, 'Without PCA SVM-poly-5': {'Accuracy': 0.22180451127819548, '
Precision': 0.9220793964016156, 'Recall': 0.22180451127819548, 'F1-Score': 0.2626143271723238}, 'With PCA SVM-poly-5':
{'Accuracy': 0.2349624060150376, 'Precision': 0.8932298987562146, 'Recall': 0.2349624060150376, 'F1-Score':
0.2563356689868129}, 'Without PCA SVM-poly-7': {'Accuracy': 0.12969924812030076, 'Precision': 0.9346954807060295, '
Recall': 0.12969924812030076, 'F1-Score': 0.16454662926632974}, 'With PCA SVM-poly-7': {'Accuracy':
0.14849624060150377, 'Precision': 0.9238477301934632, 'Recall': 0.14849624060150377, 'F1-Score': 0.18632165505917223}, '
Without PCA Random Forest': {'Accuracy': 0.9229323308270677, 'Precision': 0.9319907819167287, 'Recall':
0.9229323308270677, 'F1-Score': 0.922442188455149}, 'With PCA Random Forest': {'Accuracy': 0.7537593984962406, '
Precision': 0.7967797728762455, 'Recall': 0.7537593984962406, 'F1-Score': 0.7510920550859325}, 'Without PCA AdaBoost':
{'Accuracy': 0.07706766917293233, 'Precision': 0.8277497142318873, 'Recall': 0.07706766917293233, 'F1-Score':
0.04449044104213693}, 'With PCA AdaBoost': {'Accuracy': 0.04887218045112782, 'Precision': 0.9465974174566852, 'Recall':
0.04887218045112782, 'F1-Score': 0.026446870412520838}, 'Without PCA Bagging': {'Accuracy': 0.8082706766917294, '
Precision': 0.8360537440362004, 'Recall': 0.8082706766917294, 'F1-Score': 0.8125882906166388}, 'With PCA Bagging':
{'Accuracy': 0.7951127819548872, 'Precision': 0.8279362840814855, 'Recall': 0.7951127819548872, 'F1-Score':
0.799746318980975}, 'Without PCA Extra Trees': {'Accuracy': 0.8947368421052632, 'Precision': 0.9036708036703198, '
Recall': 0.8947368421052632, 'F1-Score': 0.892851389246834}, 'With PCA Extra Trees': {'Accuracy': 0.6729323308270677, '
Precision': 0.7360157924225332, 'Recall': 0.6729323308270677, 'F1-Score': 0.682138340215292}, 'Without PCA Gradient
Boosting': {'Accuracy': 0.6447368421052632, 'Precision': 0.7229542016477449, 'Recall': 0.6447368421052632, 'F1-Score':
0.6599676583999906}, 'With PCA Gradient Boosting': {'Accuracy': 0.35150375939849626, 'Precision': 0.4308835252787814, '
Recall': 0.35150375939849626, 'F1-Score': 0.3550120227687673}}

