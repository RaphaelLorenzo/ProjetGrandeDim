# ProjetGrandeDim
Projet Analyse de données en grande dimension

Exploration des méthodes à partir du jeu de données EEG

Résultats:

|    | Name                        | Accuracy           | Accuracy Class 0   | Accuracy Class 1 | Accuracy Gap 0-1     |
| -- | --------------------------- | ------------------ | ------------------ | ---------------- | -------------------- |
| 0  | Logistic_ReducedFull_NoP    | 0.8955223880597015 | 0.9428571428571428 | 0.84375          | 0.09910714285714284  |
| 1  | Logistic_NMF_NoP            | 0.8208955223880597 | 0.8285714285714286 | 0.8125           | 0.016071428571428625 |
| 2  | Logistic_ReducedACP_NoP     | 0.8507462686567164 | 0.8285714285714286 | 0.875            | 0.046428571428571375 |
| 3  | Logistic_ACP_L1P            | 0.8507462686567164 | 0.8571428571428571 | 0.84375          | 0.013392857142857095 |
| 4  | Logistic_NMF_L1P            | 0.8507462686567164 | 0.8571428571428571 | 0.84375          | 0.013392857142857095 |
| 5  | Logistic_Full_L1P           | 0.9104477611940298 | 0.9428571428571428 | 0.875            | 0.06785714285714284  |
| 6  | Logistic_ACP_L2P            | 0.8507462686567164 | 0.8571428571428571 | 0.84375          | 0.013392857142857095 |
| 7  | Logistic_NMF_L2P            | 0.8656716417910447 | 0.8857142857142857 | 0.84375          | 0.041964285714285676 |
| 8  | Logistic_Full_L2P           | 0.8507462686567164 | 0.8857142857142857 | 0.8125           | 0.07321428571428568  |
| 9  | Logistic_ACP_ElasticP       | 0.835820895522388  | 0.8285714285714286 | 0.84375          | 0.015178571428571375 |
| 10 | Logistic_NMF_ElasticP       | 0.8656716417910447 | 0.8857142857142857 | 0.84375          | 0.041964285714285676 |
| 11 | Logistic_Full_ElasticP      | 0.8805970149253731 | 0.9142857142857143 | 0.84375          | 0.07053571428571426  |
| 12 | Logistic_Full_SCGroupP      | 0.8507462686567164 | 0.8857142857142857 | 0.8125           | 0.07321428571428568  |
| 13 | Logistic_Full_TimeGroupP    | 0.835820895522388  | 0.8857142857142857 | 0.78125          | 0.10446428571428568  |
| 14 | Logistic_Full_ChannelGroupP | 0.835820895522388  | 0.8857142857142857 | 0.78125          | 0.10446428571428568  |
| 15 | SVC_ACP_rbfKernel           | 0.8656716417910447 | 0.9142857142857143 | 0.8125           | 0.10178571428571426  |
| 16 | SVC_ACP_linearKernel        | 0.8208955223880597 | 0.8571428571428571 | 0.78125          | 0.0758928571428571   |
| 17 | SVC_NMF_rbfKernel           | 0.835820895522388  | 0.8857142857142857 | 0.78125          | 0.10446428571428568  |
| 18 | SVC_NMF_linearKernel        | 0.8507462686567164 | 0.8857142857142857 | 0.8125           | 0.07321428571428568  |
| 19 | SVC_Full_rbfKernel          | 0.8507462686567164 | 0.9142857142857143 | 0.78125          | 0.13303571428571426  |
| 20 | SVC_Full_linearKernel       | 0.8805970149253731 | 0.9428571428571428 | 0.8125           | 0.13035714285714284  |
| 21 | SVC_ACP_Best                | 0.8059701492537313 | 0.8571428571428571 | 0.75             | 0.1071428571428571   |
| 22 | SVC_NMF_Best                | 0.8208955223880597 | 0.8571428571428571 | 0.78125          | 0.0758928571428571   |
| 23 | SVC_Full_Best               | 0.8507462686567164 | 0.9142857142857143 | 0.78125          | 0.13303571428571426  |
| 24 | RF_ACP_Best                 | 0.8208955223880597 | 0.8571428571428571 | 0.78125          | 0.0758928571428571   |
| 25 | RF_NMF_Best                 | 0.8208955223880597 | 0.8285714285714286 | 0.8125           | 0.016071428571428625 |
| 26 | RF_Full_Best                | 0.8059701492537313 | 0.8571428571428571 | 0.75             | 0.1071428571428571   |
| 27 | BaggingKN_ACP_Best          | 0.8656716417910447 | 0.9428571428571428 | 0.78125          | 0.16160714285714284  |
| 28 | BaggingKN_NMF_Best          | 0.8208955223880597 | 0.8285714285714286 | 0.8125           | 0.016071428571428625 |
| 29 | BaggingKN_Full_Best         | 0.8507462686567164 | 0.9142857142857143 | 0.78125          | 0.13303571428571426  |
| 30 | AdaBoostDT_ACP_Best         | 0.8208955223880597 | 0.8857142857142857 | 0.75             | 0.13571428571428568  |
| 31 | AdaBoostDT_NMF_Best         | 0.7761194029850746 | 0.8                | 0.75             | 0.050000000000000044 |
| 32 | AdaBoostDT_Full_Best        | 0.835820895522388  | 0.8571428571428571 | 0.8125           | 0.044642857142857095 |
|    |