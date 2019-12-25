## Interaction Network for Image Classification 
In this repository, I will try my best to adapt the method "Interaction Networksfor Learning about Objects, Relations and Physics" for the task of image classification.
Here is the adapted configuration for the number of graph nodes, number of relations (edges) in the graph, number of features for each graph node and number of features for graph edges. 

The whole configuration is happening at this line of the code:

```
    model_config = {
        'D_S': Number of states each node can be in.
        'D_R': Number of feature dimensions for each edge in the graph.
        'D_E': Number of feature dimensions to represent each effect in the graph.
        'D_X': Number of feature dimensions for each node in the graph.
        'D_P': Number of feature dimensions for the final output of the graph, representing all nodes, edges and their features in the graph.
        'NUM_CLASSES': Number of classes for classification.
    }
```

## Experiments
To study the performance of an interaction network, two scenerio's are taken into account:

* Images with Plain Concepts (i.e. a simple number.): `MNIST`
* Image Consisting of Scenes with Objects: `CIFAR-10`

#### MNIST Results
##### EXP 1: CNN + GNN

```
model_config = {
        'D_S': 20, 
        'D_R': 1,
        'D_E': 20,
        'D_X': 128,
        'D_P': 1024,
        'NUM_CLASSES': 10 } -> Loss: 0.058 | Accuracy: 9809/10000 (98%)
```

##### EXP 2: CNN + GNN
```
model_config = {
        'D_S': 1,
        'D_R': 1,
        'D_E': 20,
        'D_X': 128,
        'D_P': 2048,
        'NUM_CLASSES': 10 } -> Loss: 0.0473 | Accuracy: 9862/10000 (99%)
```

##### EXP 3 CNN + GNN

```
model_config = {
        'D_S': 1,
        'D_R': 1,
        'D_E': 128,
        'D_X': 128,
        'D_P': 2048,
        'NUM_CLASSES': 10 } -> Loss: 0.0530 | Accuracy: 9842/10000 (98%)
```

##### EXP 4: CNN Only

```
Loss: 0.0357 | Accuracy: 9880/10000 (99%)
```

#### CIFAR-10 Results


##### EXP 1: CNN Only

```
Loss: 1.282 | Accuracy: 5436/10000 (54%)
```

##### EXP 2: CNN + GNN

```
model_config = {
        'D_S': 1,
        'D_R': 1,
        'D_E': 20,
        'D_X': 128,
        'D_P': 2048,
        'NUM_CLASSES': 10 } -> Loss: 1.581 | Accuracy: 4014/10000 (40%)
```