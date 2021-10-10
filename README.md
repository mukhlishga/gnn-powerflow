# gnn-powerflow

Graph Neural Network application in power system analysis, specifically for predicting power flow calculation.

Recent advances in Neural Network (NN) offer an interesting opportunity to integrate graph topology in a Neural Network system. This framework is called Graph Neural Network (GNN). In power systems, an electrical power grid can be represented as a graph with high dimensional features and interdependency among buses. This perspective may offer a better state of the art machine learning for power systems analysis. This study seeks the opportunity to integrate power grid topology in the GNN framework for power flow application. A comparison between several GNN architectures with equivalent model complexities are discussed. The comparison is also done for various dataset sizes. The performance of GNN compared to fully connected Neural Network over different sizes of dataset is concluded.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig1.PNG?raw=true)

Graph is a structure consist of nodes and edges. Nodes are the content of the data, and edges are the connectivity between these data.
Graph is basically how the world represents itself, since many natural generated data are in the shape of graph: protein, social media, and electrical power grid.
In power grid, the buses can be seen as the nodes, and the lines can be seen as the edges. The node feature will be voltage, active power, and reactive power, while the line feature can be line current and line resistance.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig2.PNG?raw=true)

In traditional NN model like Multi Layer Perceptron (MLP), usually only process the element of the data What if we can also utilize the structure information of the data, can we improve the model accuracy? That is the main idea of GNN, where we also process the shape information of a graph data. From the protein graph data, GNN can make use of the protein element as well as protein structure.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig3.PNG?raw=true)

In a grid, the bus can be of type slack bus, pv bus, and pq bus. Every bus has known and unknown variables, e.g. PQ bus knows its P and Q, and does now its V and d.
Power flow tries to calculate the unknown variables using known variables as input using electrical formula. GNN for power flow tries to predict the unknown variables by utilizing the known variables using machine learning model.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig4.PNG?raw=true)

Fully Connected Neural Network (FCNN) is a supervised machine learning framework. It is built upon a set of known input and output, then make a predictive model based on their relation. Input, hidden layer, then output. The line that connect node are weight or trainable parameter, because this value that will be adjusted in the training
Every node value is multiplied with the parameter, until it become the predicted output. The difference of the predicted output and the ground truth is called the loss function, can be calculated using logistic regression for classification or mean squared error for regression. We tune the value of the parameter for a number of iteration until the prediction model is accurate enough, the process is called optimization. After the model is made, we can give it a new input and predict the output.
FCNN is for flat type data. FCNN does hidden features extraction using its hidden layer.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig5.PNG?raw=true)

Convolutional Neural Network (CNN) is for grid-shaped data, a data with fixed size and node ordering. For example digital image data that consists of pixels with fixed size of row and column, and fixed node ordering. CNN uses conv. filter that does locality features extraction, then flatten the output of the conv, filter, then pass it on as an input to FCNN. Flatten meaning that the data from any shape is made to be a 1 dimensional type data like this. For example for image recognition. Convolution filter leverages the locality feature of the grid-type data.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig6.PNG?raw=true)

The question is, can CNN be applied to graph data? The answer is no, because CNN is for data with fixed row and column and fixed node ordering like images, while graph data has no fixed node ordering, we don’t what is the first or last node, and the size is arbitrary.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig7.PNG?raw=true)

The main principle of GNN is its message passing. A mechanism where a target node receives information from its neighboring nodes. E.g. node 4 as target node will get message from node 1, 5, and 6, and node 4 itself from previous state. This message passing at node 4 happens at every node for 1 GNN layer. We do simple summation of the neighboring node’s message, then multiply it with trainable parameter W, then apply nonlinear activation function as the input for next layer. But how to apply this mathematically?

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig8.PNG?raw=true)

We can do that using the adjacency matrix A. Matrix Adjacency is a matrix that represent the mapping of neighboring nodes in a graph data. For example at red box at the left you can see the neighboring nodes of node 4, because it is located at 4th row. The neighboring node are node 1, 4, 5, 6, so those column will be assigned with value of 1, and the other column will be assigned with zero. If we multiply this row it with the column of matrix data X, it will become a simple summation message passing.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig9.PNG?raw=true)

A more advanced GNN type is Graph Convolution Network (GCN). The only different with previous GNN is GCN implements averaging message passing. We can do that by preprocessing the A matrix with the inverse of degree matrix. A degree matrix is a matrix that denote the number of neighbor for each node of a graph, e.g. because node 4 has 4 neighbors, then the 4th row will be divided by 4, and it become averaging message passing. We can set the row and column of W to determine how many features will be extracted in the next layer.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig10.PNG?raw=true)

This is hiddent features extraction in multi layer GNN. We applied 2 GNN layer to a graph data with 9 nodes. Initially, every node has 2 features. It is multiplied with parameter matrix W0 with 2 row and 8 column, and the output is now has 8 features after applying the first GNN layer. After that, it is multiplied with matrix parameter W1 with 8 rows and 2 columns, and now every node has 2 features at the last state, after applied by the second GNN layer. Take a look at node 4 at the last state. This node 4 previously receive information from node 6, and node 6 previously get information from node 9. This means, node 4 now has 2 hops away message passing, or 2 neighbors away information acquiring. So that means applying 2 layer of GNN is same with 2 hops away GNN.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig11.PNG?raw=true)

Just like CNN, GNN can also be seen as another layer of Deep Learning. We can apply GNN layer to a graph data, then flatten the output and make it as an input to FCNN layer.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig12.PNG?raw=true)

This is the first case of graph data I used, consists of 14 nodes. Each nodes is a PQ bus.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig13.PNG?raw=true)

And this is the second case of the graph data. Now I add to the graph various size of loops, small loop, medium loop, and big loop. I want to check whether there is any effect of adding loop to the performance of GNN.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig14.PNG?raw=true)

This is the models to be compared in the experiment. I made 3 kinds of model. The model 1 consists of 2 traditional fully connected layer. The model 2 consists of 1 GNN layer continued by 1 fully connected layer. The model 3 consists of 2 GNN layer or 2 hops away GNN ended with a fully connected layer. The three models are made to have similar number of parameter, around 26 hundred, because I want to compare the models in an equivalent level. The hyperparameters can be seen in the figure.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig15.PNG?raw=true)

This is how I generate the dataset. I use digsilent power factory program. I made 102 datasets, each dataset has 2000 data points. From these 102 datasets, I use 1 as train dataset, 1 as validation dataset, and 100 as test dataset.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig16.PNG?raw=true)

And this is the shape of 1 dataset. So it has 2000 datapoints. Each datapoint consists of 14 rows from the 14 nodes of the graph data. Each node has 2 input features and 2 output features. If we want to use this data structure on the fully connected networks, we have to flatten it first.

![alt text](https://github.com/mukhlishga/gnn-powerflow/blob/main/document/fig17.PNG?raw=true)

This is the experiment step that I conducted. The result of this experiment can be completely seen in the report document.
