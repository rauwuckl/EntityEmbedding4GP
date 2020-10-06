# Entity Embedding for Gaussian Process Regression
This code provides a method to represent categorical data by learned embedding vectors in the Gaussian Process
regression of [scikit-learn](https://scikit-learn.org/stable/).
Each level of the categorical variable is seen as an entity that is associated with a vector. The embedding vectors are learned from data. 

## Installation 
Inside this folder run
`pip install .`

## Example and Usage
The file `examples/MinimalExample.ipynb` contains a minimal example.   
The notebook requires some more dependencies:   

In the folder `examples` run
`pip install -r requirements.txt`. 

Then run `jupyter lab`

