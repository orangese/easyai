
# EasyAI

EasyAI (`easyai`) is a small wrapper API for Keras, written for use by CSII AI at Millburn High School. It simplifies Keras' syntax so that a person with little programming experience and even less knowledge of machine learning can use it. Users should use this API as a springboard from which they can start coding using advanced and capable tools. Despite what its name might imply, EasyAI is centered around neural network algorithms (mainly their applications in visual fields) due to their recent prevalence and popularity.

## Core principles

* **Ease of use.** EasyAI was created for a very specific target audience: those who are just beginning their 
machine learning (and coding) journeys. It is meant to emulate pseudocode as much as possible. Because of its overly simple design, many features are not customizable. However, since the purpose of this API is to provide a simple introduction to programming AI, a lack of functionality is acceptable.

* **Ease of transition.** Ultimately, EasyAI is just an introduction to AI programming. Hopefully, EasyAI users 
will move on to more advanced and capable machine learning libraries. Since EasyAI is built off of Keras, users will
probably find it easiest to transition to Keras from EasyAI.

## Using EasyAI

As previously stated, EasyAI focuses on usability and should be simple to use _if you have some understanding of machine learning concepts_. Since this library is meant to be a teaching library, using it does require a bit of background knowledge.

The most basic EasyAI model is the `NN` object:

```python

from easyai import NN
from easyai.layers import Input, Dense

neural_network = NN(Input(100), Dense(200), Dense(5))

```

You can add and remove layers using the `add_layer` and `rm_layer` functions:

```python

neural_network.add_layer(Dense(200), position = 1)

neural_network.rm_layer(position = 1)

```

Training is as easy as `neural_network.train()`:

```python

x_train = "your training examples here"
y_train = "your training labels here"

neural_network.train(x_train, y_train, epochs = 10)

```

Need help getting started? Run `easyai.support.examples`:

```python

from easyai.support.examples import MNIST

MNIST.mlp() # creates and runs a standard neural network on the digit classifying dataset, MNIST

```

## Downloading EasyAI

EasyAI is a Python package and can therefore be installed with Python's installation tool, `pip`. In order to install EasyAI, you must have Python >= 3.6.0.

Mac users: you may need to run the Python certificate script before using EasyAI. In order to do so, locate the `Python 3.x` folder in your `Applications` folder in Finder and run `Install Certificate.command` and `Update Shell Profile.command`.

### Installation

If you have a CUDA-capable GPU and CUDA software (if you don't know what that means, you don't have it), install EasyAI using this command: `pip3 install git+https://github.com/orangese/easyai.git#egg=easyai[gpu]`.

Otherwise, use: `pip3 install git+https://github.com/orangese/easyai.git#egg=easyai[cpu]`.

### Upgrade

If you have a CUDA-capable GPU and CUDA software, upgrade EasyAI by using the below command:  
`pip3 install --upgrade git+https://github.com/orangese/easyai.git#egg=easyai[gpu]`.

Otherwise, use: `pip3 install --upgrade git+https://github.com/orangese/easyai.git#egg=easyai[cpu]`.
