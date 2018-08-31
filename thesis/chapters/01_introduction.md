# Introduction

## Motivation

In contrast to classical machine learning models, training deep neural networks
requires navigating a huge parameter space. While most non-neural regression or
classification algorithms only require specification of a parameter set up-front
and often no more than a few, some parameters can (and should) be varied over
training time for neural networks. Looking at the popular scikit-learn library,
it can be seen that traditional methods such as SVMs, Gaussian Processes,
Decision Trees or Gradient Boosting typically require less than 10 algorithmic
parameters.

In neural networks the parameter space can have arbitrarily
many dimensions when factoring in the fact that some parameters can change over
time. Some parameters that need to be set initially are

* Network architecture (how many layers, how many units per layer)
* nonlinearity function
* type of loss
* optimization algorithm
* initial learning rate
* momentum
* weight decay
* training batch size

Parameters that can vary during training are

* learning rate (can be annealed)
* batch size (can be increased in place of learning rate reductions)
* trainability of layers

This makes finding an optimal training regimen very hard, particularly since
training for realistic problems can take a long time, meaning cross-validating
different models can be prohibitively expensive.

This thesis work is motivated by the lack of useful tools to debug deep learning
training. Without years of training and a lot of mathematical intuition and
expertise, it is often very hard to figure out why a network isn't learning or
how to ensure timely convergence.

There exist a variety of monitoring tools (see [Existing
Applications](#existing-applications)), but they are mostly low-level tools
which simplify visualization of certain network metrics. Most of the time, the
same metrics need to be reimplemented for every network because they have no
baked-in concept of reusing visualizations.  In contrast, the library developed
in this work is geared towards modularizing introspection metrics in such a way
that they are usable for any kind of model. The secondary purpose of the library
is the enablement to quickly iterate on hypothesized metrics extracted from the
training in order to diagnose problems such as

* bad initializations
* inappropirate learning rate
* layer/model saturation
* inappropriate network architecture
* bad generalization/overfitting

Detecting these problems early would cut down on the time needed to train a
model to satisfaction.

## Existing Applications

### TensorBoard

TensorBoard is a visulization toolkit originally developed for the TensorFlow
deep learning framework. It is composed of a Python library for exporting data
from the training process and a web server which reads the serialized data and
displays it in the browser. The server can be used independently from
TensorFlow, provided the data is serialized in the appropriate format. This
enables, e.g., a PyTorch port, termed ``TensorBoardX``.

During training, the developer adds operations which write scalars,
histograms, audio, or other to disk which can then be displayed in real-time in
the web browser. Besides scalar-valued functions, which could be e.g. the loss
curve or accuracy measure, TensorBoard supports histograms, audio, and embedding
data natively. However, concrete instances of these classes of training
artifact must be defined by the user and can only be reused, if the developer
creates a separate library for the computations involved.

New kinds of visualizations can be added with plugins, which require not only
writing the Python code exporting the data, but also JavaScript for actually
displaying it (d3.js is used for plotting).

### Visdom

Visdom by Facebook research fulfills more or less the same purpose as
TensorBoard, just with built-in support for Numpy and Lua Torch. The same
caveats apply

### Others

There are other tools such as [DeepVis](http://yosinski.com/deepvis) for
offline introspection, which offer insights into the training after the fact,
but do not help guiding the training process while it's running.

## Ikkuna

Ikkuna is the Python library developed for this thesis. It was designed with the
following goals in mind:

* Ease of use. Minimal configuration, maximum rewards.
* Flexible and all-encompassing API enabling creating arbitrary metrics which
  act on training artifacts
* Metrics shall be agnostic of model code.
* Plugin architecture so metrics written once can be used for any kind of model
* Framework agnosticism. Ideally, the library would support every deep learning
  framework through an extensible abstraction layer.

What it provides over the aforementioned tools is that it enables working at a
higher level of abstraction, liberating the developer from having to repeat
herself, exchanging visualizations and metrics and reduce the friction between
development and debugging.
