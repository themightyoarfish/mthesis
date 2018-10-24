# Ikkuna

Ikkuna is the Python library developed for this thesis. It targets Python 3.6
and was designed with the following goals in mind:

#. Ease of use. Minimal configuration, maximum rewards.
#. Flexible and all-encompassing API enabling creating arbitrary metrics which act on training artifacts
#. Metrics shall be agnostic of model code.
#. Plugin architecture so metrics written once can be used for any kind of model
#. Framework agnosticism. Ideally, the library would support every deep learning framework through an extensible abstraction layer.

What it provides over the aforementioned tools is that it enables working at a
higher level of abstraction, liberating the developer from having to repeat
herself, exchanging visualizations and metrics and reduce the friction between
development and debugging.

## Design Principles

Of the aforementioned goals, all except one have been accomplished. The
objective of making the library agnostic to the deep learning framework being
used (TensorFlow, PyTorch, PyCaffe, Chainer, etc.) has been neglected for practical reasons.
Enabling this kind of support is beyond the scope of this thesis and only
requires the implementation of a software layer which offers framework-agnostic
access to network modules, activations, gradients and all the other necessary
information. While this is certainly possible and useful, the PyTorch framework
has been chosen for this work to create a proof of concept. The choice is
motivated in [@sec:dl-frameworks].

The overarching architecture of this software must lend itself to this
agnosticity goal, however. As such, a very loose coupling between model code,
metric computation and visualizations is desired. Not only will this aid in
extending the framework to different deep learning libraries, but it is also a
prerequisite for allowing for modular, self-contained visualizations or metrics
which can be installed and used separately and independently of specific model
code.. The Publisher-Subscriber design pattern has been chosen for these reasons
([@sec:pubsub]).

## Deep Learning frameworks {#sec:dl-frameworks}

The currently available deep learning libraries can be located on a spectrum
between define-by-run and define-and-run. The first extreme would be a
framework such as PyTorch [@paszke2017automatic] or Chainer [@tokui2015chainer],
where there exist no two distinct execution phases -- just like in an ordinary
matrix library like NumPy, each statement immediately returns or operates on an
actual value. By contrast, graph-based frameworks like TensorFlow[^tf] require
specifying the model graph in a domain-specific language (TensorFlow has Python,
Java and C++ APIs, Caffe uses Prototxt files), compile it to a different
representation and the the model is run and trained in a second phase. While
this enables graph-based optimizations, the main downsides are that

* control flow cannot use the host language features, but must be done with
  the API used for defining models. Instead of
  ```{#lst:whilepy-pt .python}
  counter = torch.tensor(0)
  # repeated matrix multiplication
  while counter < tensor:
      counter += 1
      h = torch.matmul(W, h) + b
  ```
  one must use a construction like this
  ```{#lst:whilepy-tf .python}
  counter = tf.constant(0)
  while_condition = lambda counter: tf.less(counter, tensor)
  # loop body
  def body(counter):
      h = tf.add(tf.matmul(W, h), b)
      # increment counter
      return [tf.add(counter, 1)]

  # do the actual loop
  r = tf.while_loop(while_condition, body, [counter])
  ```

* halting execution at aribtrary points in the training is not possible,
  since the actual training is not happening in the host language, but is
  more often handed off to lower-level implementations in its entirety.

This makes conditional processing and debugging much less ergonomic.

[^tf]: Since version 1.4, TensorFlow gravitates toward define-by-run through the
  introduction of _eager execution_, which becomes the default mode in version
  2.0. Graph-based execution is still available, but not the default any longer.

All frameworks have in common that they build a graph representation of the
model, wether implicitly or explicitly. Nodes in the graph are operations while
edges are data flowing between operations. This allows naturally parallelizing
independent computations. To compute gradients, the graph can be traversed
backwards from the output node by applying the chain rule of differentiation.
Define-and-run frameworks like TensorFlow create the graph explicitly; the user
uses the API to do exactly this. The graph -- once compiled -- is fixed for the
entire training process.  PyTorch on the other hand implicitly records all
operations and also overloads operators for this purpose.  The graph is thus
recreated for each propagation through the network. This precludes some
optimizations, but makes dynamically changing networks easily achievable.

For this work, the PyTorch framework has been chosen, due to the fact that it is
growing quickly in popularity (see [@fig:popularity]) and relatively new, so the
ecosystem is not fully developed and some utilities available for e.g.
TensorFlow are not available for PyTorch. Because of this, an introspection framework for training momnitoring is
judged to present the best value proposition for PyTorch users.

![Changes in popularity of different deep learning libraries in research. Data
  was collected by keyword search over ICLR submissions
  ([http://search.iclr2019.smerity.com/search](http://search.iclr2019.smerity.com/search/);
  analogously for 2018)](diagrams/framework_popularity/popularity.pdf){#fig:popularity width=70%}

## Publisher-Subscriber {#sec:pubsub}

The Publisher-Subscriber pattern (for a detailed overview [see @eugster2003])
is a pattern for distributed computation in which publishers publish messages
either directly to any subscribers which have registered interest in them, or to
a central authority orchestrating the exchange. Messages are generally
associated with one or more topics and subscribers register interest in
receiving messages on one or more topics.

The compontens are very loosely coupled; the subscribers need not even be aware
of the publishers at all, and the publishers' only interaction with their
subscribers is relaying messages through a uniform interface or through an
optional server. A graphical schema of one possible incarnation of this pattern
is shown in [@fig:pubsub].

![One possible implementation of the Publisher-Subscriber pattern.](diagrams/architecture_diagrams/pubsub.pdf){#fig:pubsub}

This project is not distributed, but can benefit from the loose coupling in
another way: Subscribers can be defined in terms of the kind of messages they
need to compute their metric, without knowing anything about where the messages
are coming from. Concretely, as long as the appropriate data is emitted from the
training process, subscribers can work without modifications with any possible
model.

Since real-world neural networks are trained on the GPU, and
communication between host and GPU memory is expensive, making this library
truly distributed is not an objective. However, the design will simplify
asynchronous computation of metrics in the future. The Python language does not
support true multithreading[^py-threading], but since the expensive part of the
work is running on the GPU while the host code is waiting, metric computation
could happen asynchronously on the GPU as well while the expensive forward or
backward passes through the network are running. This is not currently
implemented but can be added later, if more computationally demanding metrics
are to be explored.

[^py-threading]: The `multiprocessing` module allows for truly asynchronous
  computation and communication, but the inter-process-communication is more
  expensive than memory shared between threads.

In the context of neural network training, there is only one source of
information and hence only one publisher. Therefore, the message
server is folded into the singular publisher which extracts data from the
training model and sends messages to any interested subscribers.

## Overview of the library

The software is structured into several packages. The root package is `ikkuna`
which encapsulates all core functionality. All other packages and modules contain
utilites implemented for this work specifically, but will generally not be
relevant to other users. A survey of these tools will be given in
[@sec:other-tools].

The root package diagram is shown in [@fig:pack-diag-ikkuna]

![`ikkuna` package diagram](diagrams/class_diagrams/ikkuna.pdf){#fig:pack-diag-ikkuna
width=60%}

The `models` (see [@sec:pack-models]) subpackage contains a few exemplary neural
network definitions which are wired up with the library and can thus be used to
showcase the library's functionality. The `utils` (see [@sec:pack-utils])
subpackage contains miscellaneous utility classes and functions used throughout
the core library. Lastly, the `visulization` subpackage
([@sec:pack-visualization]) contains the plotting functionality to actually show
the metrics computed during the training process.

The most important bits of the software live in the `export` subpackage
([@sec:pack-export]). It implements the Publisher-Subscriber pattern. Extracting
data from the training process, defining subscriber functionality and messages
used for communication is done here.

### The `export` subpackage {#sec:pack-export}

The `export` subpackage contains the core part of the library, i.e. it provides the
classes that handle discovering the structure of the neural network model,
attaching the appropriate callbacks and intercepting method calls on the model
so the library is informed about everything entering and exiting the model and
its individual layers. It also contains the definition for the subscriber API,
i.e. the messages that subscribers can receive, synchronisation facilities
when multiple topics are needed by a subscriber, as well as the subscriber class
interface. The package diagram is displayed in [@fig:pack-diag-export].

The package comprises three subpackages or modules

Name            Function
----            --------
`export`        Publish data from an arbitrary model and send messages
                to registered subscribers
`messages`      Define message interface; i.e. what topics exist and which
                information a message must contain
`subscriber`    Define the base class for metric subscribers

Table: `ikkuna.export` functionalities

![`ikkuna.export` package diagram](diagrams/class_diagrams/export_package_diagram.pdf){#fig:pack-diag-export width=60%}

In in slight deviation from the Publisher-Subscriber framework as displayed in
[@fig:pubsub], the `export.Exporter` class ([@fig:class-diag-exporter]) is the sole
publisher of data. There's only one source of data during training, so it is
unnecessary to accomodate for multiple subscribers. The `Exporter` is informed
of the model with its methods `set_model()` and `set_loss()`, the latter of
which is only necessary if metrics which rely on training labels should be
displayed. It can accept a filter list of classes which are to be included when
discovering the modules in the model.  For instance, it could be desirable to
only observe layers which have weights and biases associated with them, not e.g.
normalisation or reshaping layers. The `Exporter` then traverses the model
(which is really just a tree structure of modules) and adds to each a callback
invoked when input enters the layer -- in order to retrieve activations -- and
when gradients are computed for the layer outputs. The callbacks also use cached
weights -- if present -- in order to publish updates to the weights.
Furthermore, it replaces a few of the model's methods with closure wrappers so
it can

* be notified when the model is set to training or testing mode (this
  switch disables or enables layers which only make sense during one of the
  phases[^batch-norm])
* increase its own step counter automatically when a new batch is seen
* add a parameter to the model's `forward()` which can be used be subscribers to
  temporarily turn off training mode and have it revert automatically. This is
  useful for subscribers which need to evaluate the model (i.e. feed data
  through it), but do not want to generate new messages for this occasion.
* intercept labels passed to the loss function during training and publish them
  as messages so the user need not concern himself with this task
* intercept the final output of the network. This could be realised
  alternatively by identifying the last module in the network.

The `Exporter` publishes the following information at each training step

* gradients for each module
* activations for each module
* weights and biases for each module that has these properties (e.g.
  convolutional or fully-connected layers)
* updates to the weights and biases from the last step to the current one,
  provided the module has these properties
*

[^batch-norm]: There are two built-in layers this applies to. One is the batch
  normalisation layer. It normalises the output of the previous layer with the
  mean and variance over the entire batch of data. The variance is not defined
  for single data point enters the layer, as could be the case during
  inference/testing time. The second case is the dropout layer, which randomly
  zeroes out a percentage of the previous layer's activations. This is used
  during training to prevent subsequent units from becoming correlated with a
  fixed set of units in the previous layer, instead of picking up patterns
  invariant of where in the input they occur. During inference time, this is
  turned off to make full use of the trained layers.

![`ikkuna.export.Exporter` class diagram](diagrams/class_diagrams/ikkuna.export.pdf){#fig:class-diag-exporter}


### The `models` subpackage {#sec:pack-models}

![`ikkuna.models` package diagram](diagrams/class_diagrams/models_package_diagram.pdf){#fig:pack-diag-models width=60%}

This package shown in [@fig:pack-diag-models] contains model definitions for
demonstration purposes and for experimentation. Three architectures are
currently implemented:

#. A minified version of AlexNet, since the original architecture requires
   larger images [@krizhevsky2012imagenet]. The code is adapted from the PyTorch
   implementation.
#. DenseNet [@huang2017densely]. The implementation is basically the one from
   [@pleiss2017memory] [^densenet-impl] with minor modifications
#. ResNet [@he2016deep]. This implementation comes from GitHub user liukang
   [^resnet-impl] and can handle CIFAR10-sized images of 32 pixels per side, as
   opposed to most implementaions that are geared towards ImageNet examples which
   are much larger.

[^densenet-impl]: At the time of writing, the implementation is available here:
  [https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py](https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py).
  The licensing is unclear as the author references the original BSD-licensed
  implementation at
  [https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
  which was licensed by PyTorch core contributor Soumith Chintala. However, the
  code does not reproduce the BSD license text and can thus only be inspired by
  the original but cannot contain any of the code verbatim. It would require
  careful examination in order to determine whether this is the case.

[^resnet-impl]: The implementation is MIT-licensed.
  [https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)

All models are modified such that their training can be supervised by the
library.

### The `utils` subpackage {#sec:pack-utils}

As shown in [@fig:pack-diag-utils], this package defines classes for traversing
a model into a hierarchical tree of layers (called _modules_ in PyTorch lingo),
structures for adding information to PyTorch's `Module` class, and a set of
miscellaneous functions for

#. Seeding random number generators to make experiments reproducible (see [@sec:reproducibility])
#. Creating instances of weight optimizers by named
#. Initialize the weights of any model
#. Loading datasets

![`ikkuna.utils` package diagram](diagrams/class_diagrams/utils_package_diagram.pdf){#fig:pack-diag-utils width=60%}

Additionally, it contains the `numba` module which is inteded to allow
interoperability with the Numba library[^numba]. While currently not used due to
the incomplete nature of the Numba GPU array interface, it could enable
leveraging Numba in the future without transferring data to the CPU.

[^numba]: [https://numba.pydata.org/](https://numba.pydata.org/). Numba is a
  library for transforming high-level Python code into performant compiled code
  and for allowing to use the CUDA library from Python with Python arrays.
  This enables performance improvements for numeric calculations, but there is
  only a limited set of higher-level functions implemented on GPU arrays.

### The `visualization` subpackage {#sec:pack-visualization}

This package contains only a single module: `backend`. It defines the classes
shown in [@fig:class-diag-backend]. The module serves as an abstraction over
plotting libraries so that metrics need not concern themselves with how to
actually show the data.

A given metric will compute its value and dispatch it to its visualization
backend, which can currently accept scalar and histogram data. The metric class
itself need not care about how it is going to be displayed.

![`ikkuna.visualization` package diagram](diagrams/class_diagrams/visualization_package_diagram.pdf){#fig:pack-diag-visualization width=60%}

![Class diagram for classes in `ikkuna.visualization`](diagrams/class_diagrams/visualization_class_diagram.pdf){#fig:class-diag-backend}

For running the library locally, a `matplotlib`-based backend has been
implemented. Plotting routines from this library open a window directly on the
system executing the software. In practice however, deep learning code will be
executed remotely on a server with adequate compute capability and the
programmer connected via SSH. While it is possible to have remote windows show
up locally on Linux-based systems by use of X11-Forwarding, this is generally
slow and not useful for interactivity. An example is shown in [@fig:example_mpl] To remedy this issue, a plotting backend
on TensorBoard is also provided ([@sec:existing-apps]). The plotting data is
generated and processed on the remote system, but served over the web so it can
be viewed and interacted with locally (provided the network is configured so that the server responds to HTTP requests). An example is shown in [@fig:example_tb]

![Exemplary view of a matplotlib figure forwarded over SSH](diagrams/software_screens/example_mpl.png){#fig:example_mpl}

![Exemplary view of a TensorBoard session](diagrams/software_screens/example_tb.png){#fig:example_tb}


### Miscellaneous tools {#sec:other-tools}

There are a few modules which simplify development with the library but are not
part of the distribution obtained from the Python Pacakge Index or by running
the setup script.

The `train` package defines a `Trainer` class which encapsulates all the logic
and parameters needed to train a neural network on one of the datasets provided
with PyTorch. The class's capabilities include the following

* Look up model and dataset by name
* Bundle all hyperparameters
* hook the `Exporter` into the model for publishing data
* configure the optimisation algorithm to use for training
* train the model for one batch

The `Trainer` class is used in the main script (`main.py`), which serves as a command line
interface to the library while developing. When trying out the library, it can
also be used as an initial starting point.

Parameter               Explanation
---------               -----------
`-m`, `--model`         Model class to train
`-d`, `--dataset`       Dataset to train on.
                        Possible choices: `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`
`-b`, `--batch-size`    Default: 128
`-e`, `--epochs`        Default: 10
`-o`, `--optimizer`     Optimizer to use. Default: `Adam`
`-a`, `--ratio-average` Number of ratios to average for stability (currently unused).
                        Default: 10
`-s`, `--subsample`     Number of batches to ignore between updates. Default: `1`
`-v`, `--visualisation` Visualisation backend to use.  Possible choices: `tb`, `mpl`.
                        Default: `tb`
`-V`, `--verbose`       Print training progress. Default: `False`
`--spectral`-norm       Use spectral norm subscriber on weights. Default: `False`
`--histogram`           Use histogram subscriber(s)
`--ratio`               Use ratio subscriber(s)
`--test-accuracy`       Use test set accuracy subscriber. Default: `False`
`--train-accuracy`      Use train accuracy subscriber. Default: `False`
`--depth`               Depth to which to add modules. Default: `-1`

Table: Named arguments to `main.py`

The library can be installed to the local Python environment by use of the
provided setuptools script (`setup.py`). It can also be downloaded from PyPI by use of the package manager `pip`:

```{.python}
pip install ikkuna
```

### Plugin Infrastructure


### Reproducibility {#sec:reproducibility}
