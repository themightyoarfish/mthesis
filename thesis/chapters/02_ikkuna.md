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
agnosticity goal, however. As such, a very loose coupling between model code and
visualizations is desired. Not only will this aid in extending the framework to
different deep learning libraries, but it is also a prerequisite for allowing
for modular, self-contained visualizations or metrics which can be installed and
used separately and independently of specific model code.. The
Publisher-Subscriber design pattern has been chosen for
these reasons.

## Publisher-Subscriber

The Publisher-Subscriber pattern (for a detailed overview [see @eugster2003])
is a pattern for distributed computation in which publishers publish messages
directly to any subscribers which have registered interest in them. The
compontens are very loosely coupled; the subscribers need not even be aware of
the publishers at all, and the publishers' only interaction with their
subscribers is relaying messages through a uniform interface.

This project is not distributed, but can benefit from the loose coupling in
another way: Subscribers can be defined in terms of the kind of messages they
need to compute their metric, without knowing anything about where the messages
are coming from. Concretely, as long as the appropriate data is emitted from the
training process, subscribers can work without modifications with any possible
model.

## Deep Learning frameworks {#sec:dl-frameworks}

The currently available deep learning libraries can be located on a spectrum
between define-by-run and  define-and-run. The first extreme would be a
framework such as PyTorch [@paszke2017automatic] or Chainer [@tokui2015chainer],
where there exist no two distinct execution phases -- just like in an ordinary
matrix library like NumPy, each statement immediately returns or operates on an
actual value. By contrast, graph-based frameworks like TensorFlow require
specifying the model graph in a domain-specific language (TensorFlow has Python,
Java and C++ APIs, Caffe uses Prototxt files), compile it to a different
representation and the the model is run and trained in a second phase. While
this enables graph-bases optimizations, the main downsides are that control flow
cannot use the host languade features, but must be done with the API used for
defining models and secondly, that halting execution at aribtrary points in the
training is not possible, since the actual training is not happening in the host
language, but is more often handed off to lower-level implementations in its
entirety. This makes debugging much less ergonomic.

All frameworks have in common that they build a graph representation of the
model. Nodes in the graph are operations while edges are data flowing between
operations. This allows naturally parallelizing independent computations. To
compute gradients, the graph can be traversed backwards from the output node by
applying the chain rule of differentiation. Define-and-run frameworks like
TensorFlow create the graph explicitly; the user uses the API to do exactly
this. The graph -- once compiled -- is fixed for the entire training process.
PyTorch on the other hand implicitly records all operations and also overloads
operators for this purpose.  The graph is thus recreated for each propagation
through the network. This precludes some optimizations, but makes dynamically
changing networks easily achievable, which is especially handy for recurrent
networks.

For this work, the PyTorch framework has been chosen, due to the fact that it is
growing quickly in popularity (see [@fig:popularity]) and relatively new, so
the ecosystem is not as developed, so an introspection framework is judged to be
a useful to many users.

![Changes in popularity of different deep learning libraries in research. Data
  was collected by keyword search over ICLR submissions
  ([http://search.iclr2019.smerity.com/search](http://search.iclr2019.smerity.com/search/);
  analogously for 2018)](../diagrams/popularity.pdf){#fig:popularity width=70%}
