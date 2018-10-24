General ideas
=============
1. Which offline interpretation tools can be adapted to work online?

Live monitoring quantities (courtesy of http://cs231n.github.io/neural-networks-3/)
===================================================================================
1. Training/Test/Validation accuracy (also use smoothed version)
2. Loss (oscillations, trajectory)
3. Ratio between weight magnitude and weight update magnitude
4. Activation distribution for each layer (should cover entire output range of
   activation function)
5. Feature maps of early layers (especially for image nets)
6. Mutual information (Tishby), saturation
   (https://towardsdatascience.com/information-theory-of-neural-networks-ad4053f8e177)
7. Correlation within layers or accross layers
8. Edginess of early layers

Prior art
=========
https://aetros.com/


Thesis
======
Introduction
------------
* What is thesis about?
* Why is thesis useful
	* Motivation
* What are the results

Existing software
-----------------
* Tensorboard
* Visdom
* Beholder

Description of the software
-------
* Achitecture
* How to use
* Challenges
* Future plans

Hypothesised metrics
---------------------
* What is the metric?
* What problem should it diagnose?
* What is the theoretical justification?
* Which experiments show or do not show promise?