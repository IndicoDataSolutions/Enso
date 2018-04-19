.. Enso documentation master file, created by
   sphinx-quickstart on Thu Apr 19 10:27:32 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Enso's documentation!
================================

.. module:: enso

An overview of the developer interface to Enso.


Enso workflow
==============

Enso is tool intended to provide a standard interface for the benchmarking of embedding and transfer learning methods for natural language processing tasks.  Although there are other effective approaches to applying transfer learning to natural language processing, it's built on the assumption that the approach to "transfer learning" adheres to the below flow.  This approach is designed to replicate a scenario where a pool of unlabeled data is available, and labelers with subject matter expertise have a limited amount of time to provide labels for a subset of the unlabeled data.

- All examples in the dataset are "featurized" via a pre-trained source model (`python -m enso.featurize`)
- Re-represented data is separated into train and test sets
- A fixed number of examples from the train set is selected to use as training data via the selected sampling strategy
- The training data subset is optionally over or under-sampled to account for variation in class balance
- A target model is trained using the featurized training examples as inputs (`python -m enso.experiment`)
- The target model is benchmarked on all featurized test examples
- The process is repeated for all combinations of featurizers, dataset sizes, target model architectures, etc.
- Results are visualized and manually inspected (`python -m enso.visualize`)


Featurization
=============
.. autoclass:: enso.featurize.Featurizer
    :inherited-members:


Target Model Training
=====================

Base Classes
-------------
.. autoclass:: enso.experiment.Experiment
    :inherited-members:

.. autoclass:: enso.experiment.ClassificationExperiment
    :inherited-members:

.. autoclass:: enso.experiment.grid_search.GridSearch
    :inherited-members:


Included Experiments:
------------------------------

.. autoclass:: enso.experiment.logistic_regression.LogisticRegressionCV
    :inherited-members:

.. autoclass:: enso.experiment.naive_bayes.NaiveBayes
    :inherited-members:

.. autoclass:: enso.experiment.random_forest.RandomForestCV
    :inherited-members:

.. autoclass:: enso.experiment.svm.SupportVectorMachineCV
    :inherited-members:


Visualization of Results
========================

Base Classes
------------
.. autoclass:: enso.visualize.Visualizer
    :inherited-members:

Included Visualizers
---------------------
.. autoclass:: enso.visualize.facet.FacetGridVisualizer
    :inherited-members:
