.. Enso documentation master file, created by
   sphinx-quickstart on Thu Apr 19 10:27:32 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Enso
================================

.. module:: enso

Enso is tool intended to provide a standard interface for the benchmarking of embedding and transfer learning methods for natural language processing tasks.


Installation
------------
Enso is compatible with Python 3.4+.

You can install `enso` via pip:

```bash
pip install enso
```

or directly via `setup.py`:

```
git clone git@github.com:IndicoDataSolutions/Enso.git
python setup.py install
```


Enso usage and workflow
==============

Enso is tool intended to provide a standard interface for the benchmarking of embedding and transfer learning methods for natural language processing tasks.  Although there are other effective approaches to applying transfer learning to natural language processing, it's built on the assumption that the approach to "transfer learning" adheres to the below flow.  This approach is designed to replicate a scenario where a pool of unlabeled data is available, and labelers with subject matter expertise have a limited amount of time to provide labels for a subset of the unlabeled data.

* Download pre-ETL'ed source datasets for testing (`python -m enso.download`)
* All examples in the dataset are "featurized" via a pre-trained source model (`python -m enso.featurize`)
* Re-represented data is separated into train and test sets
* A fixed number of examples from the train set is selected to use as training data via the selected sampling strategy
* The training data subset is optionally over or under-sampled to account for variation in class balance
* A target model is trained using the featurized training examples as inputs (`python -m enso.experiment`)
* The target model is benchmarked on all featurized test examples
* The process is repeated for all combinations of featurizers, dataset sizes, target model architectures, etc.
* Results are visualized and manually inspected (`python -m enso.visualize`)

Running Experiments with Enso
=============================
Each component of Enso is designed to be extensible and customizable.  Base classes for :class:`enso.Featurizer`, :class:`enso.Sampler`, :class:`enso.Experiment`, :class:`enso.Metric` and :class:`enso.Visualizer` are provided in order to enable anyone to implement and test their own ideas.  Subclass those base classes and modify `enso/config.py` to run your own experiments, and consider contributing the results back to the community to help other community members test against better baselines.

Enso Configuration
==================

Experiment settings are managed through the modification of `enso/config.py`. The main parameters of interest are:

 * `DATASETS`: A list of the datasets that you want to include in your experiments.
 * `FEATURES`: A list of pre-computed features to include in your experiments. Only features for your specified datasets will be used.
 * `EXPERIMENTS`: A list of the experiments to run on top of the feature sets that have selected.
 * `METRICS`: A list of metrics you'd like to see for the combination of experiments being run
 * `TEST_SETUP`: More detailed test information, likely to vary quite a bit from run to run.
     * `train_sizes`: A list of training set sizes to be experimented with.
     * `n_splits`: The number of CV splits to perform on each run.
 * `VISUALIZATIONS`: A list of the visualizations to create for result visualization.
 * `VISUALIZATION_SETUP`: More detailed visualization information with visualization-specific options.
     * `<visualization_name>`: Mapping of all the visualization-specific options you want to pass

Dataset Formatting
==================
In order to be a valid dataset, each dataset csv in the `Data` folder must include a `"Text"` column and a `"Target"` column.  For now, the "Target" column must be a string class label.  No rows may have missing values.

Featurization
=============

Base Classes
------------
.. autoclass:: enso.featurize.Featurizer
    :inherited-members:


Local Featurizers
-----------------
These Featurizers will be run on your local machine to embed training and test examples.

.. autoclass:: enso.featurize.spacy_features.SpacyCNNFeaturizer
    :inherited-members:

.. autoclass:: enso.featurize.spacy_features.SpacyGloveFeaturizer
    :inherited-members:

Hosted Featurizers
--------------------
The organization behind `enso`, indico, hosts a variety of pre-trained models that you can employ as source models for your experiments. These API wrappers assume that an `INDICO_API_KEY` env variable is present in order to authenticate calls made to the indico API. If you would like to test / benchmark indico's hosted embeddings on larger data volumes, reach out to contact@indico.io and inquire about free API credit for academic use.

.. autoclass:: enso.featurize.indico_features.IndicoStandard
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoSentiment
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoFinance
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoTopics
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoTransformer
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoEmotion
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoFastText
    :inherited-members:

.. autoclass:: enso.featurize.indico_features.IndicoElmo
    :inherited-members:


Sampling
========
Apply a strategy to select training examples to provide to the target model.

Base Classes
------------
.. autoclass:: enso.sample.Sampler
    :inherited-members:

Included Samplers
-----------------
.. autoclass:: enso.sample.random_sampler.Random
    :inherited-members:

.. autoclass:: enso.sample.orthogonal_sampler.Orthogonal
    :inherited-members:

.. autoclass:: enso.sample.kcenter_sampler.KCenter
    :inherited-members:


Resampling
==========
After a subset of examples is selected, certain examples may be duplicated or removed to adjust
the class frequency statistics of the training data.

Included Resampling Options
---------------------------
.. autofunction:: enso.resample.oversample


Target Model Training
=====================
After featurization and sampling, a target model is trained on the selected training data.
Each `Experiment` must be self-contained -- if hyperparameter selection is required, it should
be packaged as part of the `Experiment` child class.

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
.. autoclass:: enso.visualize.facets.FacetGridVisualizer
    :inherited-members:
