# ENSO #
A library for running benchmarks on the efficacy of text representation algorithms across a wide variety of datasets.

This library is specifically engineered to make the process of adding new featurization techniques, new evaluation
datasets, and new post-processing techniques easily benchmarkable.

## Main Files ##
#### featurize.py ####
Responsible for looking through data in `Enso/Data` and pre-computing features to store in `Enso/Features`.

Finds featurizers that should be active by searching through `Enso/Featurize` and calling the featurize method of
each class that inherits from the `Featurizer` base class found in `Enso/Featurize/__init__.py`
#### experiment.py ####
Responsible for running all experiments found in `Enso/Experiments`. Ideally this is another file that shouldn't really be
altered. Configuring standard pieces like which featurizers/which experiments to run is all done through `config.py`.
#### visualize.py ####
Responsible for generating, displaying, and potentially saving any visualizations you want generated based on a
particular set of results. The parameters for the visualizations are set within `config.py`
#### config.py ####
Should be frequently edited to change the specifics of how `experiment.py` is run. The main parameters here are:
 - `DATASETS`: A list of the datasets that you want to include in your experiments
 - `FEATURES`: A list of pre-computed features to include in your experiments. Only features for your specified datasets will be used.
 *Note* The name of the featurizer will dictate the name of the features file.
 - `EXPERIMENTS`: A list of the experiments to run on top of the feature sets that have selected
 - `METRICS`: A list of metrics you'd like to see for the combination of experiments being run
 - `TEST_SETUP`: More detailed test information, likely to vary quite a bit from run to run.
   - `train_sizes`: A list of training set sizes to be experimented with. These cannot be varied with the datasets in question
   - `n_splits`: The number of CV splits to perform on each run
 - `VISUALIZATIONS`: A list of the visualizations we want to compute and create
 - `VISUALIZATION_SETUP`: More detailed visualization information with plenty of visualization-specific options
   - `save`: Boolean saying whether or not results should be saved to the relevant results directory
   - `display`: Boolean saying whether or not visualizations should be displayed as they are generated
   - `<visualization_name>`: Mapping of all the visualization-specific options you want to pass

If a list field is missing then the default assumption will be to run the experiment on all relevant members of that class

## Data Management ##
Because of the large size of both the dataset and featurized version of those datasets, we're using `privvy` for
data management. To make sure this works, whenever you add a new dataset, or featurize a new dataset add the appropriate
information to the `.privvy` file in the repo.

##### Uploading to s3 #####
`sudo -E privvy-push`

##### Downloading from s3 #####
`sudo -E privvy-pull`

## Data Formatting ##
In order to ensure interoperability the onus is on the person adding the dataset to ensure that it complies with Enso
standards. In general these standards should be quite easy to adhere to.
#### Classification Tasks ####
For classification tasks the data format should be a csv with one column labeled `Text`, and any additional targets
labeled `Target_<n>` where `n` is a unique number representing which target it is. This is meant primarily to enforce
a uniform data contract, and also to ensure that no learning can be done on top of the column labels

## Sections ##
#### Data ####
Place for storing all raw datasets. These should only be used for featurization. This should not be accessible from the
experiments folder. Look to existing datasets in this folder for examples of how a dataset should be formatted. This
section is further subdivided into sections based on the type of task with the current supported types being `Classify`,
`Regress`, and `Match`.
#### Features ####
Place for storing pre-computed features. This is what the `Enso/Experiments` draws from and all that any experiment should
have access to. These should all effectively be copies of the files in `Enso/Data` with the major text field being replaced
by fixed-length vectors. As there are multiple approached for featurization the name of the `Featurizer` is appended to
the name of the dataset.
#### Featurize ####
Place to add all featurizers. This is run through the `featurize.py` file, and should inherit
from the base class found within `Enso/Featurize/__init__.py`.
#### Experiments ####
Place for writing any new experiments. These should all follow the format outlined by the `Experiment` base classes found
in `Enso/Experiments/__init__.py`. These experiments are specific to the task being undertaken. As such you'l find not just
a signle base class, but three base classes that match with the high level `Classify`, `Regress`, and `Match` tasks.
#### Metrics ####
Place to detail different metrics you'd like returned from a specific set of experiments. These metrics are specific to
the problem being approached and so should inherit form the appropriate base class found (unsurprisingly) in
`Enso/Metrics/__init__.py`. If you want to support a new type of problem you should create a new metric base class and
provide at least a single extension of it.
#### Visualize ####
This is a place where we detail all of the visualizations that `Enso` currently supports. Similarly to each other section,
the base models are defined within `Visualize/__init__.py`. In there you will find base classes for `Classification`,
`Regression`, and `Matching` problems, just as you'd expect given the problem sets supported. Generically, each visualization
will support any number of arguments passed in via `config.py`
#### Results ####
Any time an experiment is run, the results as well as the configuration of the experiment will be stored within `Enso/Results/<timestamp>`.
The timestamp reflects when the test was run, and within each directory you should find the following:
 - `Graphs/`: Generated graphs useful for displaying results of the experiment run
 - `config.py`: The configuration that was used to generate these results
 - `Results/<dataset>/`: Raw results generated by whatever experiments were specified in `config.py`. Organized by dataset.

## Results Format ##
There's a lot of information we're trying to store in the `Results.csv` file for each test run. There are a few ways to
approach this problem, and we've opted for one where there's quite a lot of repeated information, but the data
structure is quite usable for any visualization applications that the user wants to run on the backend. The specific list
and order of fields is below. This is set within the `experiment.py` file in the `Experimentation.columns` attribute
`ID,Dataset,Featurizer,Experiment,Metric,TrainSize,Result`

## Visualization Details ##
Visualizations are made to accept a set of kwargs, and a results csv, then generating appropriate visualizations.
These visualizations are then returned to the `visualize.py` file and display as well as saving are handled by the
top-level `Vizualization` metaclass. The only two arguments that are fed directy into this top level are the `save`
and `display` paramaters

#### Classification Visualization ####
When it comes to visualization classification we run into a few gaps between the native results format and what a standard
visualization algorithm expects. There are two main areas here, related to the handling of cross-validation splits and
accuracy metrics provided on multiple classes. These two factors can either be included as an axis of visualization by passing
`Split` or `Class` into the `VISUALIZATION_OPTIONS` dictionary within `config.py`, or they can use one of the strategies
outlined below:
 - Cross-validation Splits:
   - `mean`: replace each result with the average of all results. This will reduce the size of your data frame by a factor
   of <n_splits>
 - Classes:
   - `merge`: replace multiple class score with a single score. The will check to ensure that all class scores are identical
   before merging. This only makes sense for things like `RocAuc` on binary classification tasks.
