# ENSO #
A library for running benchmarks on the efficacy of text representation algorithms across a wide variety of datasets.

This library is specifically engineered to make the process of adding new featurization techniques, new evaluation
datasets, and new post-processing techniques easily benchmarkable.

## Main Files ##
#### BUILD ####
A bazel build file for ensuring that there's no data cross-contamination and maintaining build dependencies.

Ideally this file shouldn't be changed as it only gives each main file access to the files it needs access to.

If you end up changing this significantly there's a good chance you're now comparing apples to oranges when it comes
to whatever model you'd be benchmarking.
#### featurize.py ####
Responsible for looking through data in `Enso/Data` and pre-computing features to store in `Enso/Features`.

Finds featurizers that should be active by searching through `Enso/Featurize` and calling the featurize method of
each class that inherits from the `Featurizer` base class found in `Enso/Featurize/__init__.py`
#### experiment.py ####
Responsible for running all experiments found in `Enso/Experiments`. Ideally this is another file that shouldn't really be
altered. Configuring standard pieces like which featurizers/which experiments to run is all done through `config.py`.

The configuration file detailing the experiment can either be simply a file called `config.py` sitting in the base directory
or it can be passed in at runtime as a command line argument as so: `python experiment.py <path_to_config>`
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
#### Results ####
Any time an experiment is run, the results as well as the configuration of the experiment will be stored within `Enso/Results/<timestamp>`.
The timestamp reflects when the test was run, and within each directory you should find the following:
 - `Graphs/`: Generated graphs useful for displaying results of the experiment run
 - `config.py`: The configuration that was used to generate these results
 - `Results/<dataset>/`: Raw results generated by whatever experiments were specified in `config.py`. Organized by dataset.

## Results Format ##
There's a lot of information we're trying to store in the `Results.json` file for each test run. This requires a somewhat
large and cumbersome data structure to ensure everything is being captured effectively. This format is detailed below
primarily for assistance in working with this format either in visualization or in any other application making use
of these output files.
```{<Dataset>:
        # The specific featurizer used
        <Featurizer>: {
            # Training set size is a number refering to the number of examples in the training set
            <Train set size>: {
                # Experiment refers to the model being run, i.e. GridSearchLR
                <Experiment>: {
                    # Each entry in this list represents a different CV run
                    [{<Metric>: #value, <Metric>: #value}, ...]
                }
            }
        }
    }
```
