![enso](https://i.imgur.com/Oj3O1xQ.jpg)

Enso
----
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

Download the included datasets by running:

```
python -m enso.download
```

Usage and Workflow
------------------
Although there are other effective approaches to applying transfer learning to natural language processing, it's built on the assumption that the approach to "transfer learning" adheres to the below flow.  This approach is designed to replicate a scenario where a pool of unlabeled data is available, and labelers with subject matter expertise have a limited amount of time to provide labels for a subset of the unlabeled data.

- All examples in the dataset are "featurized" via a pre-trained source model (`python -m enso.featurize`)
- Re-represented data is separated into train and test sets
- A fixed number of examples from the train set is selected to use as training data via the selected sampling strategy
- The training data subset is optionally over or under-sampled to account for variation in class balance
- A target model is trained using the featurized training examples as inputs (`python -m enso.experiment`)
- The target model is benchmarked on all featurized test examples
- The process is repeated for all combinations of featurizers, dataset sizes, target model architectures, etc.
- Results are visualized and manually inspected (`python -m enso.visualize`)

For detailed API documentation, refer to [enso.readthedocs.org](https://enso.readthedocs.org).

Contributions in the form of pull requests or issues are welcome!

Sample result visualization included below:

![Enso Results Visualization](https://i.imgur.com/T3I1T7R.png)
