"""Module for any LR-style experiment."""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
import sklearn_crfsuite
import numpy as np
import tensorflow as tf
import pandas as pd
import spacy
import crf_processing
from enso.experiment import ClassificationExperiment
from enso.experiment.grid_search import GridSearch
from enso.registry import Registry, ModeKeys


@Registry.register_experiment(
    ModeKeys.RATIONALIZED, requirements=[("Featurizer", "PlainTextFeaturizer")]
)
class BiasedLogit(ClassificationExperiment):
    """Implementation of a grid-search optimized Logistic Regression model."""
    NLP = None

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)

        self.binarizer = LabelBinarizer()
        self.n_epochs = 20
        self.batch_size = 2
        self.display_step = 1

        if self.NLP is None:
            self.NLP = spacy.load('en_vectors_web_lg')
        # Input placeholders
        x = tf.placeholder(tf.float32, [None, 300]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 1]) # 0-9 digits recognition => 10 classes
        # lets initalize some parameters
        self.W = tf.Variable(tf.zeros([300,1]))
        self.b = tf.Variable(tf.zeros([1]))
        
        yhat = tf.nn.sigmoid(x @ self.W + self.b)
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(yhat), reduction_indices=1))

        # Initialize Adam optimizer
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

        # Putting myself at the mercy of the TF gods here. Just initializing the whole shebang
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()


    def spacy_vectorize(self, texts):
        return [i.vector for i in self.NLP.pipe(texts, disable=['parser', 'ner', 'textcat'])]


    def fit(self, X, Y):
        print(Y[0])
        # placeholder for processing code here
        X_train = self.spacy_vectorize(X)
        Y_train = self.binarizer.fit_transform([y[1] for y in Y])

        with tf.Session() as sess:

            # Running init
            sess.run(init)
            
            # Training loop
            for epoch in range(n_epochs):
                loss_avg = 0
                n_batches = int(len(X_train)/self.batch_size)
                for i in range(n_batches):
                    batch_xs = X_train[i*self.batch_size:(i+1)*self.batch_size]
                    batch_ys = Y_train[i*self.batch_size:(i+1)*self.batch_size]
                    
                    # Optimization op
                    _, c = sess.run([optimizer, cost],
                            feed_dict={x: batch_xs,
                                       y: batch_ys})
                    # Loss average
                    loss_avg += c / total_batch
                    if (epoch+1) % display_step == 0:
                        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    def predict(self, X, **kwargs):
        X_vectors = self.spacy_vectorize(X)
        
        X_vectors = np.stack([X_vectors, X_vectors])
        

        feed_dict = {x: X_vectors}
        probas = tf.run(yhat, feed_dict)

        # Need to slack in the binarizer here to get labels
        labels = self.model.classes_
        
        return pd.DataFrame({label: probas[:, i] for i, label in enumerate(labels)})
