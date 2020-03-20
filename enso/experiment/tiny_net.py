import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class BetterLabelBinarizer():

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit X using the LabelBinarizer object
        self.lb.fit(X)
        # Save the classes
        self.classes_ = self.lb.classes_

    def fit_transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit + transform X using the LabelBinarizer object
        Xlb = self.lb.fit_transform(X)
        # Save the classes
        self.classes_ = self.lb.classes_
        if len(self.classes_) == 2:
            Xlb = np.hstack((1 - Xlb, Xlb))
        return Xlb

    def transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.lb.transform(X)
        if len(self.classes_) == 2:
            Xlb = np.hstack((1 - Xlb, Xlb))
        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = np.array(Xlb)
        if len(self.classes_) == 2:
            X = self.lb.inverse_transform(Xlb[:, 0])
        else:
            X = self.lb.inverse_transform(Xlb)
        return X
    

class TinyNet(hk.Module):
    def __init__(self, kernel_size: int, n_classes: int, rationale_proto: np.ndarray):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_classes = n_classes
        self.rationale_proto = rationale_proto
    
    def __call__(self, x: jnp.ndarray, length_mask: jnp.ndarray):
        """
        x: (batch, seq, n_hidden)
        """
        rationale_logits = hk.Conv1D(
            output_channels=self.n_classes,
            kernel_shape=self.kernel_size,
            w_init=hk.initializers.Constant(self.rationale_proto)
        )(x)
        rationale_log_probas = jax.nn.log_sigmoid(rationale_logits)
        rationale_probas = jnp.exp(rationale_log_probas) # (batch, seq, n_classes)
        length_mask = jnp.expand_dims(length_mask, -1)
        masked_rationale_probas = rationale_probas * length_mask
        lengths = jnp.sum(length_mask, axis=1)
        clf_embed = jnp.einsum('bsh,bsc->bhc', x, masked_rationale_probas) / np.reshape(lengths, [x.shape[0], 1, 1])  # (batch, hidden, classes)
        flat_clf_embed = jnp.reshape(clf_embed, clf_embed.shape[:1] + (clf_embed.shape[1] * clf_embed.shape[2],))
        clf_logits = hk.Linear(
            output_size=self.n_classes,
        )(flat_clf_embed)
        clf_logits = jnp.squeeze(clf_logits)
        clf_log_probas = jax.nn.log_softmax(clf_logits)
        return rationale_log_probas, clf_log_probas

