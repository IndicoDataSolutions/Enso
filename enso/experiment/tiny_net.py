import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np

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
            w_init=self.rationale_proto
        )(x)
        rationale_log_probas = jax.nn.log_sigmoid(rationale_logits)
        rationale_probas = jnp.exp(rationale_log_probas) # (batch, seq, n_classes)
        masked_rationale_probas = rationale_probas * jnp.expand_dims(length_mask, -1)
        clf_embed = jnp.einsum('bsh,bsc->bch', x, rationale_probas) # (batch, hidden, classes)
        clf_logits = hk.Linear(
            output_size=1
        )(clf_embed)
        clf_logits = jnp.squeeze(clf_logits)
        clf_log_probas = jax.nn.log_softmax(clf_logits)
        return rationale_log_probas, clf_log_probas

