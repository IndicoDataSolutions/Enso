#"""File for storing the featurizers that indico offers via API."""
from indicoio.custom import vectorize
import indicoio.config
from tqdm import tqdm

from enso.registry import Registry, ModeKeys
from enso.featurize import Featurizer
from finetune import Classifier

try:
    from finetune.base_models.gpc.model import GPCModel
except:
    pass
feat_modes = ["final_state", "clf_tok", "mean_state", "mean_tok", "max_state", "max_tok"]

@Registry.register_featurizer(ModeKeys.ANY)
class BaseIndicoFeaturizer(Featurizer):
    """
    Base model from which all indico `Featurizer`'s inherit.
    """
    domain = None
    sequence = False

    def featurize_batch(self, X, batch_size=8, **kwargs):
        """
        :param X: `pd.Series` that contains raw text to featurize
        :param batch_size: int number of examples to process per batch
        :returns: list of np.ndarray representations of text
        """
        all_features = []
        for i in tqdm(range(0, len(X), batch_size)):
            chunk_data = list(X[i:i + batch_size])
            all_features.extend(vectorize(
                chunk_data, domain=self.domain, sequence=self.sequence, **kwargs
            ))
        return all_features

@Registry.register_featurizer(ModeKeys.ANY)
class GPCFinalStateFeaturizer(Featurizer):
    sequence = False
    feat_mode = "final_state"

    def featurize_batch(self, X, batch_size=8, **kwargs):
        model = Classifier(
            batch_size=10,
            base_model=GPCModel,
            base_model_path="conv_base_30jun.jl",
            xla=False,
            feat_mode=self.feat_mode
        )
        
        a = model.featurize(X)
        return [x for x in a]

@Registry.register_featurizer(ModeKeys.ANY)
class GPCClfTokFeaturizer(GPCFinalStateFeaturizer):
    feat_mode = "clf_tok"
    
@Registry.register_featurizer(ModeKeys.ANY)
class GPCMeanStateFeaturizer(GPCFinalStateFeaturizer):
    feat_mode = "mean_state"

@Registry.register_featurizer(ModeKeys.ANY)
class GPCMeanTokFeaturizer(GPCFinalStateFeaturizer):
    feat_mode = "mean_tok"

@Registry.register_featurizer(ModeKeys.ANY)
class GPCMaxStateFeaturizer(GPCFinalStateFeaturizer):
    feat_mode = "max_state"

@Registry.register_featurizer(ModeKeys.ANY)
class GPCMaxTokFeaturizer(GPCFinalStateFeaturizer):
    feat_mode = "max_tok"
                


    
@Registry.register_featurizer(ModeKeys.ANY)
class IndicoStandard(BaseIndicoFeaturizer):
    """Featurizer that uses indico's standard features."""
    domain = 'standard'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoSentiment(BaseIndicoFeaturizer):
    """Featurizer that uses indico's sentiment features."""
    domain = 'sentiment'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoTopics(BaseIndicoFeaturizer):
    """Featurizer that uses indico's topics features."""
    domain = 'topics'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoFinance(BaseIndicoFeaturizer):
    """Featurizer that uses indico's finance features."""
    domain = 'finance'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoTransformer(BaseIndicoFeaturizer):
    """Featurizer that uses indico's transformer features."""
    domain = 'transformer'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoEmotion(BaseIndicoFeaturizer):
    """Featurizer that uses indico's emotion features."""
    domain = 'emotion'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoFastText(BaseIndicoFeaturizer):
    """Featurizer that uses indico's fasttext features."""
    domain = 'fasttext'


@Registry.register_featurizer(ModeKeys.ANY)
class IndicoElmo(BaseIndicoFeaturizer):
    """Featurizer that uses indico's finance features."""
    domain = 'elmo'


# NOTE: To remain undocumented for now until further API design decisions are made
@Registry.register_featurizer(ModeKeys.ANY)
class IndicoElmoSequence(BaseIndicoFeaturizer):
    """Featurizer that uses indico's finance features."""
    domain = 'elmo'
    sequence = True

@Registry.register_featurizer(ModeKeys.ANY)
class IndicoTransformerSequence(BaseIndicoFeaturizer):
    """Featurizer that uses indico's transformer sequence features."""
    domain = 'transformer'
    sequence = True

@Registry.register_featurizer(ModeKeys.ANY)
class IndicoStandardSequence(BaseIndicoFeaturizer):
    """Featurizer that uses indico's standard sequence features"""
    domain = 'standard'
    sequence = True
