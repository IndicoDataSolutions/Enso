from finetune import DocumentLabeler
from finetune.base_models import RoBERTa
from finetune.base_models.bert.model import BERTModelMultilingualCased
from finetune.base_models.huggingface.models import HFXLMRoberta

from enso.registry import Registry, ModeKeys
from enso.experiment.finetuning import FinetuneSequenceLabel


@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class DocumentLabeler(FinetuneSequenceLabel):
    def __init__(self, *args, **kwargs):
        self.model_config = dict(val_size=0)
        self.model_config.update(kwargs)
        self.model = DocumentLabeler(**self.model_config)