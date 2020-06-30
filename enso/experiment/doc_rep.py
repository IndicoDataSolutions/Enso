from finetune import SequenceLabeler
from finetune.base_models import RoBERTa

from enso.registry import Registry, ModeKeys
from enso.experiment.finetuning import FinetuneSequenceLabel


@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "TextContextFeaturizer")])
class SidekickSeqLab(FinetuneSequenceLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config.update(
            dict(
                # general params that differ from finetune
                base_model=RoBERTa,
                batch_size=4,
                predict_batch_size=10,
                val_size=0.0,
                crf_sequence_labeling=False,
                low_memory_mode=True,
                class_weights="log",
                # auxiliary-specific params
                use_auxiliary_info=True,
                context_dim=4,
                default_context={
                    'left': 0,
                    'right': 0,
                    'top': 0,
                    'bottom': 0,
                },
                n_context_embed_per_channel=48,
                context_in_base_model=True,
                n_layers_with_aux=-1)
        )
        self.model_config.update(kwargs)
        self.model = SequenceLabeler(**self.model_config)

    def fit(self, X, y):
        text, context = zip(*X)
        self.model.fit(text, y, context=context)

    def predict(self, X, **kwargs):
        text, context = zip(*X)
        return self.model.predict(text, context=context)

@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "TextContextFeaturizer")])
class RoBERTaSeqLab(SidekickSeqLab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = dict(
            base_model=RoBERTa,
            low_memory_mode=True,
        )
        self.model_config.update(kwargs)
        self.model = SequenceLabeler(**self.model_config)

    def fit(self, X, y):
        text, context = zip(*X)
        self.model.fit(text, y)

    def predict(self, X, **kwargs):
        text, context = zip(*X)
        return self.model.predict(text)

@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "TextContextFeaturizer")])
class LambertSeqLab(SidekickSeqLab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config.update(dict(
            pos_injection=True,
            n_layers_with_aux = 0,
            context_in_base_model = False
        ))
        self.model_config.update(kwargs)
        self.model = SequenceLabeler(**self.model_config)
            


            
        
        
        
        
        
        
        