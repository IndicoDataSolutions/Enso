from finetune import SequenceLabeler
from finetune.base_models import RoBERTa

from enso.registry import Registry, ModeKeys
from enso.experiment.finetuning import FinetuneSequenceLabel


@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
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

@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class RoBERTaSeqLab(SidekickSeqLab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = dict(
            use_auxiliary_info = False,
            n_layers_with_aux = 0,
            context_in_base_model = False,
            context_dim = 0
        )
        self.model_config.update(kwargs)
        self.model = SequenceLabeler(**self.model_config)

@Registry.register_experiment(ModeKeys.SEQUENCE, requirements=[("Featurizer", "PlainTextFeaturizer")])
class LambertSeqLab(SidekickSeqLab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config.update(dict(
            pos_injection=True,
            n_layers_with_aux = 0,
            context_in_base_model = False
        ))
        self.model_config.update(kwargs)
        self.model = SequenceLabeler()
            


            
        
        
        
        
        
        
        