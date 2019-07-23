from copy import deepcopy
from enso.config import MODE, FIX_REQUIREMENTS
from enso.mode import ModeKeys
import logging

logger = logging.getLogger(__name__)


class WarnOnce:
    def __init__(self):
        self.msgs = set()

    def filter(self, log_entry):
        log_hash = hash(log_entry.msg)
        rv = log_hash not in self.msgs
        self.msgs.add(log_hash)
        if not rv:
            del log_entry
        return rv


logger.addFilter(WarnOnce())


class ValidateExperiments:
    """
    Validates and optionally repairs experiment configurations with respect to a mode and a set of requirements.
    """

    def __init__(self, fix=True):
        self.fix = fix
        self._cache = set()

    @property
    def reg(self):
        return Registry

    def is_repeated(self, setup_dict):
        cache_key = hash(tuple(sorted(str(a) for a in setup_dict.values())))
        if cache_key in self._cache:
            logger.warning("Repairing created duplicate experiments, skipping...")
            return True
        else:
            self._cache.add(cache_key)
            return False

    @staticmethod
    def valid_dataset(dataset_str):
        mode = dataset_str.split("/")[0]
        if MODE.value != mode:
            logger.warning(
                "Dataset {} is not compatible with mode {}. Skipping.".format(
                    value, MODE.value
                )
            )
            return False
        return True

    @staticmethod
    def skip_from_negative_requirement(requirement, field_val, fix):
        requires_not = requirement.split(" ")[1]
        if field_val == requires_not:
            if fix:
                logger.warning(
                    "Unsatisfied requirements of the form not <Class> cannot be "
                    " repaired skipping instead"
                )
                return True

            else:
                raise ValueError(
                    "Config items set to use {} when requirements enforces {}".format(
                        field_val, requirement
                    )
                )
        return False

    @staticmethod
    def enforce_uniqueness(setup_dicts, experiments, skips):
        # group by similar hparams. Effectively putting experiment into the innermost loop.
        setups = []
        for setup_idx, (setup, experiment) in enumerate(zip(setup_dicts, experiments)):
            if setup_idx in skips:
                continue

            set = False
            for i, (setup_group, _) in enumerate(setups):
                if setup == setup_group:
                    setups[i][1].append(experiment)
                    set = True
                    break
            if not set:
                setups.append([setup, [experiment]])
        return setups

    def validate(self, setup_dict, experiments):
        setup_dicts = [deepcopy(setup_dict) for _ in range(len(experiments))]
        skips = []
        for exp_idx, experiment in enumerate(experiments):
            setup_dict = setup_dicts[exp_idx]
            setup_dict["Experiment"] = experiment.__name__

            for key, value in setup_dict.items():
                if key == "Dataset":
                    if not self.valid_dataset(value):
                        skips.append(exp_idx)

                for field, requirement in self.reg._requirements.get(value, []):
                    field_val = setup_dict.get(field)
                    if field_val is None:
                        raise Exception(
                            "Requirements for {} contains invalid field {}".format(
                                value, field
                            )
                        )

                    if requirement.startswith("not"):
                        if self.skip_from_negative_requirement(
                            requirement, field_val, self.fix
                        ):
                            skips.append(exp_idx)

                    else:
                        if field_val != requirement:
                            if self.fix:
                                logger.warning(
                                    "Requirements being fixed, replacing {} with {}".format(
                                        setup_dict[field], requirement
                                    )
                                )
                                setup_dict[field] = requirement
                            else:
                                raise ValueError(
                                    "Config items set to use {} when requirements enforces {}".format(
                                        setup_dict[field], requirement
                                    )
                                )

            if self.is_repeated(setup_dict):
                skips.append(setup_dict)

            del setup_dict["Experiment"]

        return self.enforce_uniqueness(setup_dicts, experiments, skips)


class Registry:
    _experiment = {}
    _featurizer = {}
    _sampler = {}
    _resampler = {}
    _requirements = {}
    _metrics = {}
    _visualizer = {}
    _modes = {}

    @classmethod
    def _decorator(
        cls, p_cls, layer, mode=None, requirements=None, registration_name=None
    ):
        """Registers & returns p_cls with registration_name or default name."""
        p_name = registration_name or p_cls.__name__
        requirements = requirements or []

        if mode is not None:
            cls._modes[p_name] = mode

        if requirements is not None:
            cls._requirements[p_name] = requirements

        if p_name in layer:
            raise LookupError("%s already registered." % p_name)
        layer[p_name] = p_cls
        p_cls.name = lambda _: p_name
        return p_cls

    @classmethod
    def register_experiment(cls, mode, requirements=None):
        return lambda p_cls: cls._decorator(
            p_cls, cls._experiment, mode=mode, requirements=requirements
        )

    @classmethod
    def register_featurizer(cls, mode, requirements=None):
        return lambda p_cls: cls._decorator(
            p_cls, cls._featurizer, mode, requirements=requirements
        )

    @classmethod
    def register_sampler(cls, mode, requirements=None):
        return lambda p_cls: cls._decorator(
            p_cls, cls._sampler, mode, requirements=requirements
        )

    @classmethod
    def register_resampler(cls, mode, requirements=None):
        return lambda p_cls: cls._decorator(
            p_cls, cls._resampler, mode, requirements=requirements
        )

    @classmethod
    def register_metric(cls, mode, requirements=None):
        return lambda p_cls: cls._decorator(
            p_cls, cls._metrics, mode, requirements=requirements
        )

    @classmethod
    def register_visualizer(cls):
        return lambda p_cls: cls._decorator(p_cls, cls._visualizer)

    @classmethod
    def _get_plugin(cls, name, type):
        try:
            return_val = type[name]
        except LookupError:
            raise LookupError("{} is not in the registry".format(name))
        required_mode = cls._modes.get(return_val.__name__, ModeKeys.ANY)
        if required_mode != MODE and required_mode != ModeKeys.ANY:
            raise ValueError(
                "{} requires mode: {} current mode is: {}".format(
                    name, required_mode, MODE
                )
            )
        return return_val

    @classmethod
    def get_experiment(cls, name):
        return cls._get_plugin(name, cls._experiment)

    @classmethod
    def get_featurizer(cls, name):
        return cls._get_plugin(name, cls._featurizer)

    @classmethod
    def get_sampler(cls, name):
        return cls._get_plugin(name, cls._sampler)

    @classmethod
    def get_resampler(cls, name):
        return cls._get_plugin(name, cls._resampler)

    @classmethod
    def get_metric(cls, name):
        return cls._get_plugin(name, cls._metrics)

    @classmethod
    def get_visualizer(cls, name):
        return cls._get_plugin(name, cls._visualizer)
