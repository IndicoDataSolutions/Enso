from enum import Enum


class ModeKeys(Enum):
    SEQUENCE = "SequenceLabeling"
    CLASSIFY = "Classify"
    ANY = "Any"
