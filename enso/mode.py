from enum import Enum


class ModeKeys(Enum):
    SEQUENCE = "SequenceLabeling"
    CLASSIFY = "Classify"
    RATIONALIZED = "RationalizedClassify"
    ANY = "Any"
