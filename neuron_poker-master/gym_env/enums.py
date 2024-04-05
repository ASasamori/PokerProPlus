"""Enums"""
from enum import Enum


class Action(Enum):
    """Allowed actions"""

    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_MIN = 3
    RAISE_2BB = 4
    RAISE_3BB = 5
    RAISE_HALF_POT = 6
    RAISE_POT = 7
    RAISE_1_5POT = 8
    RAISE_2POT = 9
    ALL_IN = 10
    SMALL_BLIND = 11
    BIG_BLIND = 12
    RAISE_2X = 13



class Stage(Enum):
    """Allowed actions"""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5
