import numpy as np
from bound import BoundingBox

class Note(object):
    NOTES = {
        -7: ("c6", 84),
        -6: ("b5", 83),
        -5: ("a5", 81),
        -4: ("g5", 79),
        -3: ("f5", 77),
        -2: ("e5", 76),
        -1: ("d5", 74),
        0: ("c5", 72),
        1: ("b4", 71),
        2: ("a4", 69),
        3: ("g4", 67),
        4: ("f4", 65),
        5: ("e4", 64),
        6: ("d4", 62),
        7: ("c4", 60),
        8: ("b3", 59),
        9: ("a3", 57),
        10: ("g3", 55),
        11: ("f3", 53),
        12: ("e3", 52),
        13: ("d3", 50),
        14: ("c3", 48),
        15: ("b2", 47),
        16: ("a2", 45),
        17: ("g2", 43),
        18: ("f2", 41),
        19: ("e2", 40),
        20: ("d2", 38),
        21: ("c2", 36),
        22: ("b1", 35)
    }

    def __init__(self, label, pitch, duration, box):
        self.label = label
        self.pitch = pitch
        self.duration = duration
        self.box = box

    def __repr__(self):
        return "(Note: {}, pitch: {}, duration: {})".format(self.label, self.pitch, self.duration)
