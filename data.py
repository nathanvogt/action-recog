import os
import json
import numpy as np


KP_TO_NAME = {
    0: "center_hip",
    1: "right_hip",
    2: "right_knee",
    3: "right_ankle",
    4: "left_hip",
    5: "left_knee",
    6: "left_ankle",
    7: "center_spine",
    8: "neck",
    9: "tip_of_neck",
    10: "head",
    11: "right_shoulder",
    12: "right_elbow",
    13: "right_wrist",
    14: "left_shoulder",
    15: "left_elbow",
    16: "left_wrist",
    17: "right_middle_foot",
    18: "right_tip_foot",
    19: "left_middle_foot",
    20: "left_tip_foot",
    21: "right_hand_1",
    22: "right_hand_2",
    23: "left_hand_1",
    24: "left_hand_2",
}

LEFT_LEG_NO_FEET = [4, 5, 6]
RIGHT_LEG_NO_FEET = [1, 2, 3]
LEFT_ARM_NO_HAND = [14, 15, 16]
RIGHT_ARM_NO_HAND = [11, 12, 13]
LEFT_HAND = [22, 24]
RIGHT_HAND = [21, 23]
LEFT_FOOT = [19, 20]
RIGHT_FOOT = [17, 18]
BACK = [0, 7, 8]
