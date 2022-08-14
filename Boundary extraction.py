# -*- coding: utf-8 -*-
# @Time : 2022/8/14 15:30
# @File : Boundary extraction.py
# @Software: PyCharm
# @Author : @white233

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from app import calc_bounding_rect
import copy

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

draw = mp.solutions.drawing_utils

img = cv2.imread('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\datasets\\C00S0001A20\\leapImg\\000.png', -1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
debug_image = copy.deepcopy(img)
results = hands.process(img)

if results.multi_hand_landmarks:
    for hand_idx, _ in enumerate(results.multi_hand_landmarks):
        hand = results.multi_hand_landmarks[hand_idx]
        draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)


print(results.multi_hand_landmarks)
brect = calc_bounding_rect(debug_image, results.multi_hand_landmarks[0])
print(brect)
plt.imshow(img)
plt.show()
