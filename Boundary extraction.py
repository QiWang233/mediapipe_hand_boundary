# -*- coding: utf-8 -*-
# @Time : 2022/8/14 15:30
# @File : Boundary extraction.py
# @Software: PyCharm
# @Author : @white233

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from app import calc_bounding_rect
import numpy as np

import copy


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                      (255, 255, 255), 3)

    return image


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

draw = mp.solutions.drawing_utils

H = 4
W = 8

for i in range(0, 32):
    img = cv2.imread('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\datasets\\C00S0001A20\\leapImg\\%03d.png' % i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_idx, _ in enumerate(results.multi_hand_landmarks):
            hand = results.multi_hand_landmarks[hand_idx]
            draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    # print(results.multi_hand_landmarks)
    brect = calc_bounding_rect(img, results.multi_hand_landmarks[0])
    brect[0] = brect[0] - 80
    brect[1] = brect[1] - 60
    brect[2] = brect[2] + 60
    brect[3] = 480

    # boundary detect
    if brect[0] < 0:
        brect[0] = 0
    elif brect[1] < 0:
        brect[1] = 0
    elif brect[2] > 640:
        brect[2] = 640
    elif brect[3] > 480:
        brect[3] = 480

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img)
    img = img[brect[1]:brect[3], brect[0]:brect[2]]
    img = cv2.resize(img, (640, 480))
    # draw_bounding_rect(True, debug_image, brect)

    print(img.shape)
    print(brect)
    plt.subplot(H, W, i + 1)
    # plt.imshow(image_crop)
    plt.imshow(img, 'gray')
plt.show()
