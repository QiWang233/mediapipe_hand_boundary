# -*- coding: utf-8 -*-
# @Time : 2022/8/15 19:18
# @File : MediaPipe Holistic.py
# @Software: PyCharm
# @Author : @white233


import cv2
import mediapipe as mp

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
            static_image_mode=True,
    ) as holistic:

        image = cv2.imread(r'D:\Install_exe\1.jpg')
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            )

        # 在图片上画身体、左右手、面部关节点
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imwrite('result.png', annotated_image)
