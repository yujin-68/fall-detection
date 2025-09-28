#MediaPipe와 관련된 모든 기능 관리

import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils #랜드마크 그릴 떄 사용
        self.pose = self.mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) #랜드마크 감지 최소 신뢰도를 50%
    
    def estimate_pose(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_rgb.flags.writeable = False #이미지를 수정 불가능 상태로 설정
        results = self.pose.process(img_rgb) #Pose 모델 실행하여 랜드마크 추정
        
        img_rgb.flags.writeable = True #이미지를 다시 쓰기 상태로 되돌림 

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
    
        return results.pose_landmarks #감지된 랜드마크 정보 반환. 없으면 None 반환