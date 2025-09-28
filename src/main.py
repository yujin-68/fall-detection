#최종 알고리즘 실행

import sys
import os
import time
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import YoloDetector
from pose_estimator import PoseEstimator
from fall_logic import FallDetectorLogic

def main():
    # 파일 경로 설정
    video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'videos', '03.mp4')
    
    yolo_detector = YoloDetector()
    pose_estimator = PoseEstimator()
    fall_logic = FallDetectorLogic()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: 다음 경로의 비디오 파일 열리지 않음 {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        current_time = time.time() #현재 시간 기록 (낙상 로직의 속도 및 시간 추적에 사용)

        # 1. YOLO를 사용하여 프레임에서 사람을 감지하고 바운딩 박스를 얻음
        detections = yolo_detector.detect_person(frame)

        # 감지된 각 사람에 대해 반복 
        for person_data in detections:
            bbox = person_data['bbox']
            track_id = 1

            # 2. MediaPipe를 사용하여 뼈대 랜드마크를 추정
            pose_landmarks = pose_estimator.estimate_pose(frame)

            # 3. 낙상 로직을 실행하여 상태를 판단 
            person_status = fall_logic.process_detection(track_id, bbox, current_time)

            # 4. 시각화 및 결과 출력 
            x1, y1, x2, y2 = bbox

            color = (0, 255, 0) #초록색
            if 'Fall' in person_status or 'Lying' in person_status:
                color = (0, 0, 255) #빨간색

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, person_status, (x1, y1 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow('Fall Detection System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # q키 누르면 루프 종료
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()