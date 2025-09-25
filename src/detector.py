#YOLO 모델과 관련된 모든 기능 관리

from ultralytics import YOLO
import cv2

class YoloDetector:
    #초기화 메서드
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
    
    #주어진 frame에서 사람 감지 메서드
    def detect_person(self, frame):
        results = self.model(frame, stream=True)

        #감지된 사람의 정보 담는 빈 리스트
        detections = []
        person_class_id = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0]) #cls < 감지된 객체의 class ID
                conf = float(box.conf[0]) #conf < 감지된 객체의 신뢰도

                if cls == person_class_id and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) #좌상단(x1, y1), 우하단(x2, y2)좌표를 정수형으로 변환

                    person_info = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    }
                    detections.append(person_info)
        #감지된 모든 사람 정보 반환
        return detections
