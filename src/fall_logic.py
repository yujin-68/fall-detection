#낙상 판단 로직

import time
import numpy as np

#낙상 판단에 필요한 모든 로직과 상태 추적을 담당하는 클래스
class FallDetectorLogic:
    def __init__(self):
        self.person_states = {} #사람별 상태 정보(이전 위치, 시간, 확률 등)를 저장할 딕셔너리
        self.VELOCITY_THRESHOLD = 150 #낙상으로 판단할 Y축 속도의 임계값 설정 (픽셀/초)
        self.ASPECT_RATIO_THRESHOLD = 1.0 #낙상으로 판단할 가로/세로 비율 임계값 설정 (x변화량/y변화량) 
        self.STILLNESS_TIME_THRESHOLD = 3 #낙상 후 움직임이 없어야 하는 최소 시간 설정 

    #매 프레임마다 YOLO 감지 결과(detection)를 처리하고 낙상 지표를 계산하는 메서드
    def process_detection(self, track_id, bbox, current_time):
        x1, y1, x2, y2 = bbox #바운딩 박스 좌표 추출

        current_center_y = (y1 + y2) / 2
        width = x2 - x1 
        height = y2 - y1

        if track_id not in self.person_states:
            self.person_states[track_id] = {
                'last_y': current_center_y,
                'last_time': current_time,
                'last_width': width,
                'last_height': height,
                'status': 'Standing',
                'fall_start_time': None
            }
            return self.person_states[track_id]['status'] #초기 상태일 때는 계산을 건너뛰고 기본 상태를 반환 
        
        state = self.person_states[track_id]

        dt = current_time - state['last_time']
        dy = current_center_y - state['last_y']

        velocity_y = dy / dt if dt > 0 else 0 

        aspect_ratio = width / height if height > 0 else 0

        # 1. Y축 속도를 이용한 급격한 하강 감지
        is_high_velocity_fall = velocity_y > self.VELOCITY_THRESHOLD

        # 2. 바운딩 박스 비율을 이용한 쓰러짐 감지 (x변화량/y변화량 > 1.0)
        is_horizontal = aspect_ratio > self.ASPECT_RATIO_THRESHOLD

        # 상태 갱신 로직 
        new_status = state['status']
        if is_high_velocity_fall:
            new_status = 'Potential Fall'
        elif is_horizontal:
            new_status = 'Lying'
        elif not is_high_velocity_fall and not is_horizontal:
            if current_center_y < (state['last_y'] - 50) and abs(velocity_y) < 5:
                new_status = 'Sitting'
            else:
                new_status = 'Standing'
        
        #상태 정보 업데이트
        state['last_y'] = current_center_y
        state['last_time'] = current_time
        state['last_width'] = width
        state['last_height'] = height
        state['status'] = new_status

        return new_status #최종 상태 반환