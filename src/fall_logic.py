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
        self.STILLNESS_Y_THRESHOLD = 5 #움직임 없음으로 판단할 픽셀 변화 임계값

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

        # 현재 정지 상태인지 확인 
        is_currently_still = abs(dy) < self.STILLNESS_Y_THRESHOLD

        # -----------------------------------------------------------------
        # 상태 갱신 로직 (정지 시간 추적 추가)
        # -----------------------------------------------------------------

        current_status = state['status']
        fall_start_time = state['fall_start_time']

        # 1. 초기 낙상 감지: 속도가 빠르거나, 현재 Potential Fall 상태가 아니라면
        if is_high_velocity_fall and current_status not in ['Potential Fall', 'Fall Detected!']:
            current_status = 'Potential Fall'
            fall_start_time = current_time  # 타이머 시작
        
        # 2. Potential Fall 상태 처리
        if current_status == 'Potential Fall':
            
            # 누워있는 상태(is_horizontal)가 지속되어야 최종 판단 진행
            if is_horizontal:
                # 정지 상태가 충분한 시간(3초) 이상 지속되었는지 확인
                if is_currently_still and (current_time - fall_start_time) >= self.STILLNESS_TIME_THRESHOLD:
                    current_status = 'Fall Detected!' # 최종 낙상 사고 확정
                    fall_start_time = None # 타이머 초기화 (더 이상 필요 없음)
                # 정지 상태가 아니라면 타이머는 계속 흐르거나, 리셋될 수 있음 (단순히 이탈 방지)
                # 여기서는 타이머가 흐르도록 유지
            else:
                # 속도는 빨랐지만 다시 서거나 앉은 자세로 돌아간 경우 (오경보)
                current_status = 'Standing'
                fall_start_time = None # 타이머 리셋

        # 3. 일반 상태 (낙상이 아닌 경우)
        elif current_status not in ['Potential Fall', 'Fall Detected!']:
            if is_horizontal:
                current_status = 'Lying'  # 단순히 누워있는 자세
            elif current_center_y < (state['last_y'] - 50) and abs(velocity_y) < 5:
                current_status = 'Sitting' # 앉은 자세
            else:
                current_status = 'Standing'
            fall_start_time = None # 타이머 리셋

        # 4. Fall Detected! 상태에서는 계속 유지
        elif current_status == 'Fall Detected!':
             # 사고가 확정되었으므로 상태를 유지 (알림 후 수동 리셋 필요)
            pass 
        
        #상태 정보 업데이트
        state['last_y'] = current_center_y
        state['last_time'] = current_time
        state['status'] = current_status
        state['fall_start_time'] = fall_start_time

        return current_status #최종 상태 반환