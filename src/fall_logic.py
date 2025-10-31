#낙상 판단 로직

import time
import numpy as np
import math

#낙상 판단에 필요한 모든 로직과 상태 추적을 담당하는 클래스
class FallDetectorLogic:
    def __init__(self):
        self.person_states = {} #사람별 상태 정보(이전 위치, 시간, 확률 등)를 저장할 딕셔너리
        self.VELOCITY_THRESHOLD = 150 #낙상으로 판단할 Y축 속도의 임계값 설정 (픽셀/초)
        self.ASPECT_RATIO_THRESHOLD = 1.0 #낙상으로 판단할 가로/세로 비율 임계값 설정 (x변화량/y변화량) 
        self.STILLNESS_TIME_THRESHOLD = 3 #낙상 후 움직임이 없어야 하는 최소 시간 설정 
        self.STILLNESS_Y_THRESHOLD = 5 #움직임 없음으로 판단할 픽셀 변화 임계값

        # 베이즈 추론(로그-오즈 누적) 관련 파라미터
        # 사전확률: 낙상은 드뭄 (약 1%)
        self.DEFAULT_LOG_ODDS = math.log(0.01 / (1 - 0.01))
        # 프레임 간 연속성 편향(아주 약하게 상승시키거나 1.0으로 비활성화 가능)
        self.TRANSITION_BIAS = math.log(1.02)
        # 의사결정 임계값 (히스테리시스 적용)
        self.FALL_THRESHOLD = 0.9
        self.RECOVER_THRESHOLD = 0.7

        # 특징 분포(간단한 가우시안/베르누이 근사) - 데이터에 맞게 튜닝 필요
        self.VEL_FALL_MU = 250
        self.VEL_FALL_SIG = 80
        self.VEL_NOT_MU = 30
        self.VEL_NOT_SIG = 40

        self.AR_FALL_MU = 1.6
        self.AR_FALL_SIG = 0.4
        self.AR_NOT_MU = 0.6
        self.AR_NOT_SIG = 0.3

        # 정지 여부에 대한 라플라스 스무딩된 베르누이 확률
        self.STILL_P_FALL = (0.9, 0.1) # (still, not-still)에서 p(still|Fall), p(not|Fall)
        self.STILL_P_NOT  = (0.2, 0.8) # (still, not-still)에서 p(still|¬Fall), p(not|¬Fall)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def _gaussian_log_likelihood(self, x, mu, sigma):
        if sigma <= 0:
            return -1e9
        z = (x - mu) / sigma
        return -0.5 * (z * z)

    def _feature_llr_velocity(self, velocity_y):
        ll_fall = self._gaussian_log_likelihood(velocity_y, self.VEL_FALL_MU, self.VEL_FALL_SIG)
        ll_not  = self._gaussian_log_likelihood(velocity_y, self.VEL_NOT_MU, self.VEL_NOT_SIG)
        return ll_fall - ll_not

    def _feature_llr_aspect(self, aspect_ratio):
        ll_fall = self._gaussian_log_likelihood(aspect_ratio, self.AR_FALL_MU, self.AR_FALL_SIG)
        ll_not  = self._gaussian_log_likelihood(aspect_ratio, self.AR_NOT_MU, self.AR_NOT_SIG)
        return ll_fall - ll_not

    def _feature_llr_still(self, is_still):
        if is_still:
            p_fall, p_not = self.STILL_P_FALL[0], self.STILL_P_NOT[0]
        else:
            p_fall, p_not = self.STILL_P_FALL[1], self.STILL_P_NOT[1]
        return math.log(p_fall + 1e-6) - math.log(p_not + 1e-6)

    def _update_posterior(self, state, velocity_y, aspect_ratio, is_still):
        log_odds = state.get('log_odds_fall', self.DEFAULT_LOG_ODDS)
        log_odds += self._feature_llr_velocity(velocity_y)
        log_odds += self._feature_llr_aspect(aspect_ratio)
        log_odds += self._feature_llr_still(is_still)
        log_odds += self.TRANSITION_BIAS
        # 수치 안정화 클리핑
        log_odds = max(min(log_odds, 20.0), -20.0)
        state['log_odds_fall'] = log_odds
        return self._sigmoid(log_odds)

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
                'fall_start_time': None,
                'log_odds_fall': self.DEFAULT_LOG_ODDS
            }
            return self.person_states[track_id]['status'] #초기 상태일 때는 계산을 건너뛰고 기본 상태를 반환 
        
        state = self.person_states[track_id]

        dt = current_time - state['last_time']
        dy = current_center_y - state['last_y']

        velocity_y = dy / dt if dt > 0 else 0 

        aspect_ratio = width / height if height > 0 else 0

        # 규칙 기반 보조 특징 (베이즈 우도에 사용)
        is_high_velocity_fall = velocity_y > self.VELOCITY_THRESHOLD
        is_horizontal = aspect_ratio > self.ASPECT_RATIO_THRESHOLD
        is_currently_still = abs(dy) < self.STILLNESS_Y_THRESHOLD

        # -----------------------------------------------------------------
        # 상태 갱신 로직 (정지 시간 추적 추가)
        # -----------------------------------------------------------------

        current_status = state['status']
        fall_start_time = state['fall_start_time']

        # 베이즈 posterior 업데이트
        p_fall = self._update_posterior(state, velocity_y, aspect_ratio, is_currently_still)

        # 히스테리시스 기반 상태 결정
        if current_status != 'Fall Detected!':
            if p_fall >= self.FALL_THRESHOLD:
                current_status = 'Fall Detected!'
                fall_start_time = None
            else:
                # 보조 상태 레이블링 (설명/시각화를 위해 유지)
                if is_horizontal:
                    current_status = 'Lying'
                elif current_center_y < (state['last_y'] - 50) and abs(velocity_y) < 5:
                    current_status = 'Sitting'
                elif is_high_velocity_fall:
                    current_status = 'Potential Fall'
                else:
                    current_status = 'Standing'
                # Potential Fall 시점 기록은 선택적
                if current_status == 'Potential Fall' and fall_start_time is None:
                    fall_start_time = current_time
                elif current_status != 'Potential Fall':
                    fall_start_time = None
        else:
            # 이미 Fall 상태인 경우 복구 임계값 아래로 충분히 내려가면 해제
            if p_fall < self.RECOVER_THRESHOLD:
                # 해제 후 현재 자세에 따라 라벨
                if is_horizontal:
                    current_status = 'Lying'
                else:
                    current_status = 'Standing'
                fall_start_time = None
        
        #상태 정보 업데이트
        state['last_y'] = current_center_y
        state['last_time'] = current_time
        state['status'] = current_status
        state['fall_start_time'] = fall_start_time

        return current_status #최종 상태 반환