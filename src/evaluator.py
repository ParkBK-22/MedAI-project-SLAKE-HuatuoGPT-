import re
import warnings

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

class SlakeEvaluator:
    @staticmethod
    def clean_text(text):
        """소문자 변환, 특수문자 제거, 공백 정리"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    @staticmethod
    def is_yes_no_question(question):
        """Yes/No 질문 판정"""
        q_lower = question.lower()
        yes_no_patterns = [
            'is ', 'are ', 'do ', 'does ', 'did ', 'can ', 'could ', 'would ',
            'has ', 'have ', 'was ', 'were ', '?', 
            'is there', 'are there', 'is it', 'does it', 'is this', 'is that'
        ]
        return any(pattern in q_lower for pattern in yes_no_patterns)

    def evaluate_yes_no(self, prediction, ground_truth):
        """
        Yes/No 형식 평가
        'yes/true/present' vs 'no/false/absent'
        """
        pred = self.clean_text(prediction)
        gt = self.clean_text(ground_truth)
        
        yes_keywords = ['yes', 'true', 'present', 'positive', 'correct', 'right', 'exist']
        no_keywords = ['no', 'false', 'absent', 'negative', 'incorrect', 'wrong', 'not']
        
        pred_is_yes = any(kw in pred for kw in yes_keywords)
        pred_is_no = any(kw in pred for kw in no_keywords)
        gt_is_yes = any(kw in gt for kw in yes_keywords)
        gt_is_no = any(kw in gt for kw in no_keywords)
        
        # 예측과 정답의 yes/no 방향 일치 여부
        if (pred_is_yes and gt_is_yes) or (pred_is_no and gt_is_no):
            return 1
        return 0

    def evaluate(self, prediction, ground_truth, question=""):
        """
        통합 평가 함수
        - Yes/No 질문: Yes/No 형식 평가
        - 기타: Exact Match 평가
        
        Args:
            prediction: 모델 출력
            ground_truth: 정답
            question: 원본 질문 (질문 타입 판정용)
            
        Returns:
            정확도 (0 or 1)
        """
        # Yes/No 질문 판정 및 평가
        if self.is_yes_no_question(question):
            return self.evaluate_yes_no(prediction, ground_truth)
        
        # 기본: Exact Match (정답 단어가 모델 답변에 포함)
        pred = self.clean_text(prediction)
        gt = self.clean_text(ground_truth)
        
        if gt in pred:
            return 1
        return 0