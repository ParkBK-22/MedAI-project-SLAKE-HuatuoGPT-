import pandas as pd
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    res_path = "results/final_results.csv"
    if not os.path.exists(res_path):
        print("결과 파일이 없습니다.")
        return

    df = pd.read_csv(res_path)

    # 1. 전체 Accuracy 계산
    print("=== Overall Accuracy by Condition ===")
    overall_acc = df.groupby('condition')['correct'].mean()
    print(overall_acc)

    # 2. 질문 타입(q_type)별 분석
    print("\n=== Accuracy by Question Type (Vision Reliance) ===")
    type_acc = df.pivot_table(index='q_type', columns='condition', values='correct', aggfunc='mean')
    print(type_acc)

    # 3. VRS (Vision Reliance Score) 산출
    # Formula: Acc(Original) - Acc(Black)
    vrs = overall_acc['original'] - overall_acc['black']
    print(f"\nTotal VRS: {vrs:.4f}")

if __name__ == "__main__":
    main()