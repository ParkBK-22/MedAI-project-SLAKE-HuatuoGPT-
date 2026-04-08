"""
SLAKE 실험 결과 상세 분석 스크립트

진단 지표, 질문 타입별 성능, Modality별 성능 등을 시각화하고 분석합니다.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

def load_results(results_dir="results"):
    """결과 파일 로드"""
    results_csv = os.path.join(results_dir, "slake_results.csv")
    diagnostics_yaml = os.path.join(results_dir, "diagnostics.yaml")
    q_type_yaml = os.path.join(results_dir, "question_type_analysis.yaml")
    
    if not os.path.exists(results_csv):
        print(f"Error: Results file not found: {results_csv}")
        return None, None, None
    
    df = pd.read_csv(results_csv)
    
    diagnostics = None
    if os.path.exists(diagnostics_yaml):
        with open(diagnostics_yaml, 'r') as f:
            diagnostics = yaml.safe_load(f)
    
    q_type_analysis = None
    if os.path.exists(q_type_yaml):
        with open(q_type_yaml, 'r') as f:
            q_type_analysis = yaml.safe_load(f)
    
    return df, diagnostics, q_type_analysis

def print_summary(df, diagnostics, q_type_analysis):
    """전체 요약 출력"""
    print("\n" + "=" * 80)
    print("SLAKE MEDICAL VQA PERTURBATION ANALYSIS - SUMMARY")
    print("=" * 80)
    
    # 기본 통계
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df) // len(df['condition'].unique())}")
    print(f"  Total evaluations: {len(df)}")
    print(f"  Conditions: {list(df['condition'].unique())}")
    print(f"  Question types: {list(df['q_type'].unique())}")
    print(f"  Modalities: {list(df['modality'].unique())}")
    
    # 진단 지표 요약
    if diagnostics:
        print(f"\nDiagnostic Metrics:")
        print(f"  VRS (Vision Reliance Score):  {diagnostics.get('VRS', 0):.4f}")
        print(f"    Interpretation: {abs(diagnostics.get('VRS', 0)):.1%} performance drop without visual info")
        
        if diagnostics.get('L_Drop'):
            print(f"  L-Drop (Location Spatial Sensitivity): {diagnostics.get('L_Drop', 0):.4f}")
            print(f"    Interpretation: {abs(diagnostics.get('L_Drop', 0)):.1%} drop for location questions with scrambled patches")
        
        print(f"  K-Ratio (Knowledge Bias):     {diagnostics.get('K_Ratio', 0):.4f}")
        print(f"    Interpretation: Model answers {diagnostics.get('K_Ratio', 0):.1%} of questions without visual info")

def print_condition_analysis(df):
    """조건별 상세 분석"""
    print("\n" + "=" * 80)
    print("CONDITION-BY-CONDITION ANALYSIS")
    print("=" * 80)
    
    conditions = sorted(df['condition'].unique())
    
    print(f"\n{'Condition':<20} {'Accuracy':<12} {'Samples':<12} {'Correct':<10}")
    print("-" * 54)
    
    for cond in conditions:
        subset = df[df['condition'] == cond]
        acc = subset['correct'].mean()
        correct_count = subset['correct'].sum()
        total = len(subset)
        print(f"{cond:<20} {acc:>10.4f}   {total:>10}    {correct_count:>8}")

def print_question_type_analysis(q_type_analysis):
    """질문 타입별 분석"""
    if not q_type_analysis:
        print("\nQuestion type analysis not available.")
        return
    
    print("\n" + "=" * 80)
    print("QUESTION TYPE ANALYSIS")
    print("=" * 80)
    
    for q_type in sorted(q_type_analysis.keys()):
        stats = q_type_analysis[q_type]
        print(f"\n{q_type}:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Overall accuracy: {stats['avg_accuracy']:.4f}")
        print(f"  Accuracy by condition:")
        for cond in sorted(stats['accuracy_by_condition'].keys()):
            acc = stats['accuracy_by_condition'][cond]
            print(f"    {cond:<20}: {acc:.4f}")

def print_modality_analysis(df):
    """이미지 모달리티별 분석"""
    print("\n" + "=" * 80)
    print("MODALITY ANALYSIS (CT vs MRI vs X-Ray)")
    print("=" * 80)
    
    modalities = sorted(df['modality'].unique())
    conditions = sorted(df['condition'].unique())
    
    print(f"\n{'Modality':<15} {' '.join(f'{c:<12}' for c in conditions)}")
    print("-" * (15 + len(conditions) * 12))
    
    for modality in modalities:
        subset = df[df['modality'] == modality]
        row = f"{modality:<15}"
        for cond in conditions:
            cond_subset = subset[subset['condition'] == cond]
            acc = cond_subset['correct'].mean() if len(cond_subset) > 0 else 0.0
            row += f" {acc:<11.4f}"
        print(row)

def print_vision_grounding_insights(df, q_type_analysis):
    """시각 정보 의존도 분석 및 인사이트"""
    print("\n" + "=" * 80)
    print("VISUAL GROUNDING INSIGHTS")
    print("=" * 80)
    
    # Original vs Black 비교
    orig_acc = df[df['condition'] == 'original']['correct'].mean()
    black_acc = df[df['condition'] == 'black']['correct'].mean()
    
    print(f"\n1. Visual Information Dependency:")
    print(f"   Original image accuracy:  {orig_acc:.4f}")
    print(f"   Black image accuracy:     {black_acc:.4f}")
    print(f"   Performance gap:          {abs(orig_acc - black_acc):.4f} ({abs(orig_acc - black_acc)*100:.1f}%)")
    
    if orig_acc - black_acc > 0.1:
        print(f"   → Strong visual information dependency detected!")
    elif orig_acc - black_acc > 0.05:
        print(f"   → Moderate visual information dependency.")
    else:
        print(f"   → Weak visual information dependency - model relies mostly on knowledge.")
    
    # LPF vs HPF 비교
    lpf_acc = df[df['condition'] == 'lpf']['correct'].mean()
    hpf_acc = df[df['condition'] == 'hpf']['correct'].mean()
    
    print(f"\n2. Frequency Information Sensitivity:")
    print(f"   Low-pass filter (LPF) accuracy:  {lpf_acc:.4f}")
    print(f"   High-pass filter (HPF) accuracy: {hpf_acc:.4f}")
    print(f"   Difference (LPF - HPF):          {lpf_acc - hpf_acc:.4f}")
    
    if lpf_acc > hpf_acc + 0.05:
        print(f"   → Model prefers low-frequency information (structural context)")
    elif hpf_acc > lpf_acc + 0.05:
        print(f"   → Model prefers high-frequency information (texture details)")
    else:
        print(f"   → Model depends equally on both frequency components")
    
    # Patch Shuffle 영향
    shuffle_acc = df[df['condition'] == 'patch_shuffle']['correct'].mean()
    print(f"\n3. Spatial Structure Importance:")
    print(f"   Patch shuffle accuracy:       {shuffle_acc:.4f}")
    print(f"   Original accuracy:            {orig_acc:.4f}")
    print(f"   Performance drop:             {orig_acc - shuffle_acc:.4f} ({(orig_acc - shuffle_acc)*100:.1f}%)")
    
    if orig_acc - shuffle_acc > 0.1:
        print(f"   → Spatial structure is crucial for model's inference!")
    elif orig_acc - shuffle_acc > 0.05:
        print(f"   → Spatial structure has moderate importance.")
    else:
        print(f"   → Spatial structure has minimal impact - model likely recognizes local patterns")

def export_detailed_results(df, results_dir="results"):
    """상세 결과를 다양한 형식으로 내보내기"""
    print(f"\n[Exporting detailed results...]")
    
    # 1. 조건별 요약 CSV
    condition_summary = df.groupby('condition').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)
    condition_summary.columns = ['Correct', 'Total', 'Accuracy']
    condition_file = os.path.join(results_dir, "condition_summary.csv")
    condition_summary.to_csv(condition_file)
    print(f"  ✓ Condition summary: {condition_file}")
    
    # 2. 질문 타입 × 조건 교차표
    pivot_table = pd.crosstab(
        df['q_type'],
        df['condition'],
        values=df['correct'],
        aggfunc='mean'
    ).round(4)
    pivot_file = os.path.join(results_dir, "qtype_condition_pivot.csv")
    pivot_table.to_csv(pivot_file)
    print(f"  ✓ Question type × Condition pivot: {pivot_file}")
    
    # 3. 실패한 샘플 분석
    failures = df[df['correct'] == 0].copy()
    if len(failures) > 0:
        failure_file = os.path.join(results_dir, "failed_samples.csv")
        failures.to_csv(failure_file, index=False)
        print(f"  ✓ Failed samples: {failure_file} ({len(failures)} failures)")
    
    # 4. 모달리티별 요약
    modality_summary = df.groupby('modality').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)
    modality_summary.columns = ['Correct', 'Total', 'Accuracy']
    modality_file = os.path.join(results_dir, "modality_summary.csv")
    modality_summary.to_csv(modality_file)
    print(f"  ✓ Modality summary: {modality_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze SLAKE perturbation experiment results")
    parser.add_argument("--results_dir", default="results", help="Results directory path")
    parser.add_argument("--export", action="store_true", help="Export detailed analysis files")
    args = parser.parse_args()
    
    # 결과 로드
    df, diagnostics, q_type_analysis = load_results(args.results_dir)
    
    if df is None:
        return
    
    # 분석 출력
    print_summary(df, diagnostics, q_type_analysis)
    print_condition_analysis(df)
    print_question_type_analysis(q_type_analysis)
    print_modality_analysis(df)
    print_vision_grounding_insights(df, q_type_analysis)
    
    # 상세 결과 내보내기
    if args.export:
        export_detailed_results(df, args.results_dir)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
