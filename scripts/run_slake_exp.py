import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import warnings

# 프로젝트 루트 경로 추가 (부모 디렉토리)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

from src.perturbations import ImagePerturber
from src.dataset_slake import SlakeDataset
from src.evaluator import SlakeEvaluator
from src.model_huatuo import HuatuoInference

def compute_diagnostics(results_df):
    """
    진단 지표 계산: VRS, L-Drop, K-Ratio
    
    Args:
        results_df: 실험 결과 DataFrame
        
    Returns:
        dict: 진단 지표 결과
    """
    diagnostics = {}
    
    # 조건별 정확도 계산
    acc_by_condition = {}
    for cond in results_df['condition'].unique():
        acc = results_df[results_df['condition'] == cond]['correct'].mean()
        acc_by_condition[cond] = acc
    
    # 1. VRS (Vision Reliance Score): EM(Original) - EM(Black)
    if 'original' in acc_by_condition and 'black' in acc_by_condition:
        vrs = acc_by_condition['original'] - acc_by_condition['black']
        diagnostics['VRS'] = vrs
    
    # 2. L-Drop (Location Drop): Acc_location(Original) - Acc_location(Shuffle)
    location_results = results_df[results_df['q_type'] == 'Location']
    if len(location_results) > 0:
        loc_acc_orig = location_results[location_results['condition'] == 'original']['correct'].mean()
        loc_acc_shuffle = location_results[location_results['condition'] == 'patch_shuffle']['correct'].mean()
        l_drop = loc_acc_orig - loc_acc_shuffle
        diagnostics['L_Drop'] = l_drop
    
    # 3. K-Ratio (Knowledge Ratio): EM(Black) / EM(Original)
    if acc_by_condition.get('original', 0) > 0:
        k_ratio = acc_by_condition.get('black', 0) / acc_by_condition['original']
        diagnostics['K_Ratio'] = k_ratio
    
    # 조건별 정확도도 저장
    diagnostics['accuracy_by_condition'] = acc_by_condition
    
    return diagnostics

def analyze_by_question_type(results_df):
    """질문 타입별 성능 분석"""
    analysis = {}
    
    for q_type in results_df['q_type'].unique():
        subset = results_df[results_df['q_type'] == q_type]
        acc_by_cond = {}
        for cond in subset['condition'].unique():
            acc = subset[subset['condition'] == cond]['correct'].mean()
            acc_by_cond[cond] = acc
        analysis[q_type] = {
            'total_samples': len(subset),
            'accuracy_by_condition': acc_by_cond,
            'avg_accuracy': subset['correct'].mean()
        }
    
    return analysis

def run_experiment(config_path):
    """
    SLAKE 이미지 Perturbation 실험 실행
    
    Args:
        config_path: 설정 파일 경로 (YAML)
    """
    # 1. 설정 로드
    print("=" * 60)
    print("SLAKE Medical VQA Perturbation Experiment")
    print("=" * 60)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"✓ Config loaded: {config_path}")
    
    # 2. 부품 초기화
    print("\nInitializing components...")
    perturber = ImagePerturber(
        lpf_sigma=cfg['perturbation']['lpf_sigma'],
        hpf_sigma=cfg['perturbation']['hpf_sigma'],
        patch_size=cfg['perturbation']['patch_size']
    )
    print(f"  ✓ ImagePerturber initialized")
    
    dataset = SlakeDataset(cfg['data']['json_path'], cfg['data']['img_dir'])
    print(f"  ✓ SlakeDataset loaded ({len(dataset)} samples)")
    
    evaluator = SlakeEvaluator()
    print(f"  ✓ SlakeEvaluator initialized")
    
    # 3. 모델 로드
    print("\nLoading model...")
    model = HuatuoInference(cfg, device=cfg['model']['device'], use_official_cli=False)
    print(f"  ✓ HuatuoGPT-Vision initialized")
    
    results = []
    
    # 4. 메인 실험 루프
    print(f"\n{'=' * 60}")
    print(f"Starting experiment on {len(dataset)} samples...")
    print(f"Conditions: {cfg['perturbation']['conditions']}")
    print(f"{'=' * 60}\n")
    
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[i]
        
        for cond in cfg['perturbation']['conditions']:
            # 이미지 변형 적용
            p_img = perturber.apply(sample['image'], cond)
            
            # 모델 추론
            prediction = model.generate_answer(p_img, sample['question'])
            
            # 채점 (질문 타입에 따라 평가 방식 결정)
            is_correct = evaluator.evaluate(
                prediction,
                sample['answer'],
                sample['question']
            )
            
            # 결과 저장
            results.append({
                "img_id": sample['img_id'],
                "condition": cond,
                "q_type": sample['q_type'],
                "modality": sample['modality'],
                "question": sample['question'],
                "gt": sample['answer'],
                "pred": prediction,
                "correct": is_correct
            })
    
    # 5. 결과 저장 (기본 결과)
    print("\n" + "=" * 60)
    print("Experiment Complete! Processing results...")
    print("=" * 60 + "\n")
    
    os.makedirs(cfg['data']['output_dir'], exist_ok=True)
    df = pd.DataFrame(results)
    
    results_csv = os.path.join(cfg['data']['output_dir'], "slake_results.csv")
    df.to_csv(results_csv, index=False)
    print(f"✓ Results saved: {results_csv}")
    
    # 6. 진단 지표 계산 및 저장
    print("\nComputing diagnostic metrics...")
    diagnostics = compute_diagnostics(df)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC METRICS")
    print("=" * 60)
    print(f"VRS (Vision Reliance Score):      {diagnostics.get('VRS', 0):.4f}")
    print(f"  → {abs(diagnostics.get('VRS', 0)):.1%} difference between Original and Black")
    print(f"L-Drop (Location Drop):           {diagnostics.get('L_Drop', 0):.4f}")
    print(f"  → {abs(diagnostics.get('L_Drop', 0)):.1%} drop for Location questions (Original → Shuffle)")
    print(f"K-Ratio (Knowledge Ratio):        {diagnostics.get('K_Ratio', 0):.4f}")
    print(f"  → Model answers {diagnostics.get('K_Ratio', 0):.1%} of questions without visual info")
    
    print("\nAccuracy by Condition:")
    for cond, acc in diagnostics['accuracy_by_condition'].items():
        print(f"  {cond:15s}: {acc:.4f} ({int(acc * len(df) / len(cfg['perturbation']['conditions']))}/{len(df) // len(cfg['perturbation']['conditions'])})")
    
    # 저장
    diagnostics_file = os.path.join(cfg['data']['output_dir'], "diagnostics.yaml")
    with open(diagnostics_file, 'w') as f:
        yaml.dump(diagnostics, f, default_flow_style=False)
    print(f"\n✓ Diagnostics saved: {diagnostics_file}")
    
    # 7. 질문 타입별 분석
    print("\nAnalyzing by question type...")
    q_type_analysis = analyze_by_question_type(df)
    
    print("\n" + "=" * 60)
    print("QUESTION TYPE ANALYSIS")
    print("=" * 60)
    for q_type, stats in q_type_analysis.items():
        print(f"\n{q_type} ({stats['total_samples']} samples):")
        print(f"  Overall accuracy: {stats['avg_accuracy']:.4f}")
        for cond, acc in stats['accuracy_by_condition'].items():
            print(f"    {cond:15s}: {acc:.4f}")
    
    # 저장
    q_type_file = os.path.join(cfg['data']['output_dir'], "question_type_analysis.yaml")
    with open(q_type_file, 'w') as f:
        yaml.dump(q_type_analysis, f, default_flow_style=False)
    print(f"\n✓ Question type analysis saved: {q_type_file}")
    
    print("\n" + "=" * 60)
    print("✓ All results completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment("configs/slake_config.yaml")