#!/usr/bin/env python
"""
Hydra 설정 테스트 스크립트
각종 설정 조합이 충돌 없이 작동하는지 확인
"""

import subprocess
import sys

def test_config(command, description):
    """설정 테스트"""
    print(f"\n테스트: {description}")
    print(f"명령어: {command}")
    
    # --cfg job 옵션으로 설정만 확인 (실제 학습 X)
    test_command = command.replace("python", "python").replace("src/", "src/") + " --cfg job"
    
    try:
        result = subprocess.run(test_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 성공: 설정이 올바릅니다.")
            return True
        else:
            print("✗ 실패: Hydra 설정 오류")
            print(f"에러: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 실패: {e}")
        return False

def main():
    """주요 설정 조합 테스트"""
    
    tests = [
        # 기본 설정
        ("python src/train_improved.py --cfg job", 
         "기본 설정"),
        
        # Fixed 모델 (수정된 버전)
        ("python src/train_improved.py ++model.type=kgat_fixed model.aggregator=bi-interaction --cfg job",
         "Fixed 모델 설정"),
        
        # 멀티 GPU
        ("python src/train_improved.py training.devices=4 training.strategy=ddp --cfg job",
         "멀티 GPU 설정"),
        
        # 개선된 모델
        ("python src/train_improved.py use_improved_model=true --cfg job",
         "개선된 모델 설정"),
        
        # 복합 설정
        ("python src/train_improved.py ++model.type=kgat_fixed training.devices=4 data.batch_size=4096 --cfg job",
         "Fixed 모델 + 멀티 GPU"),
    ]
    
    passed = 0
    failed = 0
    
    print("="*60)
    print("Hydra 설정 테스트 시작")
    print("="*60)
    
    for command, description in tests:
        if test_config(command, description):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"테스트 완료: {passed}개 성공, {failed}개 실패")
    print("="*60)
    
    if failed > 0:
        print("\n실패한 설정이 있습니다. 위의 에러 메시지를 확인하세요.")
        sys.exit(1)
    else:
        print("\n모든 설정이 정상적으로 작동합니다!")

if __name__ == "__main__":
    main()