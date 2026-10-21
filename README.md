# 학사정보 RAG 방어 실험 코드

한국외국어대학교 학사정보 RAG에서 한국어·영어 간접 프롬프트 인젝션에 대한 D0–D4 방어를 비교한 연구 코드입니다.

## 코드 구조

- `source/Thesis/src/thesis_rag/`: 문서 파싱·청킹, 임베딩·검색, 생성, 방어·탐지, 실험 실행과 통계 분석 패키지입니다.
- `source/Thesis/configs/`: 개발·본실험의 모델 revision과 실행 설정입니다.
- `source/Thesis/work/`: 실험 준비·실행·검증 스크립트입니다. 본실험 진입점은 `run_main_experiment.py`입니다.
- `source/Thesis/tests/`: 합성 단위 테스트와 작은 JSONL fixtures입니다.
- 루트 `sensitivity.py`, `requirements.txt`: 14차 개정의 사후 민감도 분석 코드와 고정 의존성입니다. `source/Thesis/` 상대 경로를 유지했습니다.

## 사용 모델과 라이브러리

| 역할 | 모델·구현 |
| --- | --- |
| 답변 생성 | `qwen3:8b-q8_0`, Ollama `/api/chat` |
| 검색 임베딩 | `qwen3-embedding:0.6b`, Ollama `/api/embed`, 1024차원 |
| 공격 탐지 | `meta-llama/Llama-Prompt-Guard-2-86M`, PyTorch·Transformers, CPU |

검색은 NumPy exact cosine Top-K, 통계 분석은 NumPy·statsmodels를 사용합니다. SciPy는 균형 표본 선택과 사후 분석에 사용합니다. 모델 revision·생성 옵션·탐지 임계값은 YAML 설정에 있습니다.

## 설치와 실행

Python 3.11–3.13을 사용합니다.

```sh
cd source/Thesis
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,models,analysis]'

# 모델 호출 없는 합성 테스트
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests

# 합성 fixture 실행
.venv/bin/thesis fixture run --config configs/development.yaml --output work/fixture-run.jsonl
.venv/bin/thesis fixture verify --results work/fixture-run.jsonl

# 승인된 입력·프로토콜·매니페스트 및 로컬 모델이 준비된 경우에만 실행
.venv/bin/python work/run_main_experiment.py
```

실제 추론에는 Ollama와 해당 모델을 별도로 준비해야 합니다. Prompt Guard는 제공자의 접근 승인을 받은 모델이 필요합니다.

사후 분석은 루트에서 별도 환경을 사용합니다. 다음 분석 명령은 원래의 `inputs/` 자료·해시 목록·참조 파일이 준비된 경우에만 실행할 수 있습니다.

```sh
cd ../..
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements.txt
.venv-analysis/bin/python sensitivity.py --root . --output analysis-sensitivity
```
