# 업비트 AI 자동 매매 & 종목 추천 시스템

강화학습(PPO) 기반 업비트 가상화폐 자동 매매 및 AI 종목 추천 시스템입니다.

## 주요 기능

### 🌐 실시간 웹앱
- **현대적인 UI/UX**: Tailwind CSS 기반 다크모드 디자인
- **실시간 모니터링**: WebSocket으로 3초마다 가격 업데이트
- **대시보드**: TOP 10 추천 종목, AI 예측, 전체 종목 목록
- **상세 분석**: 클릭 한 번으로 종목 분석 + AI 예측
- **반응형 디자인**: 모바일/태블릿/데스크톱 완벽 대응

### 🎯 종목 추천 시스템
- **기술적 분석 추천**: 20개 이상의 기술적 지표로 전체 시장 스캔
- **AI 예측 추천**: 강화학습 모델 기반 상승 예상 종목 추천
- **특정 종목 검색**: 원하는 코인 검색 및 상세 분석
- **API 키 불필요**: 공개 API만으로 모든 분석 가능

### 🤖 자동 매매 시스템
- **AI 모델 학습**: PPO(Proximal Policy Optimization) 알고리즘 사용
- **백테스팅**: 과거 데이터로 전략 성능 검증
- **실시간 자동 매매**: 학습된 모델을 사용한 실시간 트레이딩 (API 키 필요)
- **기술적 지표**: 20개 이상의 기술적 지표 활용 (SMA, EMA, MACD, RSI, 볼린저밴드, 스토캐스틱 등)

## 프로젝트 구조

```
upbit_study/
├── src/
│   ├── api/
│   │   └── upbit_client.py          # 업비트 API 클라이언트
│   ├── analyzer/
│   │   ├── market_analyzer.py       # 기술적 분석
│   │   └── ai_predictor.py          # AI 예측
│   ├── models/
│   │   └── rl_agent.py              # 강화학습 에이전트 (PPO)
│   ├── environment/
│   │   └── trading_env.py           # 트레이딩 환경 (Gymnasium)
│   ├── backtesting/
│   │   └── backtest.py              # 백테스팅 시스템
│   └── bot/
│       └── trading_bot.py           # 실시간 트레이딩 봇
├── static/js/app.js                 # 웹앱 JavaScript
├── templates/index.html             # 웹앱 HTML
├── main.py                          # 자동매매 메인 프로그램
├── recommend.py                     # 종목 추천 메인
├── webapp.py                        # 웹앱 서버
├── requirements.txt                 # 의존성 패키지
└── environment.yml                  # Conda 환경 파일
```

## 설치 방법

### 방법 A: 자동 설치 (권장)

#### Windows
```bash
cd upbit_study
setup_conda.bat
```

#### macOS/Linux
```bash
cd upbit_study
bash setup_conda.sh
```

### 방법 B: 수동 설치

```bash
# Conda 가상환경 생성
conda create -n upbit python=3.10 -y
conda activate upbit

# 패키지 설치
cd upbit_study
pip install -r requirements.txt
```

## 빠른 시작 (API 키 불필요!)

### 🌐 실시간 웹앱 (추천!)

```bash
cd upbit_study

# 웹앱 패키지 설치 (최초 1회)
pip install -r webapp_requirements.txt

# 웹앱 실행
python webapp.py

# 브라우저에서 접속
# http://localhost:8000
```

### 🚀 종목 추천 (CLI)

```bash
cd upbit_study

# 주요 코인 TOP 5 빠른 분석
python quick_recommend.py

# 전체 종목 추천 프로그램
python recommend.py
```

## 업비트 API 키 설정 (자동 매매용)

1. [업비트 사이트](https://upbit.com) 로그인
2. 마이페이지 → Open API 관리
3. Open API Key 발급
4. `.env.example` 파일을 `.env`로 복사하고 API 키를 입력

```
UPBIT_ACCESS_KEY=발급받은_Access_Key
UPBIT_SECRET_KEY=발급받은_Secret_Key
```

## 기술 스택

- **언어**: Python 3.8+
- **AI 프레임워크**: Stable-Baselines3, PyTorch, Gymnasium
- **데이터 처리**: Pandas, NumPy
- **기술적 분석**: TA-Lib
- **웹 프레임워크**: FastAPI
- **시각화**: Matplotlib, Seaborn

## 주의사항

### ⚠️ 투자 위험 경고

- 본 시스템은 **교육 및 연구 목적**으로 제작되었습니다
- 가상화폐 투자는 **고위험 투자**이며 원금 손실 가능성이 있습니다
- AI 모델의 성능은 **과거 성과가 미래를 보장하지 않습니다**
- 투자 손실에 대한 책임은 **사용자 본인**에게 있습니다

### 보안 주의사항

- API Secret Key는 **절대 공유하지 마세요**
- `.env` 파일을 Git에 **커밋하지 마세요**

---

**Happy Trading! 📈**
