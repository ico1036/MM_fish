# MM Fish — RWA Perpetual Market Maker Backtester

신규 RWA S&P500 토큰(HyperLiquid XYZ/S&P500 Perpetual) 상장 시 마켓메이킹 알고리즘을 사전 테스트하는 시뮬레이터.

과거 데이터가 없는 신규 토큰이기 때문에, **LLM 에이전트(Claude Haiku)가 다양한 시장 참여자를 시뮬레이션**하여 MM 알고리즘이 버틸 수 있는 환경을 만듭니다.

## 핵심 아이디어

```
50명의 AI 트레이더 (각자 다른 성격/전략)
    ↓ Claude Haiku가 시장 상황 보고 판단
    ↓ 배치 옥션으로 주문 매칭
    ↓ 펀딩레이트 + 청산 엔진
    ↓ MM 알고리즘 성과 측정
```

- **A-S 베이스라인**: 전통적 Avellaneda-Stoikov 마켓메이킹 (규칙 기반 상대방)
- **LLM 시뮬레이션**: 50명의 이종 AI 에이전트가 만드는 시장 (더 현실적)
- **비교 기준**: Stylized facts (fat tails, volatility clustering, Hurst exponent 등)

## 빠른 시작

### 1. 설치

```bash
git clone <repo-url> && cd MM_fish
uv sync
```

### 2. Claude Code 로그인 확인

LLM 호출은 `claude -p` (Claude Code OAuth)를 사용합니다. 별도 API 키 불필요.

```bash
# Claude Code가 설치되어 있고 로그인되어 있는지 확인
claude --version
```

### 3. 테스트 실행

```bash
uv run --no-env-file --extra dev python -m pytest backend/tests/ -q
# 252 passed
```

### 4. RWA PoC 실행

```bash
uv run --no-env-file python scripts/hands_on.py
```

두 가지 시나리오를 실행합니다:
1. **정상 상장** — S&P500 지수가 평온한 날, 토큰 상장 첫날
2. **S&P500 급락** — 상장 중 S&P500 -3% 급락 발생

결과 플롯이 `results/` 디렉토리에 생성됩니다:

```bash
open results/1_price_scenarios.html   # 가격 비교: 지수 vs 퍼프
open results/2_mm_performance.html    # MM 인벤토리 & PnL
open results/3_spreads.html           # 스프레드 시계열
open results/4_funding_rates.html     # 펀딩레이트
```

### 5. Streamlit 대시보드 (기존 BTC 비교용)

```bash
# 먼저 비교 데이터 생성 (BTC 실데이터 필요)
uv run --no-env-file python scripts/run_comparison.py --symbol BTCUSDT --ticks 1000

# 대시보드 실행
uv run --no-env-file streamlit run dashboard/app.py
```

## 아키텍처

```
backend/
├── app/
│   ├── models/
│   │   ├── market.py           # Order, Trade, LOBSnapshot
│   │   ├── perpetual.py        # Position, FundingRate, LiquidationEvent
│   │   ├── simulation.py       # TickRecord, SimStatus
│   │   └── agent_profile.py    # TraderProfile
│   ├── services/
│   │   ├── mm_agent.py         # HelixMMAgent (A-S + 펀딩레이트 인식)
│   │   ├── llm_agents.py       # LLMTrader (Claude Haiku 호출, 멀티틱 플랜)
│   │   ├── llm_client.py       # claude -p subprocess 래퍼
│   │   ├── lob_engine.py       # Limit Order Book 엔진
│   │   ├── batch_auction.py    # 배치 옥션 매칭 (pro-rata)
│   │   ├── plan_executor.py    # 멀티틱 플랜 저장/실행
│   │   ├── funding_engine.py   # 펀딩레이트 계산/적용
│   │   ├── liquidation_engine.py # 마진 체크, 강제 청산
│   │   ├── scenario_engine.py  # 시나리오 (정상/급락/펀딩스파이크)
│   │   ├── rwa_personas.py     # RWA S&P500 에이전트 페르소나
│   │   ├── comparison_engine.py # A-S vs LLM 비교 오케스트레이션
│   │   ├── stylized_facts.py   # 통계적 특성 분석
│   │   └── metrics.py          # Sharpe, drawdown, 인벤토리 통계
│   └── utils/
│       └── math_utils.py       # A-S 수학 (reservation price, optimal spread)
├── tests/                      # 252개 테스트
scripts/
├── hands_on.py                 # RWA PoC 실행 스크립트
├── run_comparison.py           # A-S vs LLM 비교 CLI
└── run_simulation.py           # 단독 시뮬레이션 실행
dashboard/
└── app.py                      # Streamlit 대시보드
```

## 에이전트 타입 (RWA S&P500)

| 타입 | 비율 | 재평가 주기 | 설명 |
|------|------|------------|------|
| TradFi 헤지 | 15% | 50틱 | 온체인 S&P 익스포저, 보수적 진입 |
| 차익거래자 | 15% | 10틱 | SPY ETF vs XYZ 퍼프 가격 차이 수렴 |
| 초기 투기꾼 | 20% | 10틱 | 고레버리지, 상장 변동성 베팅 |
| 펀딩 파머 | 10% | 100틱 | 펀딩레이트 방향 따라 진입 |
| 패닉 셀러 | 10% | 5틱 | S&P 하락 시 즉시 시장가 매도 |
| 크립토 네이티브 | 15% | 15틱 | 차트 패턴 트레이딩 |
| 기관 | 5% | 50틱 | TWAP 실행, 시장 충격 최소화 |
| HFT | 10% | 3틱 | 스프레드 제공/취소, 유동성 공급/흡수 |

## 시뮬레이션 루프

```
틱 T:
  1. 펀딩레이트 체크 (100틱마다)
  2. 청산 엔진: 마진 부족 포지션 강제 청산
  3. MM이 bid/ask 호가 제출
  4. LLM 에이전트: 재평가 필요한 에이전트만 Haiku 호출 → 멀티틱 플랜 수신
  5. 플랜에서 현재 틱 주문 수집 + 규칙기반 에이전트 주문
  6. 배치 옥션: 모든 주문을 한 번에 매칭 (pro-rata)
  7. 잔여 주문 → LOB에 대기
  8. 기록: mid price, spread, MM PnL, 펀딩레이트, 청산 수
```

## 호출 효율

멀티틱 플랜 덕분에 에이전트가 매 틱마다 LLM을 호출하지 않습니다:

```
기존: 50 에이전트 × 500틱 × 15% = 3,750 호출 (직렬, ~30분)
현재: 50 에이전트 × (500 ÷ 평균20틱) = ~1,250 호출 (~$0.30)
```

## 시나리오

| 시나리오 | 설명 | 용도 |
|---------|------|------|
| `normal_listing` | 평온한 S&P, 약간의 상승 드리프트 | 정상 운영 검증 |
| `sp500_crash` | 특정 틱에서 -3% 급락 + 고변동성 | 스트레스 테스트 |
| `funding_spike` | 롱 쏠림으로 펀딩 급등 | 펀딩 비용 리스크 |

## 개발

```bash
# 테스트
uv run --no-env-file --extra dev python -m pytest backend/tests/ -v

# 린트
uv run --no-env-file --extra dev ruff check backend/

# 특정 테스트
uv run --no-env-file --extra dev python -m pytest backend/tests/test_services/test_batch_auction.py -v
```
