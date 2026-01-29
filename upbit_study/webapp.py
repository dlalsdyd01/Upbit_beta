"""
업비트 실시간 종목 분석 웹앱
FastAPI + 현대적인 UI/UX + 자동매매
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import sys
from typing import List, Optional
import json
import math
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# .env 파일 로드
load_dotenv()

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.upbit_client import UpbitClient
from src.analyzer.market_analyzer import MarketAnalyzer
from src.analyzer.ai_predictor import AIPredictor
from src.news.signal_generator import NewsSignalGenerator
from src.models.rl_agent import TradingAgent
from src.environment.trading_env import CryptoTradingEnv

# FastAPI 앱 생성
app = FastAPI(
    title="업비트 AI 트레이딩 분석",
    description="실시간 종목 분석 및 AI 예측",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    print("[STARTUP] 서버 시작 중...")

    # 저장된 자동매매 상태 복원
    if load_auto_trading_state():
        print("[STARTUP] 자동매매 상태가 복원되었습니다.")
        print("[STARTUP] 자동매매를 재시작하려면 웹 UI에서 '자동매매 시작'을 클릭하세요.")
        print("[STARTUP] (이전 포지션 정보가 유지됩니다)")
    else:
        print("[STARTUP] 새로운 세션을 시작합니다.")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 글로벌 객체
client = UpbitClient("", "")
analyzer = MarketAnalyzer(client)
predictor = AIPredictor()
news_signal_generator = NewsSignalGenerator()  # 뉴스 신호 생성기

# 자동매매 관련 글로벌 객체
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')

# API 키가 placeholder가 아닌 실제 값인지 확인
def is_valid_api_key(key: str) -> bool:
    return key and key not in ['', 'your_access_key_here', 'your_secret_key_here']

trading_client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY) if is_valid_api_key(UPBIT_ACCESS_KEY) and is_valid_api_key(UPBIT_SECRET_KEY) else None

# 자동매매 상태 (단일 코인)
trading_bot_task: Optional[asyncio.Task] = None
trading_status = {
    "is_running": False,
    "market": "KRW-BTC",
    "interval": 60,
    "max_trade_amount": 100000,
    "start_time": None,
    "trade_count": 0,
    "current_position": None,
    "start_balance": 0,
    "current_balance": 0,
    "profit": 0,
    "profit_rate": 0,
    "last_action": None,
    "last_action_time": None,
    "last_price": 0,
    "trade_history": []
}

# ========== 원클릭 자동매매 (다중 코인) ==========
auto_trading_task: Optional[asyncio.Task] = None
auto_trading_status = {
    "is_running": False,
    "mode": "auto",
    "total_investment": 50000,
    "coin_count": 3,
    "analysis_mode": "volume_top50",
    "coin_category": "normal",  # 'safe', 'normal', 'meme', 'all'
    "trading_interval": 60,
    "allocation_mode": "weighted",  # 'equal' (균등배분) or 'weighted' (점수기반)
    "target_profit_percent": 10.0,  # 목표가 (+%)
    "stop_loss_percent": 10.0,      # 손절가 (-%)
    "start_time": None,
    "start_balance": 0,
    "current_balance": 0,
    "profit": 0,
    "profit_rate": 0,
    "positions": {},
    "selected_coins": [],
    "trade_history": []
}

# 상태 저장 파일 경로
AUTO_TRADING_STATE_FILE = "auto_trading_state.json"

def save_auto_trading_state():
    """자동매매 상태를 파일에 저장"""
    try:
        state_to_save = {
            'is_running': auto_trading_status['is_running'],
            'total_investment': auto_trading_status['total_investment'],
            'coin_count': auto_trading_status['coin_count'],
            'analysis_mode': auto_trading_status['analysis_mode'],
            'coin_category': auto_trading_status['coin_category'],
            'trading_interval': auto_trading_status['trading_interval'],
            'allocation_mode': auto_trading_status['allocation_mode'],
            'target_profit_percent': auto_trading_status['target_profit_percent'],
            'stop_loss_percent': auto_trading_status['stop_loss_percent'],
            'start_time': auto_trading_status['start_time'],
            'start_balance': auto_trading_status['start_balance'],
            'current_balance': auto_trading_status['current_balance'],
            'profit': auto_trading_status['profit'],
            'profit_rate': auto_trading_status['profit_rate'],
            'positions': auto_trading_status['positions'],
            'selected_coins': auto_trading_status['selected_coins'],
            'trade_history': auto_trading_status['trade_history'][-50:],  # 최근 50개만
            'saved_at': datetime.now().isoformat()
        }

        with open(AUTO_TRADING_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=2)

        print(f"[STATE] 자동매매 상태 저장 완료: {AUTO_TRADING_STATE_FILE}")
    except Exception as e:
        print(f"[STATE] 상태 저장 실패: {e}")

def load_auto_trading_state():
    """파일에서 자동매매 상태 복원"""
    global auto_trading_status

    try:
        if not os.path.exists(AUTO_TRADING_STATE_FILE):
            print("[STATE] 저장된 상태 파일이 없습니다.")
            return False

        with open(AUTO_TRADING_STATE_FILE, 'r', encoding='utf-8') as f:
            saved_state = json.load(f)

        # 실행 중이었던 경우만 복원
        if saved_state.get('is_running'):
            auto_trading_status.update({
                'total_investment': saved_state['total_investment'],
                'coin_count': saved_state['coin_count'],
                'analysis_mode': saved_state['analysis_mode'],
                'coin_category': saved_state['coin_category'],
                'trading_interval': saved_state['trading_interval'],
                'allocation_mode': saved_state['allocation_mode'],
                'target_profit_percent': saved_state['target_profit_percent'],
                'stop_loss_percent': saved_state['stop_loss_percent'],
                'start_time': saved_state['start_time'],
                'start_balance': saved_state['start_balance'],
                'current_balance': saved_state['current_balance'],
                'profit': saved_state['profit'],
                'profit_rate': saved_state['profit_rate'],
                'positions': saved_state['positions'],
                'selected_coins': saved_state['selected_coins'],
                'trade_history': saved_state['trade_history']
            })

            print(f"[STATE] ✅ 자동매매 상태 복원 완료")
            print(f"[STATE] - 시작 시간: {saved_state['start_time']}")
            print(f"[STATE] - 포지션: {len(saved_state['positions'])}개")
            print(f"[STATE] - 수익률: {saved_state['profit_rate']:+.2f}%")
            return True
        else:
            print("[STATE] 이전에 실행 중이 아니었으므로 복원하지 않습니다.")
            return False

    except Exception as e:
        print(f"[STATE] 상태 복원 실패: {e}")
        return False

# AI 트레이딩 에이전트
trading_agent: Optional[TradingAgent] = None

# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


def clean_dict(d):
    """NaN/Infinity 값을 None으로 변환"""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(item) for item in d]
    elif isinstance(d, float):
        if math.isnan(d) or math.isinf(d):
            return None
        return d
    return d


def normalize_score_to_100(score):
    """점수를 100점 만점으로 정규화

    Args:
        score: 원래 점수 (대략 -10 ~ +12 범위)

    Returns:
        0~100 사이의 정규화된 점수
    """
    # -10 ~ +12 범위를 0 ~ 100으로 매핑
    # -10 = 0점, 0 = 45점, +12 = 100점
    normalized = ((score + 10) / 22.0) * 100
    return max(0, min(100, round(normalized, 1)))


def calculate_trade_prices(tech_data):
    """매수/매도/손절 가격 계산

    Args:
        tech_data: 기술적 분석 데이터

    Returns:
        매수가, 매도가, 손절가 딕셔너리
    """
    current_price = tech_data.get('current_price', 0)
    rsi = tech_data.get('rsi', 50)
    bb_low = tech_data.get('bb_low', current_price * 0.95)
    bb_high = tech_data.get('bb_high', current_price * 1.05)
    recommendation = tech_data.get('recommendation', '중립')

    # 매수가 계산 - 현재가 기준으로 실제 매수 가능한 가격
    if rsi < 30:  # 과매도 - 적극 매수
        buy_price = current_price * 1.00  # 현재가에 바로 매수
        target_profit = 0.12  # 12% 목표
    elif rsi < 40:  # 저평가
        buy_price = current_price * 0.99  # 현재가의 99%
        target_profit = 0.10  # 10% 목표
    elif recommendation in ['강력 매수', '매수']:
        buy_price = current_price * 0.98  # 현재가의 98%
        target_profit = 0.08  # 8% 목표
    else:
        buy_price = current_price * 0.97  # 현재가의 97% (관망 시 조금 더 낮게)
        target_profit = 0.05  # 5% 목표

    # 매도가 계산 (목표가) - 매수가 기준으로 계산
    if recommendation in ['강력 매수', '매수', '약한 매수']:
        sell_price = buy_price * (1 + target_profit)  # 목표 수익률 적용
    else:
        sell_price = buy_price * 1.05  # 매수가 대비 5% 이익

    # 매도가는 반드시 매수가보다 높아야 함 (최소 2% 이익 보장)
    min_sell_price = buy_price * 1.02
    if sell_price < min_sell_price:
        sell_price = min_sell_price

    # 손절가 계산 - 매수가 기준으로 계산
    if rsi < 30:
        stop_loss = buy_price * 0.95  # 5% 손절
    elif recommendation in ['강력 매수', '매수']:
        stop_loss = buy_price * 0.96  # 4% 손절
    else:
        stop_loss = buy_price * 0.97  # 3% 손절

    # 손절가는 반드시 매수가보다 낮아야 함 (최대 2% 손실 이하로 설정 방지)
    max_stop_loss = buy_price * 0.98
    if stop_loss > max_stop_loss:
        stop_loss = max_stop_loss

    return {
        'buy_price': round(buy_price, 2 if buy_price < 1000 else 0),
        'sell_price': round(sell_price, 2 if sell_price < 1000 else 0),
        'stop_loss': round(stop_loss, 2 if stop_loss < 1000 else 0),
        'expected_profit_rate': round(((sell_price - buy_price) / buy_price) * 100, 2),
        'risk_rate': round(((buy_price - stop_loss) / buy_price) * 100, 2)
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return FileResponse("templates/index.html")


@app.get("/api/markets")
async def get_markets():
    """전체 KRW 마켓 조회"""
    try:
        markets = analyzer.get_all_krw_markets()

        # 모든 마켓을 100개씩 분할해서 조회 (업비트 API 제한)
        result = []
        batch_size = 100

        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            market_codes = [m['market'] for m in batch]
            tickers = client.get_ticker(market_codes)

            for market, ticker in zip(batch, tickers):
                result.append({
                    'market': market['market'],
                    'korean_name': market['korean_name'],
                    'english_name': market['english_name'],
                    'current_price': ticker['trade_price'],
                    'change_rate': ticker['signed_change_rate'] * 100,
                    'trade_volume': ticker['acc_trade_price_24h']  # 거래대금 (KRW)
                })

        return {"success": True, "data": clean_dict(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/analysis/{market}")
async def get_analysis(market: str):
    """특정 종목 상세 분석"""
    try:
        # 기술적 분석
        tech_result = analyzer.analyze_market(market, days=30)
        if not tech_result:
            return {"success": False, "error": "분석 데이터 없음"}

        # 100점 만점 점수 추가
        tech_result['score_100'] = normalize_score_to_100(tech_result['score'])

        # 매수/매도/손절 가격 계산
        trade_prices = calculate_trade_prices(tech_result)
        tech_result['trade_prices'] = trade_prices

        # AI 예측
        df = analyzer.get_market_data(market, days=30)
        ai_result = None
        if df is not None:
            ai_result = predictor.predict_market(df, market)

        # 뉴스 신호 (코인별)
        news_signal = None
        try:
            news_signal = news_signal_generator.generate_coin_signal(market=market, page_size=100)
        except Exception as e:
            print(f"[WARNING] 뉴스 신호 생성 실패: {e}")

        # 실시간 티커 정보
        ticker = client.get_ticker([market])[0]

        # 추가 시장 정보
        market_info = {
            "opening_price": ticker.get('opening_price'),
            "high_price": ticker.get('high_price'),
            "low_price": ticker.get('low_price'),
            "prev_closing_price": ticker.get('prev_closing_price'),
            "acc_trade_price": ticker.get('acc_trade_price'),
            "acc_trade_price_24h": ticker.get('acc_trade_price_24h'),
            "acc_trade_volume_24h": ticker.get('acc_trade_volume_24h'),
            "highest_52_week_price": ticker.get('highest_52_week_price'),
            "highest_52_week_date": ticker.get('highest_52_week_date'),
            "lowest_52_week_price": ticker.get('lowest_52_week_price'),
            "lowest_52_week_date": ticker.get('lowest_52_week_date'),
            "timestamp": ticker.get('timestamp')
        }

        result_data = {
            "market": market,
            "timestamp": datetime.now().isoformat(),
            "technical": tech_result,
            "ai_prediction": ai_result,
            "news_signal": news_signal,
            "market_info": market_info
        }

        return {
            "success": True,
            "data": clean_dict(result_data)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/top-recommendations")
async def get_top_recommendations(limit: int = 10):
    """상위 추천 종목"""
    try:
        markets = analyzer.get_all_krw_markets()[:50]
        results = []

        for market_info in markets:
            market = market_info['market']
            result = analyzer.analyze_market(market, days=30)

            if result and result['score'] > 0:
                df = analyzer.get_market_data(market, days=30)
                if df is not None:
                    ai_result = predictor.predict_market(df, market)
                    result['ai_action'] = ai_result['action']
                    result['ai_confidence'] = ai_result['confidence']
                else:
                    result['ai_action'] = 0
                    result['ai_confidence'] = 0

                # 100점 만점 점수 추가
                result['score_100'] = normalize_score_to_100(result['score'])

                # 매수/매도/손절 가격 계산
                trade_prices = calculate_trade_prices(result)
                result['trade_prices'] = trade_prices

                # NaN/Infinity 제거
                result = clean_dict(result)
                results.append(result)

        results.sort(key=lambda x: x['score'], reverse=True)
        return {"success": True, "data": clean_dict(results[:limit])}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/top-recommendations-stream")
async def get_top_recommendations_stream(limit: int = 10, mode: str = "all"):
    """상위 추천 종목 (진행률 스트리밍)

    Args:
        limit: 결과 개수
        mode: 분석 모드
            - all: 전체 235개 분석 (기본값)
            - volume_top50: 거래량 상위 50개 분석
            - volume_top100: 거래량 상위 100개 분석
    """
    async def generate():
        try:
            all_markets = analyzer.get_all_krw_markets()

            # 모드에 따라 분석할 종목 선택
            if mode == "volume_top50":
                # 거래량 상위 50개
                tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
                sorted_markets = sorted(
                    zip(all_markets[:100], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:50]
                markets = [m[0] for m in sorted_markets]
            elif mode == "volume_top100":
                # 거래량 상위 100개
                tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
                sorted_markets = sorted(
                    zip(all_markets[:150], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:100]
                markets = [m[0] for m in sorted_markets]
            else:
                # 전체 분석
                markets = all_markets

            total = len(markets)
            results = []

            for i, market_info in enumerate(markets, 1):
                market = market_info['market']

                # 진행률 전송
                progress = {
                    "type": "progress",
                    "current": i,
                    "total": total,
                    "market": market
                }
                yield f"data: {json.dumps(progress)}\n\n"

                # 버퍼 플러시를 위한 작은 대기
                await asyncio.sleep(0.01)

                result = analyzer.analyze_market(market, days=30)

                if result and result['score'] > 0:
                    df = analyzer.get_market_data(market, days=30)
                    if df is not None:
                        ai_result = predictor.predict_market(df, market)
                        result['ai_action'] = ai_result['action']
                        result['ai_confidence'] = ai_result['confidence']
                    else:
                        result['ai_action'] = 0
                        result['ai_confidence'] = 0

                    # 100점 만점 점수 추가
                    result['score_100'] = normalize_score_to_100(result['score'])

                    # 매수/매도/손절 가격 계산
                    trade_prices = calculate_trade_prices(result)
                    result['trade_prices'] = trade_prices

                    result = clean_dict(result)
                    results.append(result)

            # 정렬 및 최종 결과 전송
            results.sort(key=lambda x: x['score'], reverse=True)
            final_data = {
                "type": "complete",
                "data": results[:limit]
            }
            yield f"data: {json.dumps(clean_dict(final_data))}\n\n"

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/api/news-signal")
async def get_news_signal(query: str = "cryptocurrency"):
    """뉴스 감정 분석 기반 트레이딩 신호 (시장 전체)

    Args:
        query: 검색 쿼리 (기본: cryptocurrency)

    Returns:
        신호 정보 (BUY, SELL, HOLD)
    """
    try:
        signal_data = news_signal_generator.generate_signal(query=query, page_size=100)
        return {"success": True, "data": clean_dict(signal_data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news-signal/{market}")
async def get_coin_news_signal(market: str):
    """특정 코인의 뉴스 감정 분석 신호

    Args:
        market: 마켓 코드 (예: KRW-BTC, KRW-ETH)

    Returns:
        해당 코인의 뉴스 신호
    """
    try:
        signal_data = news_signal_generator.generate_coin_signal(market=market, page_size=100)
        return {"success": True, "data": clean_dict(signal_data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news-signal/combined")
async def get_combined_news_signal():
    """종합 뉴스 신호 (비트코인 + 시장 분석)"""
    try:
        signal_data = news_signal_generator.get_combined_signal()
        return {"success": True, "data": clean_dict(signal_data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news-feed")
async def get_news_feed(page_size: int = 10):
    """최신 암호화폐 뉴스 피드

    Args:
        page_size: 가져올 뉴스 개수

    Returns:
        뉴스 리스트와 감정 분석 결과
    """
    try:
        from src.news.news_collector import NewsCollector
        from src.news.sentiment_analyzer import SentimentAnalyzer

        collector = NewsCollector()
        analyzer_news = SentimentAnalyzer()

        # 뉴스 수집
        articles = collector.get_crypto_news(page_size=page_size)

        # 감정 분석
        analysis_result = analyzer_news.analyze_news_batch(articles)

        return {
            "success": True,
            "data": {
                "articles": analysis_result['articles'],
                "summary": analysis_result['summary'],
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """실시간 데이터 스트리밍"""
    await manager.connect(websocket)

    # 기본 종목 리스트
    default_markets = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']
    user_markets = default_markets.copy()
    is_connected = True

    try:
        while is_connected:
            # 클라이언트에서 종목 설정 메시지 수신 (non-blocking)
            try:
                # 0.1초 타임아웃으로 메시지 확인
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                if message.get('type') == 'set_markets':
                    new_markets = message.get('markets', [])
                    if new_markets and isinstance(new_markets, list):
                        user_markets = new_markets[:10]  # 최대 10개로 제한
                        print(f"[WS] 종목 설정 변경: {user_markets}")
            except asyncio.TimeoutError:
                pass  # 메시지 없으면 계속 진행
            except WebSocketDisconnect:
                is_connected = False
                break
            except Exception:
                pass

            # 연결 상태 확인 후 데이터 전송
            if not is_connected:
                break

            try:
                if user_markets:
                    tickers = client.get_ticker(user_markets)
                    data = []

                    for ticker in tickers:
                        data.append({
                            'market': ticker['market'],
                            'price': ticker['trade_price'],
                            'change_rate': ticker['signed_change_rate'] * 100,
                            'volume': ticker['acc_trade_volume_24h'],
                            'timestamp': datetime.now().isoformat()
                        })

                    await websocket.send_json({
                        'type': 'price_update',
                        'data': data
                    })
            except WebSocketDisconnect:
                is_connected = False
                break
            except Exception as e:
                # API 오류 등 일시적 오류는 로그만 남기고 계속
                if "websocket" not in str(e).lower():
                    print(f"실시간 데이터 오류: {e}")
                else:
                    is_connected = False
                    break

            await asyncio.sleep(3)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            manager.disconnect(websocket)
        except:
            pass


# ========== 자동매매 API ==========

class TradingStartRequest(BaseModel):
    market: str = "KRW-BTC"
    interval: int = 60
    max_trade_amount: float = 100000


def get_market_data_for_trading(market: str, count: int = 200) -> pd.DataFrame:
    """트레이딩을 위한 시장 데이터 가져오기"""
    candles = trading_client.get_candles_minute(market, unit=1, count=count)
    df = pd.DataFrame(candles)
    df = df.rename(columns={
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    })
    df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def prepare_observation(candles: pd.DataFrame) -> np.ndarray:
    """관측값 준비"""
    env = CryptoTradingEnv(candles, initial_balance=1000000)
    env.reset()
    return env._get_observation()


async def trading_loop():
    """비동기 트레이딩 루프"""
    global trading_status, trading_agent, trading_client

    print(f"[TRADING] 자동매매 루프 시작 - {trading_status['market']}")

    # AI 모델 로드
    try:
        dummy_data = get_market_data_for_trading(trading_status['market'], 200)
        dummy_env = CryptoTradingEnv(dummy_data)
        trading_agent = TradingAgent(dummy_env)
        trading_agent.load('models/crypto_trader')
        print("[TRADING] AI 모델 로드 완료")
    except Exception as e:
        print(f"[TRADING] AI 모델 로드 실패: {e}")
        trading_agent = None

    # 초기 잔고 기록
    try:
        trading_status['start_balance'] = trading_client.get_balance('KRW')
        trading_status['current_balance'] = trading_status['start_balance']
    except Exception as e:
        print(f"[TRADING] 잔고 조회 실패: {e}")

    while trading_status['is_running']:
        try:
            # 시장 데이터 가져오기
            df = get_market_data_for_trading(trading_status['market'], 200)
            current_price = float(df.iloc[-1]['close'])
            trading_status['last_price'] = current_price

            # 현재 잔고 조회
            krw_balance = trading_client.get_balance('KRW')
            crypto_symbol = trading_status['market'].split('-')[1]
            crypto_balance = trading_client.get_balance(crypto_symbol)
            total_value = krw_balance + crypto_balance * current_price

            trading_status['current_balance'] = total_value
            trading_status['profit'] = total_value - trading_status['start_balance']
            if trading_status['start_balance'] > 0:
                trading_status['profit_rate'] = (trading_status['profit'] / trading_status['start_balance']) * 100
            else:
                trading_status['profit_rate'] = 0

            # AI 액션 결정
            action = 0  # 기본: Hold
            action_text = "HOLD"

            if trading_agent:
                obs = prepare_observation(df)
                action, _ = trading_agent.predict(obs, deterministic=True)
                action = int(action)
                action_text = ['HOLD', 'BUY', 'SELL'][action]

            trading_status['last_action'] = action_text
            trading_status['last_action_time'] = datetime.now().isoformat()

            # 액션 실행
            if action == 1:  # Buy
                if trading_status['current_position'] is None and krw_balance > 5000:
                    trade_amount = min(krw_balance * 0.5, trading_status['max_trade_amount'])
                    if trade_amount >= 5000:
                        try:
                            result = trading_client.buy_market_order(trading_status['market'], trade_amount)
                            if 'error' not in result:
                                trading_status['current_position'] = 'long'
                                trading_status['trade_count'] += 1
                                trade_record = {
                                    'time': datetime.now().isoformat(),
                                    'action': 'BUY',
                                    'price': current_price,
                                    'amount': trade_amount,
                                    'uuid': result.get('uuid', 'N/A')
                                }
                                trading_status['trade_history'].append(trade_record)
                                print(f"[TRADING] 매수 체결: {trade_amount:,.0f} KRW @ {current_price:,.0f}")

                                # WebSocket으로 브로드캐스트
                                await manager.broadcast({
                                    'type': 'trading_update',
                                    'data': {'action': 'BUY', 'price': current_price, 'amount': trade_amount}
                                })
                        except Exception as e:
                            print(f"[TRADING] 매수 오류: {e}")

            elif action == 2:  # Sell
                if trading_status['current_position'] == 'long' and crypto_balance > 0:
                    try:
                        result = trading_client.sell_market_order(trading_status['market'], crypto_balance)
                        if 'error' not in result:
                            trading_status['current_position'] = None
                            trading_status['trade_count'] += 1
                            trade_record = {
                                'time': datetime.now().isoformat(),
                                'action': 'SELL',
                                'price': current_price,
                                'volume': crypto_balance,
                                'uuid': result.get('uuid', 'N/A')
                            }
                            trading_status['trade_history'].append(trade_record)
                            print(f"[TRADING] 매도 체결: {crypto_balance:.8f} @ {current_price:,.0f}")

                            # WebSocket으로 브로드캐스트
                            await manager.broadcast({
                                'type': 'trading_update',
                                'data': {'action': 'SELL', 'price': current_price, 'volume': crypto_balance}
                            })
                    except Exception as e:
                        print(f"[TRADING] 매도 오류: {e}")

            print(f"[TRADING] {datetime.now().strftime('%H:%M:%S')} - 가격: {current_price:,.0f}, 액션: {action_text}, 수익률: {trading_status['profit_rate']:+.2f}%")

        except Exception as e:
            print(f"[TRADING] 루프 오류: {e}")

        # 지정된 간격만큼 대기
        await asyncio.sleep(trading_status['interval'])

    print("[TRADING] 자동매매 루프 종료")


@app.get("/api/account/check")
async def check_api_connection():
    """API 키 유효성 확인"""
    if not trading_client or not is_valid_api_key(UPBIT_ACCESS_KEY) or not is_valid_api_key(UPBIT_SECRET_KEY):
        return {
            "success": False,
            "connected": False,
            "error": "API 키가 설정되지 않았습니다. .env 파일을 확인하세요."
        }

    try:
        accounts = trading_client.get_accounts()
        if isinstance(accounts, dict) and 'error' in accounts:
            return {
                "success": False,
                "connected": False,
                "error": accounts['error'].get('message', '알 수 없는 오류')
            }

        return {
            "success": True,
            "connected": True,
            "message": "API 연결 성공"
        }
    except Exception as e:
        return {
            "success": False,
            "connected": False,
            "error": str(e)
        }


@app.get("/api/account/balance")
async def get_account_balance():
    """계좌 잔고 조회"""
    if not trading_client:
        return {"success": False, "error": "API 키가 설정되지 않았습니다."}

    try:
        accounts = trading_client.get_accounts()
        if isinstance(accounts, dict) and 'error' in accounts:
            return {"success": False, "error": accounts['error'].get('message', '알 수 없는 오류')}

        balances = []
        for account in accounts:
            balance = float(account['balance'])
            locked = float(account['locked'])
            if balance > 0 or locked > 0:
                balances.append({
                    'currency': account['currency'],
                    'balance': balance,
                    'locked': locked,
                    'avg_buy_price': float(account['avg_buy_price']),
                    'unit_currency': account['unit_currency']
                })

        return {"success": True, "data": balances}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/trading/start")
async def start_trading(request: TradingStartRequest):
    """자동매매 시작"""
    global trading_bot_task, trading_status

    if not trading_client:
        return {"success": False, "error": "API 키가 설정되지 않았습니다."}

    if trading_status['is_running']:
        return {"success": False, "error": "이미 자동매매가 실행 중입니다."}

    # 상태 초기화
    trading_status['is_running'] = True
    trading_status['market'] = request.market
    trading_status['interval'] = request.interval
    trading_status['max_trade_amount'] = request.max_trade_amount
    trading_status['start_time'] = datetime.now().isoformat()
    trading_status['trade_count'] = 0
    trading_status['current_position'] = None
    trading_status['profit'] = 0
    trading_status['profit_rate'] = 0
    trading_status['trade_history'] = []

    # 비동기 트레이딩 루프 시작
    trading_bot_task = asyncio.create_task(trading_loop())

    return {
        "success": True,
        "message": f"자동매매 시작 - {request.market}",
        "data": {
            "market": request.market,
            "interval": request.interval,
            "max_trade_amount": request.max_trade_amount
        }
    }


@app.post("/api/trading/stop")
async def stop_trading():
    """자동매매 중지"""
    global trading_bot_task, trading_status

    if not trading_status['is_running']:
        return {"success": False, "error": "실행 중인 자동매매가 없습니다."}

    trading_status['is_running'] = False

    if trading_bot_task:
        trading_bot_task.cancel()
        try:
            await trading_bot_task
        except asyncio.CancelledError:
            pass
        trading_bot_task = None

    return {
        "success": True,
        "message": "자동매매 중지됨",
        "data": {
            "trade_count": trading_status['trade_count'],
            "profit": trading_status['profit'],
            "profit_rate": trading_status['profit_rate']
        }
    }


@app.get("/api/trading/status")
async def get_trading_status():
    """자동매매 상태 조회"""
    return {
        "success": True,
        "data": clean_dict(trading_status)
    }


@app.get("/api/trading/history")
async def get_trading_history():
    """거래 내역 조회"""
    return {
        "success": True,
        "data": clean_dict(trading_status['trade_history'])
    }


@app.get("/api/trading/realtime-price")
async def get_realtime_price(market: str = "KRW-BTC"):
    """실시간 현재가 조회"""
    try:
        ticker = client.get_ticker([market])
        if ticker and len(ticker) > 0:
            return {
                "success": True,
                "price": ticker[0]['trade_price'],
                "change_rate": ticker[0]['signed_change_rate'] * 100,
                "timestamp": datetime.now().isoformat()
            }
        return {"success": False, "error": "가격 정보 없음"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/trading/chart-data")
async def get_chart_data(market: str = "KRW-BTC", count: int = 60):
    """실시간 차트 데이터 조회 (최근 N개 분봉)"""
    try:
        candles = client.get_candles_minute(market, unit=1, count=count)

        chart_data = []
        for candle in reversed(candles):  # 시간순 정렬
            chart_data.append({
                'time': candle['candle_date_time_kst'],
                'price': candle['trade_price'],
                'open': candle['opening_price'],
                'high': candle['high_price'],
                'low': candle['low_price'],
                'close': candle['trade_price'],
                'volume': candle['candle_acc_trade_volume']
            })

        return {"success": True, "data": chart_data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== 원클릭 자동매매 API ==========

class AutoTradingStartRequest(BaseModel):
    total_investment: float = 50000
    coin_count: int = 3
    analysis_mode: str = "volume_top50"
    trading_interval: int = 60
    coin_category: str = "normal"  # 'safe', 'normal', 'meme', 'all'
    allocation_mode: str = "weighted"  # 'equal' (균등배분) or 'weighted' (점수기반)
    target_profit_percent: float = 10.0  # 목표가 (+%)
    stop_loss_percent: float = 10.0      # 손절가 (-%, 양수로 입력)


# 밈코인 및 고위험 코인 리스트
MEME_COINS = [
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'BABYDOGE',
    'ELON', 'AKITA', 'KISHU', 'SAMO', 'CATE', 'LEASH', 'BONE',
    'TURBO', 'LADYS', 'AIDOGE', 'BOB', 'WOJAK', 'CHAD', 'TOSHI',
    'COQ', 'MYRO', 'SMOG', 'SLERF', 'BOME', 'MEW', 'POPCAT', 'BRETT',
    '0G', 'NOM', 'HIPPO', 'PNUT', 'ACT', 'VIRTUAL', 'PENGU', 'TRUMP'
]

# 대형 안전 코인 리스트 (시가총액 상위 6개)
SAFE_COINS = [
    'BTC',   # 비트코인
    'ETH',   # 이더리움
    'XRP',   # 리플
    'SOL',   # 솔라나
    'ADA',   # 카르다노
    'AVAX',  # 아발란체
]


def filter_coins_by_category(markets: list, category: str) -> list:
    """카테고리에 따라 코인 필터링"""
    if category == 'all':
        # 전체 코인 (필터링 없음)
        return markets
    elif category == 'safe':
        # 안전 코인만
        return [m for m in markets if m['market'].split('-')[1] in SAFE_COINS]
    elif category == 'meme':
        # 밈코인만
        return [m for m in markets if m['market'].split('-')[1] in MEME_COINS]
    else:  # 'normal'
        # 밈코인 제외
        return [m for m in markets if m['market'].split('-')[1] not in MEME_COINS]


async def select_top_coins(coin_count: int, mode: str = "volume_top50", category: str = "normal") -> list:
    """상위 N개 코인 선택

    Args:
        coin_count: 선택할 코인 수
        mode: 분석 범위 (volume_top50, volume_top100)
        category: 코인 카테고리 (safe, normal, meme, all)
    """
    try:
        all_markets = analyzer.get_all_krw_markets()

        # 'safe' 카테고리는 SAFE_COINS에서 직접 선택
        if category == 'safe':
            # SAFE_COINS에 해당하는 마켓만 선택
            safe_markets = []
            for m in all_markets:
                coin_symbol = m['market'].split('-')[1]
                if coin_symbol in SAFE_COINS:
                    safe_markets.append(m)
            markets = safe_markets
            print(f"[AUTO-TRADING] 안전 코인 {len(markets)}개 발견")
        elif category == 'meme':
            # 밈코인만 선택 (거래량 상위)
            tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
            sorted_markets = sorted(
                zip(all_markets[:150], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )
            # 밈코인만 필터링
            markets = [m[0] for m in sorted_markets if m[0]['market'].split('-')[1] in MEME_COINS]
            print(f"[AUTO-TRADING] 밈코인 {len(markets)}개 발견")
        else:
            # 모드에 따라 분석할 종목 선택
            if mode == "volume_top50":
                tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
                sorted_markets = sorted(
                    zip(all_markets[:100], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:50]
                markets = [m[0] for m in sorted_markets]
            elif mode == "volume_top100":
                tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
                sorted_markets = sorted(
                    zip(all_markets[:150], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:100]
                markets = [m[0] for m in sorted_markets]
            else:
                markets = all_markets[:50]  # 기본값

            # 카테고리에 따라 코인 필터링 (normal, all)
            markets = filter_coins_by_category(markets, category)

        print(f"[AUTO-TRADING] 카테고리 '{category}' 필터링 후 {len(markets)}개 코인")

        # 안전 코인 / 밈코인은 간단하게 처리 (분석 실패해도 선택)
        if category in ['safe', 'meme']:
            results = []
            market_codes = [m['market'] for m in markets]
            tickers = client.get_ticker(market_codes)

            # 거래량 기준으로 정렬
            market_ticker_pairs = []
            for market_info in markets:
                market = market_info['market']
                ticker = next((t for t in tickers if t['market'] == market), None)
                if ticker:
                    market_ticker_pairs.append((market_info, ticker))

            # 거래량 내림차순 정렬
            market_ticker_pairs.sort(key=lambda x: x[1].get('acc_trade_price_24h', 0), reverse=True)

            # 순위 기반 점수 부여
            label = '안전 투자' if category == 'safe' else '밈코인'
            base_score = 70 if category == 'safe' else 60

            for i, (market_info, ticker) in enumerate(market_ticker_pairs[:coin_count]):
                # 순위에 따라 점수 차등 부여
                if coin_count <= 3:
                    score_diff = [0, -15, -30]  # 1위, 2위, 3위
                elif coin_count == 4:
                    score_diff = [0, -10, -20, -30]
                else:  # 5개
                    score_diff = [0, -8, -16, -24, -32]

                score = base_score + score_diff[i] if i < len(score_diff) else base_score - 32
                market = market_info['market']

                # AI 기술적 분석 수행
                try:
                    result = analyzer.analyze_market(market, days=30)
                    if result and result.get('current_price', 0) > 0:
                        # 기술적 분석 성공 - AI 기반 가격 계산
                        trade_prices = calculate_trade_prices(result)
                        current_price = result['current_price']
                        recommendation = result.get('recommendation', label)
                        print(f"[AUTO-TRADING] {market} AI 분석 완료: 매수가 ₩{trade_prices['buy_price']:,}")
                    else:
                        # 분석 실패 시 기본값
                        current_price = ticker.get('trade_price', 0)
                        trade_prices = {
                            'buy_price': round(current_price * 0.99),
                            'sell_price': round(current_price * 1.08),
                            'stop_loss': round(current_price * 0.96),
                            'expected_profit_rate': 8.0,
                            'risk_rate': 4.0
                        }
                        recommendation = label
                except Exception as e:
                    print(f"[AUTO-TRADING] {market} 분석 실패: {e}, 기본값 사용")
                    current_price = ticker.get('trade_price', 0)
                    trade_prices = {
                        'buy_price': round(current_price * 0.99),
                        'sell_price': round(current_price * 1.08),
                        'stop_loss': round(current_price * 0.96),
                        'expected_profit_rate': 8.0,
                        'risk_rate': 4.0
                    }
                    recommendation = label

                results.append({
                    'market': market,
                    'name': market_info.get('korean_name', market.split('-')[1]),
                    'current_price': current_price,
                    'score': score,
                    'score_100': score,
                    'recommendation': recommendation,
                    'trade_prices': trade_prices,
                    'ai_action': 0,
                    'ai_confidence': 0
                })

            print(f"[AUTO-TRADING] {label} 점수: {[c['score'] for c in results]}")
            return results

        # 일반/전체 카테고리는 분석 진행
        results = []
        for market_info in markets:
            market = market_info['market']
            try:
                result = analyzer.analyze_market(market, days=30)

                if result and result['score'] > 0:
                    df = analyzer.get_market_data(market, days=30)
                    if df is not None:
                        ai_result = predictor.predict_market(df, market)
                        result['ai_action'] = ai_result['action']
                        result['ai_confidence'] = ai_result['confidence']
                    else:
                        result['ai_action'] = 0
                        result['ai_confidence'] = 0

                    result['score_100'] = normalize_score_to_100(result['score'])
                    trade_prices = calculate_trade_prices(result)
                    result['trade_prices'] = trade_prices
                    result = clean_dict(result)
                    results.append(result)
            except Exception as e:
                print(f"[AUTO-TRADING] 코인 분석 실패 {market}: {e}")
                continue

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:coin_count]
    except Exception as e:
        print(f"[AUTO-TRADING] 코인 선택 실패: {e}")
        return []


async def select_and_allocate_coins():
    """코인 선택 및 투자금 배분"""
    global auto_trading_status

    print("[AUTO-TRADING] 상위 코인 선택 중...")
    coins = await select_top_coins(
        auto_trading_status['coin_count'],
        auto_trading_status.get('analysis_mode', 'volume_top50'),
        auto_trading_status.get('coin_category', 'normal')
    )

    if not coins:
        print("[AUTO-TRADING] 선택된 코인 없음")
        return False

    total = auto_trading_status['total_investment']
    allocation_mode = auto_trading_status.get('allocation_mode', 'weighted')

    # 투자금 배분 계산
    if allocation_mode == 'weighted':
        # 점수 기반 가중치 배분
        # 점수가 0이거나 없는 경우 최소값 10 사용
        scores = [max(coin.get('score_100', 0), 10) for coin in coins]
        total_score = sum(scores)

        # 가중치 비율 계산
        weights = [score / total_score for score in scores]
        allocations = [total * weight for weight in weights]

        print(f"[AUTO-TRADING] 점수 기반 배분: {dict(zip([c['market'] for c in coins], [f'{a:,.0f}원 ({w*100:.1f}%)' for a, w in zip(allocations, weights)]))}")
    else:
        # 균등 배분
        per_coin = total / len(coins)
        allocations = [per_coin] * len(coins)
        print(f"[AUTO-TRADING] 균등 배분: 각 {per_coin:,.0f}원")

    auto_trading_status['positions'] = {}
    auto_trading_status['selected_coins'] = []

    for i, coin in enumerate(coins):
        market = coin['market']
        allocated = allocations[i]
        allocation_percent = (allocated / total) * 100

        auto_trading_status['positions'][market] = {
            'market': market,
            'name': coin.get('name', market.split('-')[1]),
            'allocated_amount': allocated,
            'allocation_percent': round(allocation_percent, 1),
            'entry_price': None,
            'current_price': coin.get('current_price', 0),
            'volume': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'none',
            'score': coin.get('score_100', 0),
            'recommendation': coin.get('recommendation', ''),
            'trade_prices': coin.get('trade_prices', {}),
            'last_action': None,
            'trade_history': []
        }
        auto_trading_status['selected_coins'].append({
            'market': market,
            'name': coin.get('name', market.split('-')[1]),
            'score': coin.get('score_100', 0),
            'recommendation': coin.get('recommendation', ''),
            'allocated_amount': allocated,
            'allocation_percent': round(allocation_percent, 1)
        })

    print(f"[AUTO-TRADING] 선택된 코인: {[c['market'] for c in auto_trading_status['selected_coins']]}")
    print(f"[AUTO-TRADING] 배분 상세:")
    for coin in auto_trading_status['selected_coins']:
        print(f"  - {coin['market']}: {coin['allocation_percent']}% (₩{coin['allocated_amount']:,.0f})")
    return True


async def replace_sold_coin(sold_market: str):
    """매도된 코인을 새로운 코인으로 교체"""
    global auto_trading_status

    print(f"[AUTO-TRADING] {sold_market} 교체 시작...")

    # 기존 코인 목록 (교체할 코인 제외)
    existing_markets = [m for m in auto_trading_status['positions'].keys() if m != sold_market]

    # 새 코인 선택 (기존 코인 제외)
    try:
        all_coins = await select_top_coins(
            coin_count=20,  # 여유있게 선택
            mode=auto_trading_status.get('analysis_mode', 'volume_top50'),
            category=auto_trading_status.get('coin_category', 'normal')
        )

        # 기존 코인 제외하고 새 코인 찾기
        new_coin = None
        for coin in all_coins:
            if coin['market'] not in existing_markets:
                new_coin = coin
                break

        if not new_coin:
            print(f"[AUTO-TRADING] 새로운 코인을 찾을 수 없습니다. {sold_market} 제거만 진행")
            # 매도된 코인 제거
            del auto_trading_status['positions'][sold_market]
            auto_trading_status['selected_coins'] = [
                c for c in auto_trading_status['selected_coins'] if c['market'] != sold_market
            ]
            return False

        # 투자금 배분 (매도된 코인의 배분금액 사용)
        allocated_amount = auto_trading_status['positions'][sold_market]['allocated_amount']
        allocation_percent = auto_trading_status['positions'][sold_market]['allocation_percent']

        # 새 코인으로 교체
        market = new_coin['market']
        auto_trading_status['positions'][market] = {
            'market': market,
            'name': new_coin.get('name', market.split('-')[1]),
            'allocated_amount': allocated_amount,
            'allocation_percent': allocation_percent,
            'entry_price': None,
            'current_price': new_coin.get('current_price', 0),
            'volume': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'none',
            'score': new_coin.get('score_100', 0),
            'recommendation': new_coin.get('recommendation', ''),
            'trade_prices': new_coin.get('trade_prices', {}),
            'last_action': None,
            'trade_history': []
        }

        # selected_coins 업데이트
        auto_trading_status['selected_coins'] = [
            c for c in auto_trading_status['selected_coins'] if c['market'] != sold_market
        ]
        auto_trading_status['selected_coins'].append({
            'market': market,
            'name': new_coin.get('name', market.split('-')[1]),
            'score': new_coin.get('score_100', 0),
            'recommendation': new_coin.get('recommendation', ''),
            'allocated_amount': allocated_amount,
            'allocation_percent': allocation_percent
        })

        # 매도된 코인 제거
        del auto_trading_status['positions'][sold_market]

        print(f"[AUTO-TRADING] ✅ 교체 완료: {sold_market} → {market} (점수: {new_coin.get('score_100', 0)})")
        save_auto_trading_state()  # 상태 저장
        return True

    except Exception as e:
        print(f"[AUTO-TRADING] 코인 교체 실패: {e}")
        return False


async def process_auto_coin_position(market: str, position: dict):
    """개별 코인 포지션 처리"""
    global auto_trading_status, trading_client

    # 목표가/손절가 설정 (상태에서 가져오기)
    TARGET_PROFIT_PERCENT = auto_trading_status.get('target_profit_percent', 10.0)
    STOP_LOSS_PERCENT = -abs(auto_trading_status.get('stop_loss_percent', 10.0))  # 음수로 변환

    try:
        # 시장 데이터 가져오기
        df = get_market_data_for_trading(market, 200)
        current_price = float(df.iloc[-1]['close'])
        position['current_price'] = current_price

        # AI 예측
        action = 0  # 기본: Hold
        action_text = "HOLD"

        if trading_agent:
            obs = prepare_observation(df)
            action, _ = trading_agent.predict(obs, deterministic=True)
            action = int(action)
            action_text = ['HOLD', 'BUY', 'SELL'][action]

        position['last_action'] = action_text

        # 목표가/손절가 체크 (보유 중일 때)
        should_sell_by_target = False
        sell_reason = ""

        if position['status'] == 'long' and position['entry_price']:
            profit_rate = ((current_price - position['entry_price']) / position['entry_price']) * 100

            if profit_rate >= TARGET_PROFIT_PERCENT:
                should_sell_by_target = True
                sell_reason = f"목표가 도달 (+{profit_rate:.1f}%)"
                print(f"[AUTO-TRADING] {market} 목표가 도달! 수익률: +{profit_rate:.1f}%")
            elif profit_rate <= STOP_LOSS_PERCENT:
                should_sell_by_target = True
                sell_reason = f"손절가 도달 ({profit_rate:.1f}%)"
                print(f"[AUTO-TRADING] {market} 손절가 도달! 손실률: {profit_rate:.1f}%")

        # 매수 실행
        if action == 1 and position['status'] == 'none':
            krw_balance = trading_client.get_balance('KRW')
            trade_amount = min(position['allocated_amount'], krw_balance * 0.3)

            if trade_amount >= 5000:
                try:
                    result = trading_client.buy_market_order(market, trade_amount)
                    if 'error' not in result:
                        position['status'] = 'long'
                        position['entry_price'] = current_price
                        position['volume'] = trade_amount / current_price

                        trade_record = {
                            'time': datetime.now().isoformat(),
                            'market': market,
                            'action': 'BUY',
                            'price': current_price,
                            'amount': trade_amount,
                            'uuid': result.get('uuid', 'N/A')
                        }
                        position['trade_history'].append(trade_record)
                        auto_trading_status['trade_history'].append(trade_record)
                        print(f"[AUTO-TRADING] 매수 체결: {market} - {trade_amount:,.0f} KRW @ {current_price:,.0f}")
                        save_auto_trading_state()  # 상태 저장
                except Exception as e:
                    print(f"[AUTO-TRADING] 매수 오류 {market}: {e}")

        # 매도 실행 (AI 신호 또는 목표가/손절가 도달)
        elif (action == 2 or should_sell_by_target) and position['status'] == 'long':
            crypto_symbol = market.split('-')[1]
            crypto_balance = trading_client.get_balance(crypto_symbol)

            if crypto_balance > 0:
                try:
                    result = trading_client.sell_market_order(market, crypto_balance)
                    if 'error' not in result:
                        realized_pnl = (current_price - position['entry_price']) * position['volume']
                        position['realized_pnl'] += realized_pnl

                        # 매도 사유 결정
                        action_label = sell_reason if should_sell_by_target else 'SELL (AI)'

                        trade_record = {
                            'time': datetime.now().isoformat(),
                            'market': market,
                            'action': action_label,
                            'price': current_price,
                            'volume': crypto_balance,
                            'realized_pnl': realized_pnl,
                            'uuid': result.get('uuid', 'N/A')
                        }
                        position['trade_history'].append(trade_record)
                        auto_trading_status['trade_history'].append(trade_record)

                        position['status'] = 'sold'  # 매도 완료 표시 (교체 대상)
                        position['entry_price'] = None
                        position['volume'] = 0
                        position['unrealized_pnl'] = 0
                        position['sold_time'] = datetime.now().isoformat()
                        print(f"[AUTO-TRADING] 매도 체결: {market} - {crypto_balance:.8f} @ {current_price:,.0f} (손익: {realized_pnl:,.0f}) [{action_label}]")
                        print(f"[AUTO-TRADING] {market} 자리에 새로운 코인 선택 예정")
                        save_auto_trading_state()  # 상태 저장
                except Exception as e:
                    print(f"[AUTO-TRADING] 매도 오류 {market}: {e}")

        # 미실현 손익 업데이트
        if position['status'] == 'long' and position['entry_price']:
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['volume']

    except Exception as e:
        print(f"[AUTO-TRADING] 포지션 처리 오류 {market}: {e}")


async def update_auto_portfolio_status():
    """포트폴리오 전체 상태 업데이트"""
    global auto_trading_status, trading_client

    try:
        # KRW 잔고
        krw_balance = trading_client.get_balance('KRW')

        # 코인 평가액 계산
        total_crypto_value = 0
        total_unrealized_pnl = 0
        total_realized_pnl = 0

        for market, position in auto_trading_status['positions'].items():
            if position['status'] == 'long':
                total_crypto_value += position['current_price'] * position['volume']
                total_unrealized_pnl += position['unrealized_pnl']
            total_realized_pnl += position['realized_pnl']

        # 전체 상태 업데이트
        auto_trading_status['current_balance'] = krw_balance + total_crypto_value
        auto_trading_status['profit'] = auto_trading_status['current_balance'] - auto_trading_status['start_balance']

        if auto_trading_status['start_balance'] > 0:
            auto_trading_status['profit_rate'] = (auto_trading_status['profit'] / auto_trading_status['start_balance']) * 100
        else:
            auto_trading_status['profit_rate'] = 0

    except Exception as e:
        print(f"[AUTO-TRADING] 포트폴리오 상태 업데이트 오류: {e}")


async def auto_trading_loop():
    """다중 코인 자동매매 루프"""
    global auto_trading_status, trading_agent, trading_client

    print("[AUTO-TRADING] 자동매매 루프 시작")

    # AI 모델 로드
    try:
        sample_market = list(auto_trading_status['positions'].keys())[0] if auto_trading_status['positions'] else 'KRW-BTC'
        dummy_data = get_market_data_for_trading(sample_market, 200)
        dummy_env = CryptoTradingEnv(dummy_data)
        trading_agent = TradingAgent(dummy_env)
        trading_agent.load('models/crypto_trader')
        print("[AUTO-TRADING] AI 모델 로드 완료")
    except Exception as e:
        print(f"[AUTO-TRADING] AI 모델 로드 실패: {e}")
        trading_agent = None

    # 초기 잔고 기록
    try:
        auto_trading_status['start_balance'] = trading_client.get_balance('KRW')
        auto_trading_status['current_balance'] = auto_trading_status['start_balance']
    except Exception as e:
        print(f"[AUTO-TRADING] 잔고 조회 실패: {e}")

    while auto_trading_status['is_running']:
        try:
            # 각 코인 포지션 처리
            markets_to_process = list(auto_trading_status['positions'].keys())
            for market in markets_to_process:
                if not auto_trading_status['is_running']:
                    break
                position = auto_trading_status['positions'].get(market)
                if position:
                    await process_auto_coin_position(market, position)
                await asyncio.sleep(0.5)  # API 제한 방지

            # 매도된 코인 교체
            sold_coins = [
                market for market, pos in auto_trading_status['positions'].items()
                if pos.get('status') == 'sold'
            ]

            for sold_market in sold_coins:
                if not auto_trading_status['is_running']:
                    break
                print(f"[AUTO-TRADING] {sold_market} 매도 완료, 새 코인으로 교체 중...")
                await replace_sold_coin(sold_market)
                await asyncio.sleep(1)  # 교체 후 대기

            # 포트폴리오 상태 업데이트
            await update_auto_portfolio_status()

            # WebSocket 브로드캐스트
            await manager.broadcast({
                'type': 'auto_trading_update',
                'data': clean_dict(auto_trading_status)
            })

            print(f"[AUTO-TRADING] {datetime.now().strftime('%H:%M:%S')} - 수익률: {auto_trading_status['profit_rate']:+.2f}%")

            # 주기적으로 상태 저장 (매 루프마다)
            save_auto_trading_state()

        except Exception as e:
            print(f"[AUTO-TRADING] 루프 오류: {e}")

        await asyncio.sleep(auto_trading_status.get('trading_interval', 60))

    # 종료 시 최종 상태 저장
    save_auto_trading_state()
    print("[AUTO-TRADING] 자동매매 루프 종료")


@app.post("/api/auto-trading/start")
async def start_auto_trading(request: AutoTradingStartRequest):
    """원클릭 자동매매 시작"""
    global auto_trading_task, auto_trading_status

    if auto_trading_status['is_running']:
        return {"success": False, "error": "이미 자동매매가 실행 중입니다."}

    # 최소 투자금 확인
    if request.total_investment < 50000:
        return {"success": False, "error": "최소 투자금은 50,000원입니다."}

    # 상태 초기화
    auto_trading_status['is_running'] = True
    auto_trading_status['total_investment'] = request.total_investment
    auto_trading_status['coin_count'] = min(max(request.coin_count, 1), 5)  # 1-5 제한
    auto_trading_status['analysis_mode'] = request.analysis_mode
    auto_trading_status['coin_category'] = request.coin_category
    auto_trading_status['trading_interval'] = request.trading_interval
    auto_trading_status['allocation_mode'] = request.allocation_mode
    auto_trading_status['target_profit_percent'] = request.target_profit_percent
    auto_trading_status['stop_loss_percent'] = request.stop_loss_percent
    auto_trading_status['start_time'] = datetime.now().isoformat()
    auto_trading_status['start_balance'] = 0
    auto_trading_status['current_balance'] = 0
    auto_trading_status['profit'] = 0
    auto_trading_status['profit_rate'] = 0
    auto_trading_status['trade_history'] = []

    # 초기 코인 선택
    success = await select_and_allocate_coins()
    if not success:
        auto_trading_status['is_running'] = False
        return {"success": False, "error": "코인 선택에 실패했습니다."}

    # API 키가 있으면 트레이딩 루프 시작
    if trading_client:
        auto_trading_task = asyncio.create_task(auto_trading_loop())

    return {
        "success": True,
        "message": f"원클릭 자동매매 시작 - {len(auto_trading_status['selected_coins'])}개 코인",
        "data": {
            "total_investment": request.total_investment,
            "coin_count": auto_trading_status['coin_count'],
            "selected_coins": auto_trading_status['selected_coins']
        }
    }


class AutoTradingPreviewRequest(BaseModel):
    coin_count: int = 3
    analysis_mode: str = "volume_top50"
    coin_category: str = "safe"
    allocation_mode: str = "weighted"  # 'equal' or 'weighted'
    total_investment: float = 50000  # 투자금 (배분 계산용)


async def get_preview_coins_fast(coin_count: int, category: str) -> list:
    """빠른 코인 미리보기 (간단한 정보만)"""
    try:
        all_markets = analyzer.get_all_krw_markets()
        print(f"[PREVIEW] 전체 마켓 수: {len(all_markets)}")

        if category == 'safe':
            # 안전 코인만 선택
            markets = []
            for m in all_markets:
                coin_symbol = m['market'].split('-')[1]
                if coin_symbol in SAFE_COINS:
                    markets.append(m)
                    print(f"[PREVIEW] 안전 코인 발견: {m['market']}")
            print(f"[PREVIEW] 안전 코인 총 {len(markets)}개 발견")
        elif category == 'meme':
            # 밈코인만 선택 (거래량 상위)
            tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
            sorted_markets = sorted(
                zip(all_markets[:150], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )
            # 밈코인만 필터링
            markets = [m[0] for m in sorted_markets if m[0]['market'].split('-')[1] in MEME_COINS]
            print(f"[PREVIEW] 밈코인 {len(markets)}개 발견")
        elif category == 'normal':
            # 밈코인 제외, 거래량 상위
            tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
            sorted_markets = sorted(
                zip(all_markets[:100], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )[:50]
            markets = [m[0] for m in sorted_markets]
            markets = [m for m in markets if m['market'].split('-')[1] not in MEME_COINS]
        else:  # 'all'
            # 거래량 상위
            tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
            sorted_markets = sorted(
                zip(all_markets[:100], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )[:50]
            markets = [m[0] for m in sorted_markets]

        if not markets:
            print(f"[PREVIEW] 마켓이 비어있음!")
            return []

        # 선택된 마켓의 현재가 조회
        market_codes = [m['market'] for m in markets[:coin_count * 2]]  # 여유있게 조회
        print(f"[PREVIEW] 조회할 마켓: {market_codes}")

        tickers = client.get_ticker(market_codes)
        print(f"[PREVIEW] 조회된 티커 수: {len(tickers) if tickers else 0}")

        results = []
        for i, market_info in enumerate(markets):
            if len(results) >= coin_count:
                break

            market = market_info['market']
            ticker = next((t for t in tickers if t['market'] == market), None)

            if ticker:
                change_rate = ticker.get('signed_change_rate', 0) * 100
                current_price = ticker.get('trade_price', 0)

                # 순위 기반 점수 계산 (1위: 100점, 순위가 내려갈수록 감소)
                # coin_count에 따라 점수 범위 조정
                if coin_count <= 3:
                    score_range = [100, 70, 50]  # 3개: 큰 차이
                elif coin_count == 4:
                    score_range = [100, 80, 60, 40]  # 4개: 중간 차이
                else:  # 5개
                    score_range = [100, 85, 70, 55, 40]  # 5개: 단계적 차이

                score = score_range[i] if i < len(score_range) else 40

                # AI 기술적 분석 기반 trade_prices 계산
                try:
                    # 간단한 분석 데이터 생성 (현재가 기준)
                    tech_data = {
                        'current_price': current_price,
                        'rsi': 50,  # 중립
                        'bb_low': current_price * 0.95,
                        'bb_high': current_price * 1.05,
                        'recommendation': '매수' if change_rate > 0 else '중립'
                    }
                    trade_prices = calculate_trade_prices(tech_data)
                except Exception as e:
                    print(f"[PREVIEW] {market} 가격 계산 실패: {e}")
                    trade_prices = {
                        'buy_price': round(current_price * 0.99),
                        'sell_price': round(current_price * 1.08),
                        'stop_loss': round(current_price * 0.96),
                        'expected_profit_rate': 8.0,
                        'risk_rate': 4.0
                    }

                results.append({
                    'market': market,
                    'name': market_info.get('korean_name', market.split('-')[1]),
                    'current_price': current_price,
                    'change_rate': change_rate,
                    'score': score,
                    'recommendation': '분석 대기',
                    'trade_prices': trade_prices
                })
            else:
                print(f"[PREVIEW] 티커 없음: {market}")

        print(f"[PREVIEW] 최종 결과: {len(results)}개, 점수: {[c['score'] for c in results]}")
        return results
    except Exception as e:
        print(f"[AUTO-TRADING] 빠른 미리보기 실패: {e}")
        return []


@app.post("/api/auto-trading/preview")
async def preview_auto_trading(request: AutoTradingPreviewRequest):
    """코인 선택 미리보기 (빠른 버전)"""
    try:
        coin_count = min(max(request.coin_count, 1), 5)
        category = request.coin_category
        allocation_mode = request.allocation_mode
        total_investment = request.total_investment

        print(f"=" * 50)
        print(f"[PREVIEW API] 요청 받음")
        print(f"[PREVIEW API] 카테고리: '{category}'")
        print(f"[PREVIEW API] 코인 수: {coin_count}")
        print(f"[PREVIEW API] 배분 방식: '{allocation_mode}'")
        print(f"=" * 50)

        # 빠른 미리보기 사용
        coins = await get_preview_coins_fast(coin_count, category)

        if not coins:
            return {"success": False, "error": "선택된 코인이 없습니다."}

        # 배분 비율 계산
        if allocation_mode == 'weighted' and len(coins) > 0:
            # 점수 기반 가중치 배분
            scores = [max(coin.get('score', 50), 10) for coin in coins]
            total_score = sum(scores)
            for i, coin in enumerate(coins):
                weight = scores[i] / total_score
                coin['allocation_percent'] = round(weight * 100, 1)
                coin['allocated_amount'] = round(total_investment * weight)
        else:
            # 균등 배분
            per_coin = total_investment / len(coins) if coins else 0
            percent_per_coin = 100 / len(coins) if coins else 0
            for coin in coins:
                coin['allocation_percent'] = round(percent_per_coin, 1)
                coin['allocated_amount'] = round(per_coin)

        return {
            "success": True,
            "data": {
                "coin_count": len(coins),
                "coin_category": request.coin_category,
                "allocation_mode": allocation_mode,
                "selected_coins": coins
            }
        }
    except Exception as e:
        print(f"[AUTO-TRADING] 미리보기 실패: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/auto-trading/stop")
async def stop_auto_trading():
    """원클릭 자동매매 중지 및 전체 청산"""
    global auto_trading_task, auto_trading_status, trading_client

    if not auto_trading_status['is_running']:
        return {"success": False, "error": "실행 중인 자동매매가 없습니다."}

    auto_trading_status['is_running'] = False

    # 태스크 취소
    if auto_trading_task:
        auto_trading_task.cancel()
        try:
            await auto_trading_task
        except asyncio.CancelledError:
            pass
        auto_trading_task = None

    # 모든 포지션 청산
    print("[AUTO-TRADING] 전체 포지션 청산 중...")
    for market, position in auto_trading_status['positions'].items():
        if position['status'] == 'long':
            crypto_symbol = market.split('-')[1]
            try:
                crypto_balance = trading_client.get_balance(crypto_symbol)
                if crypto_balance > 0:
                    result = trading_client.sell_market_order(market, crypto_balance)
                    if 'error' not in result:
                        print(f"[AUTO-TRADING] 청산 완료: {market}")

                        trade_record = {
                            'time': datetime.now().isoformat(),
                            'market': market,
                            'action': 'SELL (청산)',
                            'price': position['current_price'],
                            'volume': crypto_balance
                        }
                        auto_trading_status['trade_history'].append(trade_record)
            except Exception as e:
                print(f"[AUTO-TRADING] 청산 오류 {market}: {e}")

    # 최종 상태 업데이트
    await update_auto_portfolio_status()

    return {
        "success": True,
        "message": "자동매매 중지 및 전체 청산 완료",
        "data": clean_dict(auto_trading_status)
    }


@app.get("/api/auto-trading/status")
async def get_auto_trading_status():
    """원클릭 자동매매 상태 조회"""
    # 현재 KRW 잔고 조회 (자동매매 시작 전에도 표시하기 위함)
    available_balance = 0
    if trading_client:
        try:
            available_balance = trading_client.get_balance('KRW')
        except Exception as e:
            print(f"[AUTO-TRADING] 잔고 조회 실패: {e}")

    result = dict(auto_trading_status)
    result['available_balance'] = available_balance

    return {
        "success": True,
        "data": clean_dict(result)
    }


@app.get("/api/auto-trading/mini-charts")
async def get_mini_charts():
    """선택된 코인들의 미니 차트 데이터 조회"""
    if not auto_trading_status['selected_coins']:
        return {"success": True, "data": {}}

    charts = {}
    for coin in auto_trading_status['selected_coins']:
        market = coin['market']
        try:
            # 최근 30개 분봉 데이터
            candles = client.get_candles_minute(market, unit=5, count=30)
            prices = [c['trade_price'] for c in reversed(candles)]
            charts[market] = prices
        except Exception as e:
            print(f"[MINI-CHART] {market} 데이터 조회 실패: {e}")
            charts[market] = []

    return {"success": True, "data": charts}


if __name__ == "__main__":
    print("="*60)
    print("[START] 업비트 실시간 분석 웹앱 시작")
    print("="*60)
    print()
    print("[WEB] 웹 브라우저에서 접속하세요:")
    print("   http://localhost:8000")
    print()
    print("[EXIT] 종료: Ctrl+C")
    print("="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
