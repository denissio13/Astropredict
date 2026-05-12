#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import datetime
import math
import time
from itertools import product

# Попытка импорта библиотек
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌ Ошибка: Требуется pandas и numpy. Установите: pip install pandas numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ Ошибка: Требуется requests. Установите: pip install requests")
    sys.exit(1)

# Проверка GPU (CuPy)
USE_GPU = False
np_array = np.array

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        device_id = cp.cuda.runtime.getDevice()
        prop = cp.cuda.runtime.getDeviceProperties(device_id)
        device_name = prop['name'].decode('utf-8')
        print(f"✅ GPU Detected: {device_name}")
        print("🚀 Using GPU acceleration for calculations...")
        USE_GPU = True
        np_array = cp.array
        def to_host(x):
            return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x
    else:
        print("⚠️ CUDA detected but no devices found. Falling back to CPU.")
except ImportError:
    print("⚠️ CuPy not installed. Running on CPU (slower). Install with: pip install cupy-cuda12x")
except Exception as e:
    print(f"⚠️ GPU Error: {e}. Falling back to CPU.")

# Импорты астрологии (ИСПРАВЛЕНО: один блок try-except)
HAS_SWISSEPHE = False
try:
    import swisseph as swe
    HAS_SWISSEPHE = True
    print("✅ swisseph loaded. Using high precision calculations.")
    swe.set_ephe_path(None) 
except ImportError:
    print("⚠️ swisseph not installed. Using simplified approximations.")

class AstroCalculator:
    def __init__(self):
        if HAS_SWISSEPHE:
            swe.set_topo_centric(0.0, 0.0, 0.0)

    def get_sun_moon_positions(self, date_str):
        d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        jul_day = swe.julday(d.year, d.month, d.day, 0.0)
        
        if HAS_SWISSEPHE:
            sun_lon = swe.calc_ut(jul_day, swe.SUN)[0][0]
            moon_lon = swe.calc_ut(jul_day, swe.MOON)[0][0]
        else:
            base_date = datetime.datetime(2000, 1, 1)
            days_diff = (d - base_date).days
            sun_lon = (280.46 + 0.9856474 * days_diff) % 360
            moon_lon = (218.32 + 13.176396 * days_diff) % 360
            
        return sun_lon % 360, moon_lon % 360

    def get_moon_phase(self, sun_lon, moon_lon):
        diff = (moon_lon - sun_lon) % 360
        phase = diff / 360.0
        illumination = (1 - math.cos(math.radians(diff))) / 2
        return phase, illumination

    def get_zodiac_sign(self, longitude):
        return int(longitude // 30)

    def get_aspect(self, sun_lon, moon_lon):
        diff = abs(sun_lon - moon_lon)
        if diff > 180: diff = 360 - diff
        aspects = [0, 60, 90, 120, 180]
        tolerance = 5
        for asp in aspects:
            if abs(diff - asp) <= tolerance:
                return asp
        return -1

class CryptoDataLoader:
    def __init__(self, exchange='binance'):
        self.exchange = exchange
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_data(self, symbol, limit=1000):
        filename = f"{self.data_dir}/{self.exchange}_{symbol}_daily.csv"
        
        if os.path.exists(filename):
            stat = os.stat(filename)
            if time.time() - stat.st_mtime < 86400:
                print(f"📂 Loading cached data for {symbol}...")
                return pd.read_csv(filename)

        print(f"🌐 Fetching data for {symbol} from {self.exchange}...")
        df = pd.DataFrame()
        
        try:
            if self.exchange == 'binance':
                url = f"https://api.binance.com/api/v3/klines"
                params = {'symbol': symbol, 'interval': '1d', 'limit': limit}
                resp = requests.get(url, params=params)
                data = resp.json()
                
                if isinstance(data, dict) and 'msg' in data:
                    raise Exception(data['msg'])
                
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 
                                                 'close_time', 'quote_av', 'trades', 'tb_base_av', 
                                                 'tb_quote_av', 'ignore'])
                df['date'] = pd.to_datetime(df['time'], unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
            elif self.exchange == 'bybit':
                url = f"https://api.bybit.com/v5/market/kline"
                params = {'category': 'spot', 'symbol': symbol, 'interval': 'D', 'limit': limit}
                resp = requests.get(url, params=params)
                data = resp.json()
                
                if data['retCode'] != 0:
                    raise Exception(data['retMsg'])
                    
                rows = data['result']['list']
                df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume', 
                                                 'turnover', 'confirmations'])
                df['date'] = pd.to_datetime(df['time'], unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.iloc[::-1].reset_index(drop=True)

            df.to_csv(filename, index=False)
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            if os.path.exists(filename):
                print("⚠️ Fallback to old cache...")
                return pd.read_csv(filename)
            return None

class AstrologicalAnalyzer:
    def __init__(self, df, astro_calc, weights=None):
        self.df = df.copy()
        self.astro_calc = astro_calc
        self.weights = weights or {
            'moon_phase': 3.0,
            'moon_sign': 2.0,
            'illumination': 2.0,
            'sun_moon_aspect': 2.0,
            'venus_sign': 1.0,
            'mars_sign': 1.0
        }
        self.cache_features = {}

    def calculate_astro_features(self, date_obj):
        date_str = date_obj.strftime("%Y-%m-%d")
        if date_str in self.cache_features:
            return self.cache_features[date_str]

        s_lon, m_lon = self.astro_calc.get_sun_moon_positions(date_str)
        phase, illum = self.astro_calc.get_moon_phase(s_lon, m_lon)
        m_sign = self.astro_calc.get_zodiac_sign(m_lon)
        aspect = self.astro_calc.get_aspect(s_lon, m_lon)
        
        v_lon = (s_lon - 45) % 360
        m_planet_lon = (s_lon + 60) % 360
        
        if HAS_SWISSEPHE:
             d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
             jd = swe.julday(d.year, d.month, d.day, 0.0)
             v_lon = swe.calc_ut(jd, swe.VENUS)[0][0]
             m_planet_lon = swe.calc_ut(jd, swe.MARS)[0][0]
             
        v_sign = self.astro_calc.get_zodiac_sign(v_lon)
        mars_sign = self.astro_calc.get_zodiac_sign(m_planet_lon)

        features = {
            'phase': phase,
            'moon_sign': m_sign,
            'illumination': illum,
            'aspect': aspect,
            'venus_sign': v_sign,
            'mars_sign': mars_sign
        }
        self.cache_features[date_str] = features
        return features

    def calculate_similarity_score(self, f1, f2):
        score = 0.0
        total_weight = 0.0

        diff_phase = abs(f1['phase'] - f2['phase'])
        if diff_phase > 0.5: diff_phase = 1 - diff_phase
        score += (1 - diff_phase) * self.weights['moon_phase']
        total_weight += self.weights['moon_phase']

        if f1['moon_sign'] == f2['moon_sign']:
            score += self.weights['moon_sign']
        total_weight += self.weights['moon_sign']

        diff_illum = abs(f1['illumination'] - f2['illumination'])
        score += (1 - diff_illum) * self.weights['illumination']
        total_weight += self.weights['illumination']

        if f1['aspect'] == f2['aspect']:
            score += self.weights['sun_moon_aspect']
        total_weight += self.weights['sun_moon_aspect']

        if f1['venus_sign'] == f2['venus_sign']:
            score += self.weights['venus_sign']
        total_weight += self.weights['venus_sign']

        if f1['mars_sign'] == f2['mars_sign']:
            score += self.weights['mars_sign']
        total_weight += self.weights['mars_sign']

        return score / total_weight if total_weight > 0 else 0

    def find_similar_patterns(self, target_idx, window=5, top_n=10, similarity_threshold=0.85):
        if target_idx < window:
            return []

        target_dates = self.df['date'].iloc[target_idx-window:target_idx].values
        target_features = [self.calculate_astro_features(pd.Timestamp(d).to_pydatetime()) for d in target_dates]
        
        matches = []
        end_search = target_idx - window 
        
        for i in range(0, end_search):
            hist_dates = self.df['date'].iloc[i:i+window].values
            if len(hist_dates) < window: break
            
            hist_features = [self.calculate_astro_features(pd.Timestamp(d).to_pydatetime()) for d in hist_dates]
            
            scores = []
            for tf, hf in zip(target_features, hist_features):
                scores.append(self.calculate_similarity_score(tf, hf))
            
            avg_score = sum(scores) / len(scores)
            
            if avg_score >= similarity_threshold:
                matches.append({
                    'index': i + window - 1,
                    'score': avg_score,
                    'date': self.df['date'].iloc[i+window-1],
                    'next_open': self.df['open'].iloc[i+window],
                    'next_close': self.df['close'].iloc[i+window],
                    'next_high': self.df['high'].iloc[i+window],
                    'next_low': self.df['low'].iloc[i+window]
                })
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_n]

    def get_prediction_direction(self, matches):
        if not matches:
            return "NEUTRAL", 0.0, 0.0
        
        bullish = 0
        bearish = 0
        total_pnl = 0.0
        count = 0
        
        for m in matches:
            move = m['next_close'] - m['next_open']
            if move > 0: bullish += 1
            else: bearish += 1
            
            total_pnl += move 
            count += 1
            
        direction = "BULLISH" if bullish > bearish else ("BEARISH" if bearish > bullish else "NEUTRAL")
        confidence = max(bullish, bearish) / count if count > 0 else 0
        avg_pnl = total_pnl / count if count > 0 else 0
        
        return direction, confidence, avg_pnl

    def backtest(self, start_idx, end_idx, window=5, min_matches=3, tolerance=0.85):
        total_trades = 0
        wins = 0
        total_pnl = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        max_drawdown = 0.0
        current_drawdown = 0.0
        errors = []

        print(f"\n🔄 Starting Backtest... (GPU={'Yes' if USE_GPU else 'No'})")
        
        for i in range(start_idx, end_idx):
            matches = self.find_similar_patterns(i, window=window, top_n=20, similarity_threshold=tolerance)
            
            if len(matches) >= min_matches:
                direction, conf, avg_pnl = self.get_prediction_direction(matches)
                
                if i + 1 >= len(self.df): break
                
                entry_price = self.df['close'].iloc[i]
                exit_price = self.df['close'].iloc[i+1]
                
                trade_pnl = 0.0
                
                if direction == "BULLISH":
                    trade_pnl = exit_price - entry_price
                elif direction == "BEARISH":
                    trade_pnl = entry_price - exit_price
                
                if trade_pnl > 0:
                    wins += 1
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)
                    errors.append({
                        'date': str(self.df['date'].iloc[i+1])[:10],
                        'direction': direction,
                        'pnl': trade_pnl,
                        'matches': len(matches)
                    })

                total_pnl += trade_pnl
                total_trades += 1
                
                current_drawdown += trade_pnl
                if current_drawdown < 0:
                    if abs(current_drawdown) > max_drawdown:
                        max_drawdown = abs(current_drawdown)
                else:
                    current_drawdown = 0

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.9
        avg_win = (gross_profit / wins) if wins > 0 else 0
        avg_loss = (gross_loss / (total_trades - wins)) if (total_trades - wins) > 0 else 0
        
        errors.sort(key=lambda x: x['pnl'])
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': pf,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'errors': errors[:5]
        }

    def optimize_parameters(self, candle_range=200, mode='basic'):
        print(f"\n🔍 Optimizing Parameters... (Mode: {mode}, GPU={'Yes' if USE_GPU else 'No'})")
        
        best_result = None
        best_score = -float('inf')
        best_params = {}
        
        windows = [3, 5, 7, 10]
        thresholds = [0.75, 0.80, 0.85, 0.90]
        min_matches_list = [1, 2, 3, 5]
        
        weight_sets = [None]
        if mode == 'full':
            print("⚠️ Full mode: Testing different weight configurations (this will take a while)...")
            weight_sets = [
                None,
                {'moon_phase': 5.0, 'moon_sign': 1.0, 'illumination': 1.0, 'sun_moon_aspect': 1.0, 'venus_sign': 1.0, 'mars_sign': 1.0},
                {'moon_phase': 1.0, 'moon_sign': 3.0, 'illumination': 3.0, 'sun_moon_aspect': 1.0, 'venus_sign': 1.0, 'mars_sign': 1.0},
                {'moon_phase': 2.0, 'moon_sign': 2.0, 'illumination': 2.0, 'sun_moon_aspect': 3.0, 'venus_sign': 1.0, 'mars_sign': 1.0}
            ]

        total_combinations = len(windows) * len(thresholds) * len(min_matches_list) * len(weight_sets)
        current_combo = 0

        for w in windows:
            for t in thresholds:
                for m in min_matches_list:
                    for ws in weight_sets:
                        current_combo += 1
                        print(f"\rProgress: {current_combo}/{total_combinations}", end="")
                        
                        if ws:
                            self.weights = ws
                        
                        start_i = max(0, len(self.df) - candle_range - w - 50)
                        end_i = len(self.df) - 1
                        
                        if end_i - start_i < 10: continue

                        res = self.backtest(start_i, end_i, window=w, min_matches=m, tolerance=t)
                        
                        if res['total_trades'] < 5: continue

                        score = res['total_pnl'] - (res['max_drawdown'] * 0.5) + (res['profit_factor'] * 10)
                        
                        if score > best_score:
                            best_score = score
                            best_result = res
                            best_params = {
                                'window': w,
                                'tolerance': t,
                                'min_matches': m,
                                'weights': ws
                            }
        
        print("\n\n🏆 OPTIMIZATION COMPLETE")
        print("="*40)
        if best_result:
            wp = best_params['weights']
            w_str = "Default" if wp is None else "Custom"
            print(f"Best Params: Window={best_params['window']}, Thr={best_params['tolerance']}, MinMatch={best_params['min_matches']}, Weights={w_str}")
            print(f"Trades: {best_result['total_trades']} | WinRate: {best_result['win_rate']:.2f}%")
            print(f"Total PnL: {best_result['total_pnl']:.2f} | Profit Factor: {best_result['profit_factor']:.2f}")
            print(f"Max Drawdown: {best_result['max_drawdown']:.2f}")
            print(f"Avg Win: {best_result['avg_win']:.2f} | Avg Loss: {best_result['avg_loss']:.2f}")
            
            if best_result['errors']:
                print("\n⚠️ Top 5 Worst Trades:")
                for err in best_result['errors']:
                    print(f"   Date: {err['date']}, Dir: {err['direction']}, PnL: {err['pnl']:.2f}, Matches: {err['matches']}")
            
            print("\n💡 Recommendations:")
            if best_result['profit_factor'] < 1.2:
                print("   - Profit Factor low. Try increasing 'tolerance' to filter weak patterns.")
            if best_result['max_drawdown'] > best_result['total_pnl'] * 0.5:
                print("   - High drawdown relative to profit. Reduce 'window' size or increase 'min_matches'.")
            if best_result['win_rate'] < 45:
                print("   - Low win rate. The strategy might be counter-trend. Check aspect weights.")
            if best_result['total_trades'] < 20:
                print("   - Too few trades. Decrease 'min_matches' or 'tolerance' to find more patterns.")
                
            output_file = "optimization_result.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'params': best_params,
                    'stats': {k: v for k, v in best_result.items() if k != 'equity_curve'},
                    'timestamp': datetime.datetime.now().isoformat()
                }, f, indent=2)
            print(f"\n💾 Detailed results saved to {output_file}")
        else:
            print("❌ No profitable configuration found.")

def main():
    parser = argparse.ArgumentParser(description="Astrological Crypto Analyzer with GPU Support")
    parser.add_argument('--pairs', type=str, required=True, help="Pairs list (e.g., BTCUSDT,ETHUSDT)")
    parser.add_argument('--exchange', type=str, default='binance', choices=['binance', 'bybit'])
    parser.add_argument('--target-date', '-d', type=str, help="Target date for prediction (YYYY-MM-DD)")
    parser.add_argument('--backtest', action='store_true', help="Run backtest on historical data")
    parser.add_argument('--optimize', action='store_true', help="Run parameter optimization")
    parser.add_argument('--mode', type=str, default='basic', choices=['basic', 'full'], help="Optimization mode: basic (fast) or full (slow, tests weights)")
    parser.add_argument('--candles', type=int, default=200, help="Number of candles for backtest/optimize")
    parser.add_argument('--similarity', type=float, default=0.85, help="Similarity threshold")
    parser.add_argument('--max-matches', type=int, default=10, help="Max matches to consider")

    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(',')]
    loader = CryptoDataLoader(exchange=args.exchange)
    astro_calc = AstroCalculator()

    for pair in pairs:
        print(f"\n{'='*20} {pair} {'='*20}")
        df = loader.fetch_data(pair)
        if df is None or len(df) < 50:
            print(f"❌ Not enough data for {pair}")
            continue

        analyzer = AstrologicalAnalyzer(df, astro_calc)

        if args.optimize:
            analyzer.optimize_parameters(candle_range=args.candles, mode=args.mode)
        elif args.backtest:
            start_i = max(0, len(df) - args.candles - 20)
            end_i = len(df) - 1
            res = analyzer.backtest(start_i, end_i, window=5, min_matches=2, tolerance=args.similarity)
            print(json.dumps(res, indent=2))
        else:
            last_idx = len(df) - 1
            if args.target_date:
                try:
                    t_date = pd.to_datetime(args.target_date)
                    mask = df['date'] == t_date
                    if not mask.any():
                        idx = (df['date'] - t_date).abs().argsort()[:1].values[0]
                        last_idx = idx
                        print(f"⚠️ Exact date not found, using nearest: {df['date'].iloc[last_idx]}")
                    else:
                        last_idx = mask[mask].index[0]
                except:
                    pass

            matches = analyzer.find_similar_patterns(last_idx, top_n=args.max_matches, similarity_threshold=args.similarity)
            direction, conf, avg_pnl = analyzer.get_prediction_direction(matches)
            
            print(f"🔮 Prediction for {df['date'].iloc[last_idx]}:")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {conf:.2%}")
            print(f"   Avg Hist PnL: {avg_pnl:.2f}")
            print(f"   Matches Found: {len(matches)}")

if __name__ == "__main__":
    main()
