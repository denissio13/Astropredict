#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astrological Trading Analyzer with GPU Acceleration
Supports: Binance/Bybit data loading, Astrological patterns, Backtesting, Optimization
GPU: NVIDIA CUDA (via CuPy) for fast pattern matching

Usage:
    python astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT
    python astro_trading_analyzer.py --pairs BTCUSDT --backtest --candles 200
    python astro_trading_analyzer.py --pairs BTCUSDT --optimize
    python astro_trading_analyzer.py --pairs BTCUSDT --target-date 2024-06-15
"""

import argparse
import json
import os
import sys
import datetime
import math
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# Try to import GPU libraries, fallback to CPU if not available
USE_GPU = False
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_GPU = True
        print(f"✅ GPU Detected: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
        print(f"🚀 Using GPU acceleration for calculations...")
    else:
        print("⚠️  CUDA device found but no devices available. Falling back to CPU.")
except ImportError:
    print("⚠️  CuPy not installed. Falling back to CPU (NumPy).")
    print("   To enable GPU: pip install cupy-cuda12x")
except Exception as e:
    print(f"⚠️  CUDA Error: {e}. Falling back to CPU.")

if USE_GPU:
    xp = cp
else:
    xp = np

# Optional: Swiss Ephemeris
try:
    import swisseph as swe
    HAS_SWISSEPH = True
except ImportError:
    HAS_SWISSEPH = False
    print("⚠️  swisseph not installed. Using simplified approximations.")

DATA_DIR = Path("market_data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class AstroCalculator:
    def __init__(self):
        if HAS_SWISSEPH:
            swe.set_sid_mode(swe.SIDM_LAHIRI)

    def get_julian_day(self, date_str):
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return swe.julday(dt.year, dt.month, dt.day, 0.0) if HAS_SWISSEPH else None

    def get_sun_moon_positions(self, date_str):
        if HAS_SWISSEPH:
            jd = self.get_julian_day(date_str)
            sun_pos, _ = swe.calc_ut(jd, swe.SUN)
            moon_pos, _ = swe.calc_ut(jd, swe.MOON)
            return sun_pos[0], moon_pos[0]
        else:
            base_date = datetime.datetime(2000, 1, 1)
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            days_diff = (target_date - base_date).days
            sun_lon = (280.46 + 0.985647 * days_diff) % 360
            moon_lon = (218.316 + 13.176396 * days_diff) % 360
            return sun_lon, moon_lon

    def get_moon_phase(self, sun_lon, moon_lon):
        return (moon_lon - sun_lon) % 360


class CryptoDataLoader:
    @staticmethod
    def fetch_data(symbol, exchange='binance', limit=1000):
        symbol = symbol.upper()
        filename = DATA_DIR / f"{exchange}_{symbol}_daily.csv"

        if filename.exists():
            print(f"📂 Loading cached data for {symbol}...")
            return pd.read_csv(filename, parse_dates=['date'])

        print(f"🌐 Downloading data for {symbol} from {exchange}...")
        try:
            if exchange == 'binance':
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
                response = requests.get(url, timeout=10)
                data = response.json()
                if not data:
                    raise ValueError("No data from Binance.")
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                df['date'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']].astype({
                    'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
                })
            elif exchange == 'bybit':
                url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=D&limit={limit}"
                response = requests.get(url, timeout=10)
                data = response.json()
                if data['retCode'] != 0:
                    raise ValueError(f"Bybit Error: {data['retMsg']}")
                rows = data['result']['list']
                df = pd.DataFrame(rows, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['date'] = pd.to_datetime(df['start'], unit='ms')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']].astype({
                    'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
                })
            else:
                raise ValueError("Unsupported exchange.")

            df.sort_values('date', inplace=True)
            df.to_csv(filename, index=False)
            print(f"💾 Saved to {filename}")
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            if filename.exists():
                return pd.read_csv(filename, parse_dates=['date'])
            return None


class AstrologicalAnalyzer:
    def __init__(self, df, astro_calc):
        self.df = df.reset_index(drop=True)
        self.astro_calc = astro_calc

    def _prepare_astro_features_gpu(self, dates):
        n = len(dates)
        sun_lons, moon_lons, phases = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, d in enumerate(dates):
            s, m = self.astro_calc.get_sun_moon_positions(d.strftime("%Y-%m-%d"))
            sun_lons[i], moon_lons[i] = s, m
            phases[i] = self.astro_calc.get_moon_phase(s, m)
        return (xp.asarray(sun_lons), xp.asarray(moon_lons), xp.asarray(phases)) if USE_GPU else (sun_lons, moon_lons, phases)

    def find_similar_patterns(self, target_idx, window=20, top_n=10, similarity_threshold=0.85):
        if target_idx >= len(self.df):
            return []
        target_date = self.df.iloc[target_idx]['date']
        ts, tm = self.astro_calc.get_sun_moon_positions(target_date.strftime("%Y-%m-%d"))
        tp = self.astro_calc.get_moon_phase(ts, tm)

        hist_dates = self.df.iloc[:target_idx]['date'].values
        if len(hist_dates) < window:
            return []

        hs, hm, hp = self._prepare_astro_features_gpu(hist_dates)
        t_s, t_m, t_p = xp.array([ts]), xp.array([tm]), xp.array([tp])

        def norm_diff(a, b):
            diff = (a - b) % 360
            return xp.where(diff > 180, diff - 360, diff)

        sun_diff = xp.abs(norm_diff(hs, t_s)) / 180.0
        moon_diff = xp.abs(norm_diff(hm, t_m)) / 180.0
        phase_diff = xp.abs(hp - t_p) / 180.0

        scores = 1.0 - (0.4 * sun_diff + 0.4 * moon_diff + 0.2 * phase_diff)
        mask = scores >= similarity_threshold
        valid_indices = xp.where(mask)[0]
        valid_scores = scores[mask]

        if len(valid_indices) == 0:
            return []

        if len(valid_indices) > top_n:
            sorted_idx = xp.argsort(-valid_scores)[:top_n]
            valid_indices, valid_scores = valid_indices[sorted_idx], valid_scores[sorted_idx]

        if USE_GPU:
            valid_indices, valid_scores = cp.asnumpy(valid_indices), cp.asnumpy(valid_scores)

        return [{'date': hist_dates[int(i)], 'score': float(s), 'index': int(i)} for i, s in zip(valid_indices, valid_scores)]

    def predict_direction(self, matches, lookforward=1):
        if not matches:
            return "NEUTRAL", 0.0
        bullish, bearish = 0, 0
        for m in matches:
            idx = m['index']
            if idx + lookforward < len(self.df):
                curr, future = self.df.iloc[idx]['close'], self.df.iloc[idx + lookforward]['close']
                if future > curr: bullish += 1
                elif future < curr: bearish += 1
        total = bullish + bearish
        if total == 0: return "NEUTRAL", 0.0
        prob = bullish / total
        if prob > 0.6: return "BULLISH", prob
        if prob < 0.4: return "BEARISH", 1 - prob
        return "NEUTRAL", 0.5

    def backtest(self, start_idx=None, end_idx=None, window=20, min_matches=5, tolerance=0.85):
        print("\n🔄 Starting Backtest..." + (" (GPU)" if USE_GPU else " (CPU)"))
        df_len = len(self.df)
        if start_idx is None: start_idx = window
        if end_idx is None: end_idx = df_len - 1

        correct, total, details = 0, 0, []
        for i in range(start_idx, min(end_idx, df_len - 1)):
            matches = self.find_similar_patterns(i, window=window, top_n=20, similarity_threshold=tolerance)
            if len(matches) < min_matches: continue

            pred_dir, conf = self.predict_direction(matches)
            actual = self.df.iloc[i+1]['close'] > self.df.iloc[i]['close']
            actual_dir = "BULLISH" if actual else "BEARISH"

            if pred_dir != "NEUTRAL":
                total += 1
                if pred_dir == actual_dir: correct += 1
                details.append({'date': str(self.df.iloc[i]['date']), 'pred': pred_dir, 'actual': actual_dir, 'res': 'WIN' if pred_dir == actual_dir else 'LOSS'})

        acc = (correct / total * 100) if total > 0 else 0
        print(f"✅ Backtest: {correct}/{total} ({acc:.2f}%)")
        return {'accuracy': acc, 'total': total, 'correct': correct, 'details': details}

    def optimize_parameters(self, candle_range=200):
        print("\n🔍 Optimizing Parameters..." + (" (GPU)" if USE_GPU else " (CPU)"))
        windows, thresholds, mins = [10, 20, 30], [0.80, 0.85, 0.90], [3, 5, 7]
        best_score, best_params, log = -1, {}, []

        for w, t, m in product(windows, thresholds, mins):
            res = self.backtest(start_idx=w, end_idx=min(len(self.df)-1, w+candle_range), window=w, min_matches=m, tolerance=t)
            score = res['accuracy'] * math.sqrt(res['total']) if res['total'] > 0 else 0
            log.append({'w': w, 't': t, 'm': m, 'acc': res['accuracy'], 'score': score})
            if score > best_score:
                best_score, best_params = score, {'window': w, 'threshold': t, 'min_matches': m, 'accuracy': res['accuracy']}

        print(f"🏆 Best: Win={best_params.get('window')} Thr={best_params.get('threshold')} Min={best_params.get('min_matches')} Acc={best_params.get('accuracy', 0):.2f}%")
        with open(RESULTS_DIR / "optimization_log.json", 'w') as f:
            json.dump({'best': best_params, 'log': log}, f, indent=2)
        return best_params


def main():
    parser = argparse.ArgumentParser(description="Astro Crypto Analyzer (GPU Ready)")
    parser.add_argument('--pairs', type=str, required=True, help="Pairs: BTCUSDT,ETHUSDT")
    parser.add_argument('--exchange', type=str, default='binance', choices=['binance', 'bybit'])
    parser.add_argument('--target-date', '-d', type=str, help="Target date YYYY-MM-DD")
    parser.add_argument('--backtest', action='store_true', help="Run backtest")
    parser.add_argument('--optimize', action='store_true', help="Optimize params")
    parser.add_argument('--candles', type=int, default=200, help="Candles for backtest/optimize")
    parser.add_argument('--tolerance', type=float, default=0.85, help="Similarity threshold")
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(',')]
    astro = AstroCalculator()

    for pair in pairs:
        print(f"\n{'='*20} {pair} {'='*20}")
        df = CryptoDataLoader.fetch_data(pair, args.exchange)
        if df is None or df.empty: continue

        analyzer = AstrologicalAnalyzer(df, astro)

        if args.optimize:
            analyzer.optimize_parameters(args.candles)
        elif args.backtest:
            res = analyzer.backtest(end_idx=min(len(df)-1, len(df)-args.candles), tolerance=args.tolerance)
            out = RESULTS_DIR / f"backtest_{pair}_{datetime.date.today()}.json"
            with open(out, 'w') as f: json.dump(res, f, indent=2)
            print(f"💾 Saved to {out}")
        else:
            idx = len(df) - 2
            if args.target_date:
                try:
                    td = pd.to_datetime(args.target_date)
                    match = df[df['date'] == td]
                    idx = match.index[0] if not match.empty else (df['date'] - td).abs().argsort()[0]
                except: pass
            
            if idx < 20: continue
            matches = analyzer.find_similar_patterns(idx, window=20, similarity_threshold=args.tolerance)
            direction, conf = analyzer.predict_direction(matches)
            print(f"🔮 Prediction for {df.iloc[idx]['date'].strftime('%Y-%m-%d')}: {direction} ({conf:.2%})")
            if matches: print(f"   Top Match: {matches[0]['date']} (Score: {matches[0]['score']:.2f})")


if __name__ == "__main__":
    main()
