#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astrological Trading Analyzer with GPU Acceleration
Supports: Binance/Bybit data loading, Astrological patterns, Backtesting, Optimization
GPU: NVIDIA CUDA (via CuPy) for fast pattern matching

Usage:
    python astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT
    python astro_trading_analyzer.py --pairs BTCUSDT --backtest --candles 200
    python astro_trading_analyzer.py --pairs BTCUSDT --optimize --mode basic
    python astro_trading_analyzer.py --pairs BTCUSDT --optimize --mode full
    python astro_trading_analyzer.py --pairs BTCUSDT --target-date 2024-06-15
"""

import argparse
import json
import os
import sys
import datetime
import math
import time
from itertools import product
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
HAS_SWISSEPHE = False
try:
    import swisseph as swe
    HAS_SWISSEPHE = True
    print("✅ swisseph loaded. Using high precision calculations.")
    swe.set_ephe_path(None)
except ImportError:
    HAS_SWISSEPHE = False
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
            # Конвертация даты в строку независимо от типа (numpy.datetime64 и др.)
            if hasattr(d, 'strftime'):
                date_str = d.strftime("%Y-%m-%d")
            else:
                date_str = str(d)[:10]
            
            s, m = self.astro_calc.get_sun_moon_positions(date_str)
            sun_lons[i], moon_lons[i] = s, m
            phases[i] = self.astro_calc.get_moon_phase(s, m)
        return (xp.asarray(sun_lons), xp.asarray(moon_lons), xp.asarray(phases)) if USE_GPU else (sun_lons, moon_lons, phases)

    def find_similar_patterns(self, target_idx, window=20, top_n=10, similarity_threshold=0.85):
        if target_idx >= len(self.df):
            return []
        target_date = self.df.iloc[target_idx]['date']
        
        # Конвертация даты в строку
        if hasattr(target_date, 'strftime'):
            target_date_str = target_date.strftime("%Y-%m-%d")
        else:
            target_date_str = str(target_date)[:10]
            
        ts, tm = self.astro_calc.get_sun_moon_positions(target_date_str)
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

    def backtest(self, start_idx=None, end_idx=None, window=20, min_matches=5, tolerance=0.85, detailed=False):
        """
        Расширенный бэктест с финансовой статистикой.
        """
        print("\n🔄 Starting Backtest..." + (" (GPU)" if USE_GPU else " (CPU)"))
        df_len = len(self.df)
        if start_idx is None: start_idx = window
        if end_idx is None: end_idx = df_len - 1

        correct, total = 0, 0
        wins, losses = [], []
        all_trades = []
        
        # Для расчета PnL нам нужно знать размер движения цены
        for i in range(start_idx, min(end_idx, df_len - 1)):
            matches = self.find_similar_patterns(i, window=window, top_n=20, similarity_threshold=tolerance)
            if len(matches) < min_matches: continue

            pred_dir, conf = self.predict_direction(matches)
            
            # Реальное движение
            close_curr = self.df.iloc[i]['close']
            close_next = self.df.iloc[i+1]['close']
            actual_move = close_next - close_curr
            actual_dir = "BULLISH" if actual_move > 0 else "BEARISH"
            
            # Если прогноз нейтральный - пропускаем
            if pred_dir == "NEUTRAL":
                continue

            total += 1
            
            # Расчет результата сделки (в пунктах цены)
            # Если купили (BULLISH) и цена выросла - прибыль
            # Если продали (BEARISH) и цена упала - прибыль
            pnl = 0.0
            if pred_dir == "BULLISH":
                pnl = actual_move
                if actual_move > 0:
                    correct += 1
                    wins.append({'date': str(self.df.iloc[i]['date']), 'pnl': pnl, 'conf': conf})
                else:
                    losses.append({'date': str(self.df.iloc[i]['date']), 'pnl': pnl, 'conf': conf})
            elif pred_dir == "BEARISH":
                pnl = -actual_move  # Инвертируем, так как шортим
                if actual_move < 0:
                    correct += 1
                    wins.append({'date': str(self.df.iloc[i]['date']), 'pnl': pnl, 'conf': conf})
                else:
                    losses.append({'date': str(self.df.iloc[i]['date']), 'pnl': pnl, 'conf': conf})
            
            all_trades.append({
                'date': str(self.df.iloc[i]['date']),
                'pred': pred_dir,
                'actual': actual_dir,
                'pnl': pnl,
                'res': 'WIN' if pnl > 0 else 'LOSS'
            })

        # Статистика
        acc = (correct / total * 100) if total > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in all_trades)
        gross_profit = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0))
        
        avg_win = (gross_profit / len(wins)) if wins else 0
        avg_loss = (gross_loss / len(losses)) if losses else 0
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Max Drawdown
        peak = 0
        max_dd = 0
        current_equity = 0
        for t in all_trades:
            current_equity += t['pnl']
            if current_equity > peak:
                peak = current_equity
            dd = peak - current_equity
            if dd > max_dd:
                max_dd = dd

        print(f"📊 Total Trades: {total}")
        print(f"🎯 Win Rate: {acc:.2f}% ({correct}/{total})")
        print(f"💰 Net PnL: {total_pnl:.4f} ({'+' if total_pnl > 0 else ''}{total_pnl:.2f}%)")
        print(f"📈 Gross Profit: {gross_profit:.4f}")
        print(f"📉 Gross Loss: {gross_loss:.4f}")
        print(f"📊 Avg Win: {avg_win:.4f} | Avg Loss: {avg_loss:.4f}")
        print(f"🏆 Profit Factor: {profit_factor:.2f}")
        print(f"⚠️  Max Drawdown: {max_dd:.4f}")
        
        if losses:
            sorted_losses = sorted(losses, key=lambda x: x['pnl'])[:5]
            print("\n❌ Top 5 Worst Trades:")
            for l in sorted_losses:
                print(f"   {l['date']}: {l['pnl']:.4f} (Conf: {l['conf']:.2f})")

        # Рекомендации
        print("\n💡 Recommendations:")
        if profit_factor < 1.0:
            print("   ⚠️ Strategy is losing money. Consider tightening similarity threshold or increasing min_matches.")
        elif profit_factor < 1.5:
            print("   ⚡ Strategy is marginal. Try optimizing parameters further.")
        else:
            print("   ✅ Strategy looks profitable. Consider live testing with small size.")
            
        if max_dd > gross_profit * 0.5:
            print("   ⚠️ High drawdown relative to profit. Consider reducing position size or filtering low confidence signals.")
            
        if len(wins) > 0 and len(losses) > 0:
            win_rate_by_conf = {}
            # Группировка по уверенности (примерно)
            for w in wins:
                bucket = int(w['conf'] * 10) * 0.1
                if bucket not in win_rate_by_conf: win_rate_by_conf[bucket] = {'w':0, 'l':0}
                win_rate_by_conf[bucket]['w'] += 1
            for l in losses:
                bucket = int(l['conf'] * 10) * 0.1
                if bucket not in win_rate_by_conf: win_rate_by_conf[bucket] = {'w':0, 'l':0}
                win_rate_by_conf[bucket]['l'] += 1
            
            best_conf = 0
            best_ratio = 0
            for k, v in win_rate_by_conf.items():
                if v['w'] + v['l'] > 2: # Минимум 3 сделки
                    ratio = v['w'] / (v['w'] + v['l'])
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_conf = k
            if best_conf > 0:
                print(f"   🎯 Best results with confidence > {best_conf:.1f} (Win rate: {best_ratio*100:.1f}%)")

        result = {
            'accuracy': acc, 
            'total': total, 
            'correct': correct, 
            'net_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'trades': all_trades
        }
        
        if detailed:
            return result
        else:
            # Для оптимизации возвращаем только ключевые метрики
            return {'accuracy': acc, 'total': total, 'correct': correct, 'score': total_pnl}

    def optimize_parameters(self, candle_range=200, mode='basic'):
        """
        Оптимизация параметров.
        mode='basic': перебор window, threshold, min_matches с фиксированными весами.
        mode='full': дополнительный перебор весов астрологических факторов.
        """
        print(f"\n🔍 Optimizing Parameters... (Mode: {mode})" + (" (GPU)" if USE_GPU else " (CPU)"))
        
        windows = [10, 20, 30, 50]
        thresholds = [0.80, 0.85, 0.90, 0.95]
        mins = [3, 5, 7, 10]
        
        # Наборы весов для полного режима
        weight_sets = [None]  # Default weights
        if mode == 'full':
            print("⚠️  Full mode: Testing different weight configurations (this will take a while)...")
            weight_sets = [
                None,  # Default: phase=3, sign=2, illum=2, aspect=2, venus=1, mars=1
                {'moon_phase': 5.0, 'moon_sign': 1.0, 'illumination': 1.0, 'sun_moon_aspect': 1.0, 'venus_sign': 1.0, 'mars_sign': 1.0},
                {'moon_phase': 1.0, 'moon_sign': 3.0, 'illumination': 3.0, 'sun_moon_aspect': 1.0, 'venus_sign': 1.0, 'mars_sign': 1.0},
                {'moon_phase': 2.0, 'moon_sign': 2.0, 'illumination': 2.0, 'sun_moon_aspect': 3.0, 'venus_sign': 1.0, 'mars_sign': 1.0},
                {'moon_phase': 1.0, 'moon_sign': 1.0, 'illumination': 1.0, 'sun_moon_aspect': 1.0, 'venus_sign': 3.0, 'mars_sign': 3.0}
            ]
        
        best_score, best_params, log = -float('inf'), {}, []
        total_combos = len(windows) * len(thresholds) * len(mins) * len(weight_sets)
        current_combo = 0
        
        if mode == 'basic':
            print(f"📊 Testing {total_combos} combinations (Basic Mode)...")
        else:
            print(f"📊 Testing {total_combos} combinations (Full Mode with weight variations)...")

        for w, t, m in product(windows, thresholds, mins):
            for ws in weight_sets:
                current_combo += 1
                print(f"\rProgress: {current_combo}/{total_combos}", end="", flush=True)
                
                # Временно меняем веса если нужно
                original_weights = None
                if ws:
                    original_weights = self.weights.copy()
                    self.weights = ws
                
                try:
                    res = self.backtest(
                        start_idx=w, 
                        end_idx=min(len(self.df)-1, w+candle_range), 
                        window=w, 
                        min_matches=m, 
                        tolerance=t,
                        detailed=True  # Получаем полную статистику
                    )
                except Exception as e:
                    if original_weights:
                        self.weights = original_weights
                    continue
                
                # Восстанавливаем веса
                if original_weights:
                    self.weights = original_weights
                
                # Скоринг: PnL - штраф за просадку + бонус за PF
                pnl = res.get('net_pnl', 0)
                dd = res.get('max_drawdown', 0)
                pf = res.get('profit_factor', 0)
                trades = res.get('total', 0)
                
                if trades < 5:  # Пропускаем если мало сделок
                    continue
                    
                score = pnl - (dd * 0.5) + (pf * 10)
                
                log.append({
                    'w': w, 't': t, 'm': m, 
                    'weights': 'custom' if ws else 'default',
                    'pnl': pnl, 'dd': dd, 'pf': pf,
                    'acc': res['accuracy'], 'total': trades,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'window': w, 
                        'threshold': t, 
                        'min_matches': m,
                        'weights': ws,
                        'pnl': pnl,
                        'drawdown': dd,
                        'profit_factor': pf,
                        'accuracy': res['accuracy'],
                        'total_trades': trades
                    }

        print("\n\n" + "="*50)
        print("🏆 OPTIMIZATION COMPLETE")
        print("="*50)
        
        if not best_params:
            print("❌ No profitable configuration found.")
            return
            
        wp = best_params.get('weights')
        w_str = "Default" if wp is None else "Custom"
        
        print(f"Best Configuration:")
        print(f"   Window: {best_params['window']}")
        print(f"   Threshold: {best_params['threshold']}")
        print(f"   Min Matches: {best_params['min_matches']}")
        print(f"   Weights: {w_str}")
        print(f"\n📈 Results:")
        print(f"   Net PnL: {best_params['pnl']:.4f}")
        print(f"   Max Drawdown: {best_params['drawdown']:.4f}")
        print(f"   Profit Factor: {best_params['profit_factor']:.2f}")
        print(f"   Win Rate: {best_params['accuracy']:.2f}%")
        print(f"   Total Trades: {best_params['total_trades']}")
        
        # Рекомендации
        print("\n💡 Recommendations:")
        if best_params['profit_factor'] < 1.2:
            print("   ⚠️ Profit Factor is low. Try increasing 'tolerance' to filter weak patterns.")
        if best_params['drawdown'] > abs(best_params['pnl']) * 0.5 and best_params['pnl'] > 0:
            print("   ⚠️ High drawdown relative to profit. Reduce 'window' size or increase 'min_matches'.")
        if best_params['accuracy'] < 45:
            print("   ⚠️ Low win rate. The strategy might be counter-trend. Check aspect weights.")
        if best_params['total_trades'] < 20:
            print("   ⚠️ Too few trades. Decrease 'min_matches' or 'tolerance' to find more patterns.")
        if best_params['profit_factor'] > 1.5:
            print("   ✅ Strategy looks profitable! Consider live testing with small position size.")
            
        # Сохранение полного лога
        log_data = {
            'best_params': best_params,
            'top_10_runs': sorted(log, key=lambda x: x['score'], reverse=True)[:10],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        out_file = RESULTS_DIR / f"optimization_{datetime.date.today()}.json"
        with open(out_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
        print(f"\n💾 Detailed results saved to: {out_file}")
        return best_params


def main():
    parser = argparse.ArgumentParser(description="Astro Crypto Analyzer (GPU Ready)")
    parser.add_argument('--pairs', type=str, required=True, help="Pairs: BTCUSDT,ETHUSDT")
    parser.add_argument('--exchange', type=str, default='binance', choices=['binance', 'bybit'])
    parser.add_argument('--target-date', '-d', type=str, help="Target date YYYY-MM-DD")
    parser.add_argument('--backtest', action='store_true', help="Run backtest")
    parser.add_argument('--optimize', action='store_true', help="Optimize params")
    parser.add_argument('--mode', type=str, default='basic', choices=['basic', 'full'], 
                        help="Optimization mode: basic (fast) or full (slow, tests weights)")
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
            analyzer.optimize_parameters(args.candles, mode=args.mode)
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
