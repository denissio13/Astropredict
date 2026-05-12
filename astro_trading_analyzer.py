#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astrological Trading Analyzer with Auto Data Fetching
Analyzes historical price data based on astrological patterns (Moon phases, planetary positions)
to forecast the direction of the current daily candle for Crypto pairs.
Fetches data automatically from Binance or Bybit.

Usage:
    python astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT --source binance
    python astro_trading_analyzer.py --pairs SOLUSDT --source bybit --similarity 0.85
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import optional heavy libraries
try:
    import swisseph as swe
    HAS_SWISSEPH = True
except ImportError:
    HAS_SWISSEPH = False
    print("⚠️  Warning: swisseph not installed. Using simplified astronomical calculations.")
    print("   For better accuracy, install it: pip install pyswisseph")

import pandas as pd
import numpy as np
import requests
import time

# --- Configuration ---
DEFAULT_DATA_DIR = "data"
CACHE_EXPIRY_HOURS = 24

# ============================================================================
# DATA FETCHING MODULE
# ============================================================================

def fetch_binance_data(symbol: str, timeframe: str = '1d', limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical K-line data from Binance Public API.
    Symbol format: 'BTCUSDT', 'ETHUSDT' (case insensitive).
    """
    symbol = symbol.upper().replace('/', '').replace('-', '')
    
    # Auto-append USDT if no quote currency detected
    if not any(quote in symbol for quote in ['USDT', 'BUSD', 'USDC', 'FDUSD', 'BTC', 'ETH', 'BNB', 'SOL']):
        symbol += 'USDT'
    
    url = "https://api.binance.com/api/v3/klines"
    
    print(f"📡 Fetching data for {symbol} from Binance...")
    
    params = {
        'symbol': symbol,
        'interval': timeframe,
        'limit': limit
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if not data or not isinstance(data, list):
            raise ValueError(f"No data returned for {symbol} from Binance.")
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        print(f"✅ Loaded {len(df)} candles for {symbol} (Range: {df.index[0].date()} to {df.index[-1].date()})")
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching {symbol} from Binance: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error processing {symbol} from Binance: {e}")
        return pd.DataFrame()


def fetch_bybit_data(symbol: str, timeframe: str = 'D', limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical K-line data from Bybit Public API (V5).
    Symbol format: 'BTCUSDT'
    """
    symbol = symbol.upper().replace('/', '').replace('-', '')
    
    if not any(quote in symbol for quote in ['USDT', 'USDC', 'BTC', 'ETH', 'SOL']):
        symbol += 'USDT'

    url = "https://api.bybit.com/v5/market/kline"
    
    print(f"📡 Fetching data for {symbol} from Bybit...")
    
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': timeframe,
        'limit': limit
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        
        if result.get('retCode') != 0:
            raise ValueError(f"Bybit API error: {result.get('retMsg', 'Unknown error')}")
            
        data = result['result']['list']
        
        if not data:
            raise ValueError(f"No data returned for {symbol} from Bybit.")
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        print(f"✅ Loaded {len(df)} candles for {symbol} (Range: {df.index[0].date()} to {df.index[-1].date()})")
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching {symbol} from Bybit: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error processing {symbol} from Bybit: {e}")
        return pd.DataFrame()


def get_market_data(pairs: List[str], source: str = 'binance') -> Dict[str, pd.DataFrame]:
    """
    Orchestrates data fetching for a list of pairs.
    Returns a dictionary mapping symbol -> DataFrame
    Данные сохраняются в папку market_data/<source>_<symbol>_daily.csv
    """
    data_store = {}
    source = source.lower()
    
    if source not in ['binance', 'bybit']:
        print(f"⚠️  Unknown source '{source}'. Defaulting to Binance.")
        source = 'binance'
    
    # Создаем директорию для кэширования данных
    cache_dir = Path("market_data")
    cache_dir.mkdir(exist_ok=True)
    
    for pair in pairs:
        try:
            filename = f"{source}_{pair.upper()}_daily.csv"
            filepath = cache_dir / filename
            
            # Проверяем кэш
            if filepath.exists():
                print(f"[INFO] Загрузка данных из кэша: {filepath}")
                df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            else:
                # Скачиваем новые данные
                if source == 'binance':
                    df = fetch_binance_data(pair)
                elif source == 'bybit':
                    df = fetch_bybit_data(pair)
                
                # Сохраняем в кэш если данные получены успешно
                if not df.empty:
                    df.to_csv(filepath)
                    print(f"[INFO] Данные сохранены в: {filepath}")
            
            if not df.empty:
                data_store[pair.upper()] = df
            else:
                print(f"⚠️  Skipping {pair} due to empty data.")
                
        except Exception as e:
            print(f"❌ Error fetching data for {pair}: {e}")
            print(f"⚠️  Skipping {pair}.")
            
        # Small delay to be polite to APIs if many pairs
        if len(pairs) > 1:
            time.sleep(0.2)
            
    return data_store

# ============================================================================
# ASTROLOGICAL CALCULATION ENGINE
# ============================================================================

class AstroEngine:
    """Handles all astronomical/astrological calculations"""
    
    def __init__(self):
        if HAS_SWISSEPH:
            swe.set_ephe_path(None)  # Use default built-in ephemeris
            
    def get_julian_day(self, dt: datetime) -> float:
        """Convert datetime to Julian Day"""
        if HAS_SWISSEPH:
            return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60.0)
        else:
            # Simplified Julian Day calculation (sufficient for our approximations)
            # Based on Meeus algorithm
            year = dt.year
            month = dt.month
            day = dt.day + (dt.hour + dt.minute/60.0) / 24.0
            
            if month <= 2:
                year -= 1
                month += 12
                
            A = int(year / 100)
            B = 2 - A + int(A / 4)
            
            jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
            return jd
    
    def get_planet_position(self, dt: datetime, planet_id: int) -> float:
        """
        Get ecliptic longitude of a planet in degrees [0-360]
        Uses Swiss Ephemeris if available, otherwise simplified calculation
        """
        jd = self.get_julian_day(dt)
        
        if HAS_SWISSEPH:
            # Flag for mean node vs true node etc can be added, using default for now
            result, ret_flag = swe.calc_ut(jd, planet_id)
            return result[0]  # Longitude
        else:
            # Simplified approximation (not accurate for trading, but functional for demo)
            return self._simplified_planet_pos(dt, planet_id)
    
    def _simplified_planet_pos(self, dt: datetime, planet_id: int) -> float:
        """Very rough approximation for when swisseph is missing"""
        # Reference date: J2000.0
        ref_date = datetime(2000, 1, 1, 12, 0)
        days_diff = (dt - ref_date).total_seconds() / 86400.0
        
        # Approximate orbital periods (days) and reference longitudes
        # These are very rough averages
        planets_data = {
            0: {'period': 365.25, 'offset': 280.46},   # Sun
            1: {'period': 27.32, 'offset': 218.32},    # Moon
            2: {'period': 87.97, 'offset': 252.25},    # Mercury
            3: {'period': 224.70, 'offset': 181.98},   # Venus
            4: {'period': 686.98, 'offset': 355.43},   # Mars
            5: {'period': 4332.59, 'offset': 34.35},   # Jupiter
            6: {'period': 10759.22, 'offset': 50.07},  # Saturn
        }
        
        if planet_id not in planets_data:
            return 0.0
            
        p = planets_data[planet_id]
        angle = (p['offset'] + (360.0 * days_diff / p['period'])) % 360.0
        return angle

    def get_moon_phase(self, dt: datetime) -> Tuple[float, str]:
        """
        Calculate moon phase (0.0-1.0) and name
        0.0 = New Moon, 0.5 = Full Moon
        """
        if HAS_SWISSEPH:
            jd = self.get_julian_day(dt)
            sun_lon = swe.calc_ut(jd, 0)[0][0]
            moon_lon = swe.calc_ut(jd, 1)[0][0]
            elongation = (moon_lon - sun_lon) % 360.0
            phase = elongation / 360.0
        else:
            # Simplified but more accurate lunar phase calculation
            phase = self._simplified_moon_phase(dt)
            
        # Determine phase name
        if phase < 0.03 or phase > 0.97:
            name = "New Moon"
        elif phase < 0.22:
            name = "Waxing Crescent"
        elif phase < 0.28:
            name = "First Quarter"
        elif phase < 0.47:
            name = "Waxing Gibbous"
        elif phase < 0.53:
            name = "Full Moon"
        elif phase < 0.72:
            name = "Waning Gibbous"
        elif phase < 0.78:
            name = "Last Quarter"
        elif phase < 0.97:
            name = "Waning Crescent"
        else:
            name = "New Moon"
            
        return phase, name

    def _simplified_moon_phase(self, dt: datetime) -> float:
        """Rough moon phase calculation using known synodic cycles"""
        # Reference: Known new moon on Jan 6, 2000 at 18:14 UTC
        ref_date = datetime(2000, 1, 6, 18, 14)
        diff_days = (dt - ref_date).total_seconds() / 86400.0
        synodic_month = 29.53058867  # Average synodic month length
        phase = (diff_days % synodic_month) / synodic_month
        return phase

    def get_aspects(self, dt: datetime) -> List[Dict]:
        """
        Calculate major aspects between planets for a given date
        Returns list of aspects: {planet1, planet2, angle, orb, type}
        """
        planet_names = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
        planet_ids = [0, 1, 2, 3, 4, 5, 6]
        
        positions = {}
        for name, pid in zip(planet_names, planet_ids):
            positions[name] = self.get_planet_position(dt, pid)
            
        aspects = []
        major_aspects = [
            {'name': 'Conjunction', 'angle': 0, 'orb': 8},
            {'name': 'Opposition', 'angle': 180, 'orb': 8},
            {'name': 'Trine', 'angle': 120, 'orb': 8},
            {'name': 'Square', 'angle': 90, 'orb': 7},
            {'name': 'Sextile', 'angle': 60, 'orb': 6},
        ]
        
        for i, p1 in enumerate(planet_names):
            for j, p2 in enumerate(planet_names):
                if j <= i:
                    continue
                    
                diff = abs(positions[p1] - positions[p2])
                if diff > 180:
                    diff = 360 - diff
                    
                for asp in major_aspects:
                    orb = abs(diff - asp['angle'])
                    if orb <= asp['orb']:
                        aspects.append({
                            'planet1': p1,
                            'planet2': p2,
                            'type': asp['name'],
                            'angle': asp['angle'],
                            'orb': round(orb, 2),
                            'exactness': 1.0 - (orb / asp['orb'])
                        })
                        
        return aspects

    def generate_astro_signature(self, dt: datetime) -> Dict:
        """
        Generate a complete astrological signature for a date
        Used for pattern matching
        """
        phase_val, phase_name = self.get_moon_phase(dt)
        aspects = self.get_aspects(dt)
        
        # Get positions
        positions = {}
        planet_names = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
        for i, name in enumerate(planet_names):
            positions[name] = round(self.get_planet_position(dt, i), 2)
            
        # Discretize positions into sectors (30° = 1 zodiac sign) for pattern matching
        sectors = {}
        for name, pos in positions.items():
            sectors[f"{name}_sector"] = int(pos // 30)
            
        return {
            'date': dt,
            'moon_phase_val': phase_val,
            'moon_phase_name': phase_name,
            'moon_sector': int(positions['Moon'] // 30),
            'sun_sector': int(positions['Sun'] // 30),
            'sectors': sectors,
            'positions': positions,
            'aspects': aspects,
            'aspect_count': len(aspects),
            'dominant_aspect': aspects[0]['type'] if aspects else None
        }

# ============================================================================
# PATTERN ANALYSIS & FORECASTING
# ============================================================================

class PatternAnalyzer:
    """Analyzes patterns and generates forecasts"""
    
    def __init__(self, astro_engine: AstroEngine):
        self.astro = astro_engine
        
    def calculate_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """
        Calculate similarity score between two astrological signatures
        Score ranges from 0.0 (completely different) to 1.0 (identical)
        """
        score = 0.0
        max_score = 0.0
        
        # Moon phase similarity (weight: 30%)
        phase_diff = abs(sig1['moon_phase_val'] - sig2['moon_phase_val'])
        if phase_diff > 0.5:
            phase_diff = 1.0 - phase_diff
        phase_score = 1.0 - phase_diff
        score += phase_score * 0.30
        max_score += 0.30
        
        # Moon sector similarity (weight: 20%)
        if sig1['moon_sector'] == sig2['moon_sector']:
            score += 0.20
        else:
            # Adjacent sectors get partial credit
            diff = abs(sig1['moon_sector'] - sig2['moon_sector'])
            if diff == 1 or diff == 11:
                score += 0.10
        max_score += 0.20
        
        # Sun sector similarity (weight: 15%)
        if sig1['sun_sector'] == sig2['sun_sector']:
            score += 0.15
        max_score += 0.15
        
        # Aspect pattern similarity (weight: 25%)
        # Compare count and types
        count_diff = abs(sig1['aspect_count'] - sig2['aspect_count'])
        count_score = max(0, 1.0 - (count_diff * 0.2))
        score += count_score * 0.15
        max_score += 0.15
        
        # Check for same dominant aspect
        if sig1['dominant_aspect'] and sig2['dominant_aspect']:
            if sig1['dominant_aspect'] == sig2['dominant_aspect']:
                score += 0.10
        max_score += 0.10
        
        # Normalize to 0-1
        if max_score > 0:
            return score / max_score
        return 0.0
    
    def find_similar_patterns(self, df: pd.DataFrame, target_date: datetime, 
                              min_similarity: float = 0.7, max_matches: int = 50) -> List[Dict]:
        """
        Find historical dates with similar astrological signatures
        """
        target_sig = self.astro.generate_astro_signature(target_date)
        matches = []
        
        # Iterate through historical data (exclude last few days to avoid lookahead bias)
        cutoff_date = target_date - timedelta(days=2)
        
        for idx, row in df[df.index < cutoff_date].iterrows():
            hist_date = idx.to_pydatetime()
            hist_sig = self.astro.generate_astro_signature(hist_date)
            
            similarity = self.calculate_similarity(target_sig, hist_sig)
            
            if similarity >= min_similarity:
                # Calculate price movement for the "current" candle (the day itself)
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']
                low_price = row['low']
                
                if close_price > open_price:
                    direction = 'BULLISH'
                    strength = (close_price - open_price) / open_price
                elif close_price < open_price:
                    direction = 'BEARISH'
                    strength = (open_price - close_price) / open_price
                else:
                    direction = 'NEUTRAL'
                    strength = 0.0
                    
                matches.append({
                    'date': hist_date,
                    'similarity': similarity,
                    'direction': direction,
                    'strength': strength,
                    'open': open_price,
                    'close': close_price,
                    'high': high_price,
                    'low': low_price,
                    'astro_sig': hist_sig
                })
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:max_matches]
    
    def generate_forecast(self, matches: List[Dict]) -> Dict:
        """
        Generate forecast based on matched patterns
        """
        if not matches:
            return {
                'direction': 'UNKNOWN',
                'confidence': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_matches': 0,
                'avg_strength': 0.0,
                'message': 'No similar patterns found'
            }
        
        bullish = sum(1 for m in matches if m['direction'] == 'BULLISH')
        bearish = sum(1 for m in matches if m['direction'] == 'BEARISH')
        neutral = sum(1 for m in matches if m['direction'] == 'NEUTRAL')
        total = len(matches)
        
        # Weighted voting by similarity
        bullish_weight = sum(m['similarity'] for m in matches if m['direction'] == 'BULLISH')
        bearish_weight = sum(m['similarity'] for m in matches if m['direction'] == 'BEARISH')
        
        total_weight = bullish_weight + bearish_weight
        if total_weight > 0:
            bullish_ratio = bullish_weight / total_weight
        else:
            bullish_ratio = 0.5
            
        avg_strength = np.mean([m['strength'] for m in matches])
        
        # Determine direction
        if bullish > bearish * 1.2:  # 20% threshold for clarity
            direction = 'BULLISH'
        elif bearish > bullish * 1.2:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL/UNCERTAIN'
            
        # Confidence based on consensus and sample size
        base_confidence = abs(bullish - bearish) / total
        sample_factor = min(1.0, total / 20.0)  # Max out at 20 samples
        confidence = base_confidence * sample_factor
        
        return {
            'direction': direction,
            'confidence': round(confidence, 3),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'total_matches': total,
            'avg_strength': round(avg_strength, 4),
            'bullish_ratio': round(bullish_ratio, 3),
            'top_matches': matches[:5]  # Include top 5 for reference
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Astrological Crypto Trading Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --pairs BTCUSDT
  python %(prog)s --pairs ETHUSDT,SOLUSDT,BNBUSDT --source bybit
  python %(prog)s --pairs BTCUSDT --similarity 0.8 --max-matches 30
        """
    )
    
    parser.add_argument(
        '--pairs', '-p',
        type=str,
        required=True,
        help='Crypto pair(s) to analyze, comma-separated (e.g., BTCUSDT,ETHUSDT)'
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='binance',
        choices=['binance', 'bybit'],
        help='Data source (default: binance)'
    )
    
    parser.add_argument(
        '--similarity',
        type=float,
        default=0.65,
        help='Minimum similarity threshold for pattern matching (0.0-1.0, default: 0.65)'
    )
    
    parser.add_argument(
        '--max-matches', '-m',
        type=int,
        default=50,
        help='Maximum number of historical matches to consider (default: 50)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='astro_analysis_result.json',
        help='Output JSON file path (default: astro_analysis_result.json)'
    )
    
    args = parser.parse_args()
    
    # Parse pairs
    pairs = [p.strip().upper() for p in args.pairs.split(',')]
    pairs = [p for p in pairs if p]  # Remove empty strings
    
    if not pairs:
        print("❌ Error: No valid pairs provided!")
        sys.exit(1)
    
    print("=" * 60)
    print("🔮 ASTROLOGICAL CRYPTO TRADING ANALYZER")
    print("=" * 60)
    print(f"📊 Pairs: {', '.join(pairs)}")
    print(f"📡 Source: {args.source.upper()}")
    print(f"🎯 Similarity Threshold: {args.similarity}")
    print(f"📈 Max Matches: {args.max_matches}")
    print("=" * 60)
    
    # Initialize components
    astro_engine = AstroEngine()
    analyzer = PatternAnalyzer(astro_engine)
    
    # Fetch data
    print("\n⏳ Fetching market data...")
    data_store = get_market_data(pairs, source=args.source)
    
    if not data_store:
        print("❌ No data could be fetched. Exiting.")
        sys.exit(1)
    
    results = {}
    
    # Analyze each pair
    for symbol, df in data_store.items():
        print(f"\n{'='*60}")
        print(f"🪐 ANALYZING {symbol}")
        print(f"{'='*60}")
        
        # Get current date (last date in dataset)
        current_date = df.index[-1]
        print(f"📅 Analysis Date: {current_date.date()}")
        
        # Generate current astro signature
        current_sig = astro_engine.generate_astro_signature(current_date)
        print(f"🌙 Moon Phase: {current_sig['moon_phase_name']} ({current_sig['moon_phase_val']:.2f})")
        print(f"☀️  Sun Position: {current_sig['positions']['Sun']:.1f}°")
        print(f"🌙 Moon Position: {current_sig['positions']['Moon']:.1f}°")
        print(f"🔗 Active Aspects: {current_sig['aspect_count']}")
        if current_sig['dominant_aspect']:
            print(f"   Dominant: {current_sig['dominant_aspect']}")
        
        # Find similar patterns
        print(f"\n🔍 Searching for historical patterns (similarity ≥ {args.similarity})...")
        matches = analyzer.find_similar_patterns(
            df, 
            current_date, 
            min_similarity=args.similarity,
            max_matches=args.max_matches
        )
        
        print(f"✅ Found {len(matches)} similar historical patterns")
        
        # Generate forecast
        forecast = analyzer.generate_forecast(matches)
        
        # Display results
        print(f"\n{'─'*40}")
        print("📊 FORECAST RESULTS")
        print(f"{'─'*40}")
        print(f"Direction:      {forecast['direction']}")
        print(f"Confidence:     {forecast['confidence']*100:.1f}%")
        print(f"Bullish Cases:  {forecast['bullish_count']}")
        print(f"Bearish Cases:  {forecast['bearish_count']}")
        print(f"Neutral Cases:  {forecast['neutral_count']}")
        print(f"Avg Strength:   {forecast['avg_strength']*100:.2f}%")
        
        if forecast['total_matches'] > 0:
            print(f"\n🏆 Top 3 Historical Matches:")
            for i, match in enumerate(forecast.get('top_matches', [])[:3], 1):
                print(f"   {i}. {match['date'].date()} (Similarity: {match['similarity']:.2f}, Result: {match['direction']})")
        
        # Store result
        results[symbol] = {
            'analysis_date': current_date.isoformat(),
            'astro_signature': {
                'moon_phase': current_sig['moon_phase_name'],
                'moon_phase_value': current_sig['moon_phase_val'],
                'sun_position': current_sig['positions']['Sun'],
                'moon_position': current_sig['positions']['Moon'],
                'aspect_count': current_sig['aspect_count'],
                'dominant_aspect': current_sig['dominant_aspect']
            },
            'forecast': forecast,
            'matches_found': len(matches)
        }
    
    # Save results to JSON
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'pairs': pairs,
            'source': args.source,
            'similarity_threshold': args.similarity,
            'max_matches': args.max_matches
        },
        'results': results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {args.output}")
    print("=" * 60)
    print("✨ Analysis Complete!")
    print("=" * 60)
    
    # Summary table
    print("\n📋 SUMMARY:")
    print(f"{'Pair':<12} {'Direction':<18} {'Confidence':<12} {'Matches':<8}")
    print("-" * 50)
    for symbol, res in results.items():
        fc = res['forecast']
        print(f"{symbol:<12} {fc['direction']:<18} {fc['confidence']*100:>6.1f}%      {fc['total_matches']:<8}")


if __name__ == '__main__':
    main()
