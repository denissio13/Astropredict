#!/usr/bin/env python3
"""
Астрологический анализ торговых паттернов
Сопоставляет положения планет и лунный календарь с графиком цены
для прогнозирования направления дневной свечи
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import math

# Для расчёта положений планет (требуется установка: pip install swisseph)
try:
    import swisseph as swe
    SWISSEPHE_AVAILABLE = True
except ImportError:
    SWISSEPHE_AVAILABLE = False
    print("Warning: swisseph not installed. Using simplified planetary calculations.")


class AstrologicalAnalyzer:
    """Класс для астрологического анализа рыночных данных"""
    
    def __init__(self, ephemeris_path: str = None):
        """
        Инициализация анализатора
        
        Args:
            ephemeris_path: Путь к файлам эфемерид (для swisseph)
        """
        if SWISSEPHE_AVAILABLE and ephemeris_path:
            swe.set_ephe_path(ephemeris_path)
        
        # Планеты для анализа
        self.planets = {
            'Sun': 0,      # Солнце
            'Moon': 1,     # Луна
            'Mercury': 2,  # Меркурий
            'Venus': 3,    # Венера
            'Mars': 4,     # Марс
            'Jupiter': 5,  # Юпитер
            'Saturn': 6,   # Сатурн
        }
        
        # Лунные фазы
        self.moon_phases = {
            'New Moon': 0,           # Новолуние
            'Waxing Crescent': 1,    # Растущая луна
            'First Quarter': 2,      # Первая четверть
            'Waxing Gibbous': 3,     # Прибывающая луна
            'Full Moon': 4,          # Полнолуние
            'Waning Gibbous': 5,     # Убывающая луна
            'Last Quarter': 6,       # Последняя четверть
            'Waning Crescent': 7,    # Убывающий серп
        }
    
    def get_planet_position(self, date: datetime, planet_name: str) -> float:
        """
        Получить положение планеты на заданную дату
        
        Args:
            date: Дата для расчёта
            planet_name: Название планеты
            
        Returns:
            Долгота планеты в градусах (0-360)
        """
        if SWISSEPHE_AVAILABLE:
            julian_day = swe.julday(date.year, date.month, date.day, 0.0)
            planet_id = self.planets.get(planet_name, 0)
            result = swe.calc_ut(julian_day, planet_id)
            return result[0][0]  # Долгота
        else:
            # Упрощённый расчёт (для демонстрации)
            return self._simplified_planet_position(date, planet_name)
    
    def _simplified_planet_position(self, date: datetime, planet_name: str) -> float:
        """Упрощённый расчёт положения планет (если swisseph недоступен)"""
        # Базовые периоды обращения планет (в днях)
        periods = {
            'Sun': 365.25,
            'Moon': 27.32,
            'Mercury': 87.97,
            'Venus': 224.70,
            'Mars': 686.98,
            'Jupiter': 4332.59,
            'Saturn': 10759.22,
        }
        
        # Базовые позиции на 01.01.2000
        base_positions = {
            'Sun': 280.5,
            'Moon': 218.3,
            'Mercury': 261.8,
            'Venus': 185.4,
            'Mars': 113.2,
            'Jupiter': 238.7,
            'Saturn': 50.1,
        }
        
        base_date = datetime(2000, 1, 1)
        days_since_base = (date - base_date).days
        
        period = periods.get(planet_name, 365.25)
        base_pos = base_positions.get(planet_name, 0)
        
        position = (base_pos + (days_since_base / period) * 360) % 360
        return position
    
    def get_moon_phase(self, date: datetime) -> Tuple[str, float]:
        """
        Определить фазу луны и её освещённость
        
        Args:
            date: Дата для расчёта
            
        Returns:
            Кортеж (название фазы, процент освещённости)
        """
        # Получаем долготы Солнца и Луны
        sun_lon = self.get_planet_position(date, 'Sun')
        moon_lon = self.get_planet_position(date, 'Moon')
        
        # Угол между Солнцем и Луной
        angle = (moon_lon - sun_lon) % 360
        
        # Процент освещённости
        illumination = (1 - math.cos(math.radians(angle))) / 2 * 100
        
        # Определение фазы
        if angle < 22.5:
            phase = 'New Moon'
        elif angle < 67.5:
            phase = 'Waxing Crescent'
        elif angle < 112.5:
            phase = 'First Quarter'
        elif angle < 157.5:
            phase = 'Waxing Gibbous'
        elif angle < 202.5:
            phase = 'Full Moon'
        elif angle < 247.5:
            phase = 'Waning Gibbous'
        elif angle < 292.5:
            phase = 'Last Quarter'
        else:
            phase = 'Waning Crescent'
        
        return phase, illumination
    
    def get_astrological_signature(self, date: datetime) -> Dict:
        """
        Создать астрологическую сигнатуру для даты
        
        Args:
            date: Дата для анализа
            
        Returns:
            Словарь с астрологическими параметрами
        """
        signature = {}
        
        # Положение Луны
        signature['moon_longitude'] = self.get_planet_position(date, 'Moon')
        signature['moon_sign'] = int(signature['moon_longitude'] // 30)  # Знак зодиака (0-11)
        
        # Фаза луны
        phase, illumination = self.get_moon_phase(date)
        signature['moon_phase'] = phase
        signature['moon_illumination'] = illumination
        signature['moon_phase_id'] = self.moon_phases[phase]
        
        # Положения планет
        for planet in self.planets.keys():
            signature[f'{planet.lower()}_longitude'] = self.get_planet_position(date, planet)
            signature[f'{planet.lower()}_sign'] = int(signature[f'{planet.lower()}_longitude'] // 30)
        
        # Аспекты между планетами (важные углы)
        signature['sun_moon_aspect'] = self._calculate_aspect(
            signature['sun_longitude'], 
            signature['moon_longitude']
        )
        
        signature['venus_mars_aspect'] = self._calculate_aspect(
            signature['venus_longitude'], 
            signature['mars_longitude']
        )
        
        return signature
    
    def _calculate_aspect(self, lon1: float, lon2: float) -> str:
        """
        Рассчитать аспект между двумя планетами
        
        Args:
            lon1: Долгота первой планеты
            lon2: Долгота второй планеты
            
        Returns:
            Название аспекта
        """
        diff = abs(lon1 - lon2) % 360
        if diff > 180:
            diff = 360 - diff
        
        # Орбисы для аспектов (допустимое отклонение)
        aspects = [
            (0, 'Conjunction', 8),      # Соединение
            (60, 'Sextile', 6),         # Секстиль
            (90, 'Square', 8),          # Квадрат
            (120, 'Trine', 8),          # Трин
            (180, 'Opposition', 8),     # Оппозиция
        ]
        
        for angle, name, orb in aspects:
            if abs(diff - angle) <= orb:
                return name
        
        return 'None'
    
    def calculate_pattern_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """
        Рассчитать схожесть двух астрологических сигнатур
        
        Args:
            sig1: Первая сигнатура
            sig2: Вторая сигнатура
            
        Returns:
            Коэффициент схожести (0-1, где 1 - полное совпадение)
        """
        similarity_score = 0
        max_score = 0
        
        # Схожесть фазы луны (вес: 3)
        if sig1['moon_phase_id'] == sig2['moon_phase_id']:
            similarity_score += 3
        max_score += 3
        
        # Схожесть знака луны (вес: 2)
        if sig1['moon_sign'] == sig2['moon_sign']:
            similarity_score += 2
        max_score += 2
        
        # Схожесть освещённости луны (вес: 2)
        illum_diff = abs(sig1['moon_illumination'] - sig2['moon_illumination'])
        similarity_score += max(0, 2 - illum_diff / 25)
        max_score += 2
        
        # Схожесть аспекта Солнце-Луна (вес: 2)
        if sig1['sun_moon_aspect'] == sig2['sun_moon_aspect']:
            similarity_score += 2
        max_score += 2
        
        # Схожесть знака Венеры (вес: 1)
        if sig1['venus_sign'] == sig2['venus_sign']:
            similarity_score += 1
        max_score += 1
        
        # Схожесть знака Марса (вес: 1)
        if sig1['mars_sign'] == sig2['mars_sign']:
            similarity_score += 1
        max_score += 1
        
        return similarity_score / max_score
    
    def analyze_historical_patterns(
        self, 
        price_data: pd.DataFrame, 
        current_date: datetime,
        min_similarity: float = 0.7,
        max_matches: int = 20
    ) -> Dict:
        """
        Найти схожие астрологические паттерны в истории и проанализировать цену
        
        Args:
            price_data: DataFrame с историческими данными (index - даты, колонки: open, high, low, close)
            current_date: Текущая дата для анализа
            min_similarity: Минимальный порог схожести
            max_matches: Максимальное количество найденных паттернов
            
        Returns:
            Словарь с результатами анализа и прогнозом
        """
        # Получаем сигнатуру для текущей даты
        current_sig = self.get_astrological_signature(current_date)
        
        matches = []
        
        # Ищем схожие паттерны в истории
        for idx, row in price_data.iterrows():
            if idx >= current_date:
                continue
                
            hist_sig = self.get_astrological_signature(idx)
            similarity = self.calculate_pattern_similarity(current_sig, hist_sig)
            
            if similarity >= min_similarity:
                # Определяем направление свечи следующего дня
                next_day_idx = idx + timedelta(days=1)
                if next_day_idx in price_data.index:
                    next_close = price_data.loc[next_day_idx, 'close']
                    current_close = row['close']
                    direction = 1 if next_close > current_close else (-1 if next_close < current_close else 0)
                    
                    matches.append({
                        'date': idx,
                        'similarity': similarity,
                        'direction': direction,
                        'price_change': (next_close - current_close) / current_close * 100,
                        'current_close': current_close,
                        'next_close': next_close,
                    })
        
        # Сортируем по схожести
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        matches = matches[:max_matches]
        
        # Анализируем результаты
        if not matches:
            return {
                'status': 'no_matches',
                'message': 'Не найдено схожих паттернов',
                'prediction': None,
            }
        
        # Расчёт статистики
        total_matches = len(matches)
        bullish_count = sum(1 for m in matches if m['direction'] == 1)
        bearish_count = sum(1 for m in matches if m['direction'] == -1)
        neutral_count = sum(1 for m in matches if m['direction'] == 0)
        
        avg_price_change = np.mean([m['price_change'] for m in matches])
        weighted_direction = np.average(
            [m['direction'] for m in matches],
            weights=[m['similarity'] for m in matches]
        )
        
        # Прогноз
        if weighted_direction > 0.3:
            prediction = 'BULLISH'
            confidence = min(weighted_direction, 1.0) * 100
        elif weighted_direction < -0.3:
            prediction = 'BEARISH'
            confidence = abs(min(weighted_direction, -1.0)) * 100
        else:
            prediction = 'NEUTRAL'
            confidence = (1 - abs(weighted_direction)) * 100
        
        return {
            'status': 'success',
            'current_date': current_date.strftime('%Y-%m-%d'),
            'current_signature': current_sig,
            'total_matches': total_matches,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'bullish_percentage': bullish_count / total_matches * 100,
            'bearish_percentage': bearish_count / total_matches * 100,
            'avg_price_change': avg_price_change,
            'weighted_direction': weighted_direction,
            'prediction': prediction,
            'confidence': confidence,
            'matches': matches,
        }
    
    def generate_report(self, analysis_result: Dict) -> str:
        """
        Сгенерировать текстовый отчёт по анализу
        
        Args:
            analysis_result: Результаты анализа
            
        Returns:
            Текстовый отчёт
        """
        if analysis_result['status'] != 'success':
            return f"Статус: {analysis_result['message']}"
        
        report = []
        report.append("=" * 60)
        report.append("АСТРОЛОГИЧЕСКИЙ АНАЛИЗ ТОРГОВОГО ПАТТЕРНА")
        report.append("=" * 60)
        report.append(f"\nДата анализа: {analysis_result['current_date']}")
        report.append(f"\nНайдено схожих паттернов: {analysis_result['total_matches']}")
        report.append(f"\n--- Статистика паттернов ---")
        report.append(f"Бычьих свечей: {analysis_result['bullish_count']} ({analysis_result['bullish_percentage']:.1f}%)")
        report.append(f"Медвежьих свечей: {analysis_result['bearish_count']} ({analysis_result['bearish_percentage']:.1f}%)")
        report.append(f"Нейтральных свечей: {analysis_result['neutral_count']}")
        report.append(f"\nСреднее изменение цены: {analysis_result['avg_price_change']:.2f}%")
        report.append(f"\n--- ПРОГНОЗ ---")
        report.append(f"Направление: {analysis_result['prediction']}")
        report.append(f"Уверенность: {analysis_result['confidence']:.1f}%")
        
        report.append(f"\n--- Текущая астрологическая сигнатура ---")
        sig = analysis_result['current_signature']
        report.append(f"Фаза Луны: {sig['moon_phase']} ({sig['moon_illumination']:.1f}%)")
        report.append(f"Знак Луны: {sig['moon_sign']} (0-Овен, 1-Телец, ...)")
        report.append(f"Аспект Солнце-Луна: {sig['sun_moon_aspect']}")
        report.append(f"Аспект Венера-Марс: {sig['venus_mars_aspect']}")
        
        report.append(f"\n--- Топ-5 наиболее схожих паттернов ---")
        for i, match in enumerate(analysis_result['matches'][:5], 1):
            report.append(f"{i}. {match['date'].strftime('%Y-%m-%d')} "
                         f"(схожесть: {match['similarity']:.2f}, "
                         f"изменение: {match['price_change']:+.2f}%)")
        
        report.append("\n" + "=" * 60)
        report.append("ПРЕДУПРЕЖДЕНИЕ: Данный анализ не является финансовой рекомендацией!")
        report.append("=" * 60)
        
        return "\n".join(report)


def load_price_data(filepath: str) -> pd.DataFrame:
    """
    Загрузить данные графика из CSV файла
    
    Ожидаемый формат CSV:
    date,open,high,low,close,volume
    2020-01-01,1.1234,1.1250,1.1200,1.1240,1000
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        DataFrame с данными графика
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df


def main():
    """Основная функция для запуска анализа"""
    
    # Пример использования
    print("Запуск астрологического анализатора...")
    
    # Создаём анализатор
    analyzer = AstrologicalAnalyzer()
    
    # Генерируем тестовые данные (в реальности загрузите из файла)
    print("\nГенерация тестовых данных графика...")
    dates = pd.date_range(start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), freq='D')
    
    # Симулируем случайные данные графика (замените на реальные данные)
    np.random.seed(42)
    base_price = 1.1000
    prices = []
    for i in range(len(dates)):
        change = np.random.randn() * 0.005
        base_price *= (1 + change)
        prices.append(base_price)
    
    price_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.003)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.003)) for p in prices],
        'close': prices,
    }, index=dates)
    
    # Текущая дата для анализа
    current_date = datetime.now()
    
    print(f"\nАнализ паттернов для даты: {current_date.strftime('%Y-%m-%d')}")
    
    # Выполняем анализ
    result = analyzer.analyze_historical_patterns(
        price_data=price_data,
        current_date=current_date,
        min_similarity=0.6,
        max_matches=30
    )
    
    # Генерируем и выводим отчёт
    report = analyzer.generate_report(result)
    print("\n" + report)
    
    # Сохраняем результаты в JSON
    with open('astro_analysis_result.json', 'w', encoding='utf-8') as f:
        # Преобразуем даты в строки для JSON
        result_serializable = result.copy()
        if 'matches' in result_serializable:
            for match in result_serializable['matches']:
                match['date'] = match['date'].strftime('%Y-%m-%d')
        json.dump(result_serializable, f, indent=2, default=str)
    
    print(f"\nРезультаты сохранены в файл: astro_analysis_result.json")
    
    return result


if __name__ == '__main__':
    main()
