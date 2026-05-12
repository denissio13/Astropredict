# Астрологический анализатор торговых паттернов

Скрипт для автоматической загрузки криптовалютных данных с Binance/Bybit, анализа исторических данных на предмет схожих астрологических паттернов и прогнозирования направления дневной свечи.

## Возможности

- **Автоматическая загрузка данных**: Скачивание исторических данных с Binance и Bybit для криптовалютных пар
- **Расчёт положений планет**: Солнце, Луна, Меркурий, Венера, Марс, Юпитер, Сатурн
- **Определение фазы Луны**: 8 фаз лунного цикла с процентом освещённости
- **Расчёт аспектов**: Соединение, Секстиль, Квадрат, Трин, Оппозиция
- **Поиск схожих паттернов**: Сравнение астрологических сигнатур дат
- **Анализ ценовых движений**: Статистика бычьих/медвежьих свечей для схожих дат
- **Прогноз направления**: BULLISH / BEARISH / NEUTRAL с уровнем уверенности
- **Мульти-парный анализ**: Поддержка анализа нескольких пар одновременно
- **Бэктестинг стратегии**: Проверка точности прогнозов на исторических данных
- **Самообучение и оптимизация**: Автоматический подбор оптимальных параметров для максимальной точности
- **Прогноз на конкретную дату**: Возможность указать любую дату для анализа

## Установка зависимостей

```bash
# Обязательные зависимости
pip install pandas numpy requests

# Опционально: для точных расчётов положений планет
pip install swisseph
```

**Примечание**: Если `swisseph` не установлен, скрипт использует упрощённую модель расчёта позиций планет на основе их орбитальных периодов.

## Использование

### Быстрый запуск с автоматической загрузкой данных

```bash
# Анализ одной пары
python astro_trading_analyzer.py --pairs BTCUSDT

# Анализ нескольких пар через запятую
python astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT,SOLUSDT

# Выбор биржи (по умолчанию Binance)
python astro_trading_analyzer.py --pairs ETHUSDT --exchange bybit

# Настройка параметров анализа
python astro_trading_analyzer.py --pairs BTCUSDT --similarity 0.8 --max-matches 50
```

### Параметры командной строки

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `--pairs` | Список криптовалютных пар через запятую | Обязательно |
| `--exchange` | Биржа для загрузки данных (binance/bybit) | binance |
| `--data` | Путь к папке для сохранения данных | data/ |
| `--similarity` | Минимальный порог схожести паттернов (0-1) | 0.7 |
| `--max-matches` | Максимальное количество найденных паттернов | 30 |
| `--target-date` | Дата для прогноза (YYYY-MM-DD) | Последняя доступная дата |
| `--backtest` | Запустить режим бэктестинга | Выключено |
| `--optimize` | Запустить режим оптимизации параметров | Выключено |
| `--candles` | Количество свечей для бэктестинга/оптимизации | 200 |
| `--tolerance` | Допустимое отклонение для точности прогноза (%) | 0.5 |

### Примеры команд

```bash
# Анализ Bitcoin с Binance
python astro_trading_analyzer.py --pairs BTCUSDT

# Анализ Ethereum и Solana с Bybit
python astro_trading_analyzer.py --pairs ETHUSDT,SOLUSDT --exchange bybit

# Глубокий анализ с большим количеством совпадений
python astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT --similarity 0.6 --max-matches 100

# Прогноз на конкретную историческую дату
python astro_trading_analyzer.py --pairs BTCUSDT --target-date 2024-06-15

# Бэктестинг стратегии на последних 200 свечах
python astro_trading_analyzer.py --pairs BTCUSDT --backtest --candles 200

# Бэктестинг с настройкой точности (допустимое отклонение 1%)
python astro_trading_analyzer.py --pairs ETHUSDT --backtest --candles 300 --tolerance 1.0

# Оптимизация параметров для поиска наилучшей конфигурации
python astro_trading_analyzer.py --pairs BTCUSDT --optimize --candles 200

# Комбинированный режим: оптимизация с последующим прогнозом
python astro_trading_analyzer.py --pairs SOLUSDT --optimize --candles 150 --similarity 0.75
```

## Хранение данных

Скрипт автоматически скачивает и сохраняет исторические данные в папку `data/`:

```
data/
├── binance_BTCUSDT_daily.csv
├── binance_ETHUSDT_daily.csv
└── bybit_SOLUSDT_daily.csv
```

Формат файлов CSV:
```csv
date,open,high,low,close,volume
2020-01-01 00:00:00,1.1234,1.1250,1.1200,1.1240,1000
...
```

При повторном запуске скрипт использует кэшированные данные, если они есть.

## Результаты бэктестинга и оптимизации

При запуске в режиме бэктестинга или оптимизации скрипт создаёт JSON файлы с подробной статистикой:

```
backtest_BTCUSDT_20240615_143022.json  # Результаты бэктестинга
optimize_BTCUSDT_20240615_145530.json  # Результаты оптимизации
```

### Структура отчёта бэктестинга

```json
{
  "symbol": "BTCUSDT",
  "mode": "backtest",
  "parameters": {
    "candles_tested": 200,
    "similarity_threshold": 0.7,
    "tolerance_percent": 0.5
  },
  "statistics": {
    "total_predictions": 180,
    "correct_predictions": 126,
    "accuracy_percent": 70.0,
    "bullish_correct": 68,
    "bearish_correct": 58,
    "neutral_correct": 0,
    "average_confidence": 85.3
  }
}
```

### Структура отчёта оптимизации

```json
{
  "symbol": "BTCUSDT",
  "mode": "optimize",
  "best_parameters": {
    "similarity_threshold": 0.75,
    "max_matches": 40,
    "tolerance_percent": 0.8
  },
  "best_accuracy": 73.5,
  "tested_combinations": 45,
  "top_configurations": [
    {"similarity": 0.75, "max_matches": 40, "accuracy": 73.5},
    {"similarity": 0.70, "max_matches": 35, "accuracy": 71.2},
    ...
  ]
}
```

## Веса параметров схожести

При расчёте схожести астрологических сигнатур используются следующие веса:

- Фаза Луны: 3
- Знак Луны: 2
- Освещённость Луны: 2
- Аспект Солнце-Луна: 2
- Знак Венеры: 1
- Знак Марса: 1

## Выводимые данные

Скрипт генерирует:

1. **Текстовый отчёт** в консоль для каждой анализируемой пары с:
   - Названием пары и биржей
   - Датой анализа
   - Количеством найденных паттернов
   - Статистикой бычьих/медвежьих свечей
   - Прогнозом и уровнем уверенности
   - Текущей астрологической сигнатурой
   - Топ-5 наиболее схожих исторических паттернов

2. **JSON файл** `astro_analysis_result.json` с полными данными анализа по всем парам

## Пример вывода

```
============================================================
АСТРОЛОГИЧЕСКИЙ АНАЛИЗ ТОРГОВОГО ПАТТЕРНА
============================================================

Пара: BTCUSDT (Binance)
Дата анализа: 2026-05-12

Найдено схожих паттернов: 30

--- Статистика паттернов ---
Бычьих свечей: 16 (53.3%)
Медвежьих свечей: 14 (46.7%)
Нейтральных свечей: 0

Среднее изменение цены: 0.06%

--- ПРОГНОЗ ---
Направление: NEUTRAL
Уверенность: 92.9%

--- Текущая астрологическая сигнатура ---
Фаза Луны: Waning Crescent (12.9%)
Знак Луны: 0 (0-Овен, 1-Телец, ...)
Аспект Солнце-Луна: None
Аспект Венера-Марс: None

--- Топ-5 наиболее схожих паттернов ---
1. 2024-05-04 (схожесть: 0.90, изменение: +0.09%)
2. 2024-06-02 (схожесть: 0.90, изменение: +1.60%)
...

------------------------------------------------------------
Пара: ETHUSDT (Binance)
...
```

## API для программного использования

Скрипт также можно использовать как библиотеку в своём коде:

```python
from astro_trading_analyzer import CryptoDataLoader, AstrologicalAnalyzer
from datetime import datetime

# Загрузка данных с биржи
loader = CryptoDataLoader(exchange='binance')
price_data = loader.download_pair('BTCUSDT', timeframe='1d', limit=1000)

# Создание анализатора
analyzer = AstrologicalAnalyzer()

# Дата для анализа
current_date = datetime.now()

# Анализ паттернов
result = analyzer.analyze_historical_patterns(
    price_data=price_data,
    current_date=current_date,
    min_similarity=0.7,
    max_matches=30
)

# Генерация отчёта
report = analyzer.generate_report(result, symbol='BTCUSDT')
print(report)

# Бэктестинг стратегии
backtest_stats = analyzer.backtest(
    df=price_data,
    candles=200,
    tolerance_percent=0.5
)
print(f"Точность стратегии: {backtest_stats['accuracy_percent']:.2f}%")

# Оптимизация параметров
opt_results = analyzer.optimize_parameters(
    df=price_data,
    candles=200
)
print(f"Лучшие параметры: {opt_results['best_parameters']}")
print(f"Лучшая точность: {opt_results['best_accuracy']:.2f}%")
```

### Основные классы и методы

**CryptoDataLoader:**
- `download_pair(symbol, timeframe='1d', limit=1000)` - Скачать данные пары
- `save_to_csv(data, symbol, exchange)` - Сохранить в CSV
- `load_from_csv(symbol, exchange)` - Загрузить из CSV

**AstrologicalAnalyzer:**
- `get_planet_position(date, planet_name)` - Положение планеты на дату
- `get_moon_phase(date)` - Фаза и освещённость Луны
- `get_astrological_signature(date)` - Полная астрологическая сигнатура
- `calculate_pattern_similarity(sig1, sig2)` - Коэффициент схожести сигнатур
- `analyze_historical_patterns(price_data, current_date, ...)` - Основной анализ
- `generate_report(analysis_result, symbol='')` - Генерация текстового отчёта
- `backtest(df, candles=200, tolerance_percent=0.5)` - Бэктестинг стратегии на исторических данных
- `optimize_parameters(df, candles=200)` - Поиск оптимальных параметров через перебор комбинаций
- `get_prediction_direction(bullish_count, bearish_count)` - Определение направления прогноза

## Важное предупреждение

⚠️ **Данный анализ не является финансовой рекомендацией!**

Астрологический анализ рынков носит экспериментальный характер и не должен использоваться как единственный источник информации для принятия торговых решений. Всегда проводите собственный анализ и консультируйтесь с финансовыми специалистами.

## Лицензия

Скрипт предоставлен "как есть" для образовательных и исследовательских целей.
