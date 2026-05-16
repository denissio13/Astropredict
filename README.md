# AstroPredict

Анализатор дневных сигналов для крипты и акций MOEX на основе астрологических признаков и исторических паттернов.

Скрипт умеет:
- загружать дневные свечи (`binance`, `bybit`, `moex`),
- считать астропризнаки (через `pyswisseph` или fallback),
- делать прогноз на следующую дневную свечу,
- запускать бэктест,
- делать оптимизацию параметров с отдельными рейтингами `NORMAL` и `REVERSE`,
- в конце прогноза печатать готовый текст для Twitter.

---

## 1) Что важно понять новичку

Скрипт работает с дневками и использует только закрытые дневные свечи.

Это значит:
- сигнал считается по последней закрытой дневной свече,
- и относится к текущей новой дневной свече.

Для Binance дневка закрывается в `00:00 UTC`.
Для Екатеринбурга (`UTC+5`) это `05:00`.

Практика:
- запускать расчёт через 1-3 минуты после переключения дневки (например, в `05:02`).

---

## 2) Установка

### Базовые зависимости

```bash
pip install pandas numpy requests
```

### Точные астрологические расчёты (рекомендуется)

```bash
pip install pyswisseph
```

Проверка:

```bash
python3 - << 'PY'
import swisseph as swe
print("julday:", hasattr(swe, "julday"))
print("calc_ut:", hasattr(swe, "calc_ut"))
PY
```

### Опционально: CuPy

```bash
pip install cupy-cuda12x
```

Примечание: сейчас основная нагрузка всё равно CPU-bound. Для ускорения оптимизации используется многопроцессность (`--workers`).

---

## 3) Быстрый старт

### Прогноз (одна пара)

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT
```

### Прогноз (несколько пар)

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT
```

### Бэктест

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT --backtest --candles 200
```

### Оптимизация

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT --optimize --mode basic --candles 200 --workers 12
```

---

## 4) Режимы работы

Скрипт имеет 3 режима:

1. `Прогноз` (по умолчанию):
   - если не переданы `--optimize` и `--backtest`, скрипт даёт прогноз по каждой паре.

2. `Бэктест` (`--backtest`):
   - тест на истории с текущими параметрами.

3. `Оптимизация` (`--optimize`):
   - перебор параметров, отдельный скоринг для `NORMAL` и `REVERSE`, топ-3 в конце.

---

## 5) Все CLI-флаги

### Обязательные

- `--pairs`
  - список тикеров через запятую.
  - Примеры:
    - крипта: `BTCUSDT,ETHUSDT`
    - MOEX: `SBER,GAZP`

### Источник данных

- `--exchange`
  - `binance` (по умолчанию), `bybit`, `moex`.

- `--moex-board`
  - код board для MOEX, по умолчанию `TQBR`.
  - Используется только при `--exchange moex`.

### Режимы

- `--backtest` — включить бэктест.
- `--optimize` — включить оптимизацию.
- `--mode` — режим оптимизации: `basic` или `full`.

### Параметры расчёта

- `--candles`
  - сколько последних свечей использовать для бэктеста/оптимизации.

- `--similarity`
  - порог похожести паттернов (используется в обычном прогнозе и бэктесте).

- `--max-matches`
  - максимум совпадений в прогнозе.

- `--target-date` / `-d`
  - прогноз относительно выбранной даты (`YYYY-MM-DD`) в истории.

### Параллелизм

- `--workers`
  - количество процессов CPU в оптимизации.
  - по умолчанию: `cpu_count - 2`.

### Использование оптимизированной стратегии в live-прогнозе

- `--use-optimized`
  - брать параметры из `optimization_result.json`.

- `--opt-file`
  - путь до файла с оптимизацией (по умолчанию `optimization_result.json`).

- `--strategy`
  - `auto`, `normal`, `reverse`.
  - `auto` выбирает лучший score между top normal/reverse для конкретной пары.

---

## 6) Примеры команд

### 6.1 Крипта, прогноз с оптимизированными настройками

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT --use-optimized --strategy auto
```

### 6.2 Крипта, оптимизация full

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT --optimize --mode full --candles 250 --workers 12
```

### 6.3 MOEX, оптимизация

```bash
python3 astro_trading_analyzer.py --exchange moex --moex-board TQBR --pairs SBER,GAZP --optimize --mode basic --candles 220 --workers 8
```

### 6.4 MOEX, прогноз из оптимизированных параметров

```bash
python3 astro_trading_analyzer.py --exchange moex --moex-board TQBR --pairs SBER,GAZP --use-optimized --strategy auto
```

### 6.5 Бэктест одной пары

```bash
python3 astro_trading_analyzer.py --pairs ETHUSDT --backtest --candles 200 --similarity 0.8
```

---

## 7) Как читать результаты оптимизации

Для каждой комбинации параметров считаются два сценария:

- `NORMAL` — сигнал как есть.
- `REVERSE` — сигнал инвертируется (bullish <-> bearish).

В конце выводятся:
- `TOP 3 NORMAL`
- `TOP 3 REVERSE`

Ключевые метрики:
- `Trades` — число сделок,
- `WinRate` — доля прибыльных,
- `PnL` — итог в USD (при risk-модели ниже),
- `PF` — profit factor,
- `DD$`, `DD%Peak` — max drawdown от пика,
- `DD$Init`, `DD%Init` — просадка от стартового депозита,
- `MaxWinStreak`, `MaxLossStreak`, `WorstLosingStreakPnL`.

Risk-модель оптимизации фиксирована:
- стартовый депозит: `$1000`,
- плечо: `x1`,
- ставка в сделку: `$20`.

Также даётся рекомендация по плечу под лимит `DD_init <= 25%`.

---

## 8) Структура `optimization_result.json`

Файл хранит результаты по каждой паре отдельно:

```json
{
  "pairs": {
    "BTCUSDT": { "top_normal": [...], "top_reverse": [...], "timestamp": "..." },
    "ETHUSDT": { "top_normal": [...], "top_reverse": [...], "timestamp": "..." }
  },
  "last_updated_pair": "ETHUSDT"
}
```

Это позволяет безопасно работать с несколькими инструментами без перезаписи результатов друг друга.

---

## 9) Twitter-вывод

В режиме обычного прогноза (не `--optimize`, не `--backtest`) скрипт в конце печатает готовый блок:

- заголовок `AstroPredict dd/mm/yy`,
- сигнал по каждой паре,
- confidence, matches,
- setup (`Normal/Reverse`, `W/T/M`),
- дисклеймер.

Можно сразу копировать и публиковать.

---

## 10) Где хранятся данные

- Исторические свечи: папка `data/`.
  - Примеры:
    - `data/binance_BTCUSDT_daily.csv`
    - `data/moex_TQBR_SBER_daily.csv`
- Оптимизация: `optimization_result.json`.

Кэш свечей обновляется раз в сутки.

---

## 11) Частые проблемы

### `NameError/AttributeError` по `swisseph`

Ставьте именно `pyswisseph`, не заглушку `swisseph`:

```bash
pip uninstall -y swisseph
pip install -U pyswisseph
```

### Мало данных / нет сигналов

- увеличьте `--candles`,
- снизьте `--similarity` (для backtest/manual),
- проверьте корректность тикера и биржи.

### GPU видна, но util около 0%

Нормально для текущей версии: основная нагрузка — CPU и multiprocessing.

---

## 12) Рекомендованный ежедневный workflow

1. Периодически переоптимизировать (например, раз в неделю):

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT --optimize --mode basic --candles 250 --workers 12
```

2. Каждый день после закрытия дневки (например, 05:02 Екатеринбург):

```bash
python3 astro_trading_analyzer.py --pairs BTCUSDT,ETHUSDT --use-optimized --strategy auto
```

3. Скопировать блок `READY TO POST (TWITTER)` и опубликовать.

---

## 13) Disclaimer

Проект только для исследования и обучения.
Это не инвестиционная рекомендация.
Результаты на истории не гарантируют результат в будущем.
