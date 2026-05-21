# 🔥 Heat Analytics System

Система анализа теплопотребления многоквартирных домов (МКД) для магистерской диссертации.

## 📋 Описание

Веб-приложение для загрузки, анализа и визуализации данных тепловычислителей МКД. Включает:

- **Загрузка Excel-файлов** с данными тепловычислителей
- **Статистическое моделирование** (OLS, Huber, Ridge, Lasso, Quantile regression)
- **Временные ряды** (декомпозиция, Holt-Winters, Prophet)
- **Обнаружение аномалий** (EWMA, Isolation Forest, LOF)
- **Кластеризация** (K-Means++, DBSCAN, GMM)
- **Сравнение с нормативами** и классы энергоэффективности
- **Экспорт отчётов** в PDF/CSV

## 🛠 Технологический стек

| Компонент | Технология |
|-----------|------------|
| Backend | Python 3.10+, FastAPI, SQLAlchemy (async), Pydantic |
| Analytics | pandas, numpy, scikit-learn, statsmodels, prophet |
| Frontend | React 18, Vite, TypeScript, TailwindCSS, Recharts |
| Database | SQLite (aiosqlite) |
| Контейнеризация | Docker, docker-compose |

## 🚀 Быстрый старт

### Требования

- Docker и docker-compose
- Или Python 3.10+ и Node.js 18+ для локальной разработки

### Запуск через Docker (рекомендуется)

```bash
cd heat-analytics

# Сборка и запуск
docker-compose up --build

# Приложение доступно по адресам:
# - Backend API: http://localhost:8000
# - Frontend UI: http://localhost:3000
# - Swagger Docs: http://localhost:8000/docs
```

### Локальная разработка

#### Backend

```bash
cd heat-analytics

# Установка зависимостей
poetry install

# Активация виртуального окружения
poetry shell

# Запуск сервера
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd heat-analytics/frontend

# Установка зависимостей
npm install

# Запуск dev-сервера
npm run dev
```

## 📁 Структура проекта

```
heat-analytics/
├── backend/
│   ├── main.py                 # FastAPI приложение
│   ├── config/                 # Настройки (settings.py, norms.yaml)
│   ├── models/                 # SQLAlchemy ORM + Pydantic схемы
│   ├── routers/                # API эндпоинты
│   ├── services/               # Бизнес-логика (parser, db, analytics_runner)
│   └── analytics/              # Статистические модели
├── frontend/
│   ├── src/
│   │   ├── components/         # React компоненты
│   │   ├── pages/              # Страницы приложения
│   │   └── services/           # API клиент
│   └── package.json
├── data/                       # Данные и weather stub
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 📡 API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/api/upload` | Загрузка Excel файла |
| POST | `/api/analyze` | Запуск анализа |
| GET | `/api/results/{run_id}` | Получение результатов |
| GET | `/api/buildings` | Список МКД |
| POST | `/api/export/pdf` | Экспорт PDF отчёта |
| POST | `/api/export/csv` | Экспорт CSV отчёта |
| GET | `/health` | Health check |

## 📊 Формат входных данных

Excel-файл должен содержать:
- Метаданные в строках 1-4
- Данные начиная со строки 6
- Колонки: Дата, Время, T1, T2, P1, P2, V1, V2, M1, M2, Q, НС, и др.

## 🧮 Аналитические методы

### Регрессия (1.2.1)
- OLS (МНК)
- Robust (Huber)
- Ridge/Lasso
- Quantile

### Временные ряды (1.2.2)
- Сезонная декомпозиция
- Holt-Winters
- Prophet

### Аномалии (2.2.iii)
- EWMA
- Isolation Forest
- Local Outlier Factor

### Кластеризация (1.2.3)
- K-Means++
- DBSCAN
- Gaussian Mixture

## 🧪 Тестирование

```bash
# Запуск тестов
pytest

# С покрытием
pytest --cov=backend --cov-report=html
```

## 📝 Лицензия

Проект создан для образовательных целей в рамках магистерской диссертации.

## 👨‍💻 Автор

Студент магистратуры
Email: student@example.com
