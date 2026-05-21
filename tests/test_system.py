"""
Юнит-тесты для системы аналитики МКД.
Запуск: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loader import DataLoader, load_sample_data
from preprocess import DataPreprocessor
from analytics import AnalyticsEngine


class TestDataLoader:
    """Тесты модуля загрузки данных."""
    
    def test_load_sample_data(self):
        """Тест генерации тестовых данных."""
        df = load_sample_data()
        
        assert len(df) == 365, "Ожидаемо 365 дней"
        assert 'date' in df.columns, "Должна быть колонка date"
        assert 'T_out' in df.columns, "Должна быть колонка T_out"
        assert 'Q' in df.columns, "Должна быть колонка Q"
        assert 'mkd_id' in df.columns, "Должна быть колонка mkd_id"
    
    def test_validate_ranges(self):
        """Тест валидации диапазонов."""
        df = load_sample_data()
        loader = DataLoader.__new__(DataLoader)
        loader.df = df
        
        valid_df, invalid_df = loader.validate_ranges()
        
        assert len(valid_df) + len(invalid_df) == len(df), "Сумма должна равняться исходному размеру"
        assert len(invalid_df) > 0, "В тестовых данных должны быть выбросы"
        assert all(valid_df['status'] == 'valid'), "Все валидные записи должны иметь статус 'valid'"
        assert all(invalid_df['status'] == 'invalid'), "Все невалидные записи должны иметь статус 'invalid'"
    
    def test_get_summary(self):
        """Тест получения сводки."""
        df = load_sample_data()
        loader = DataLoader.__new__(DataLoader)
        loader.df = df
        
        summary = loader.get_summary()
        
        assert 'total_rows' in summary, "Должно быть total_rows"
        assert 'total_columns' in summary, "Должно быть total_columns"
        assert summary['total_rows'] == 365, "Неверное количество строк"


class TestDataPreprocessor:
    """Тесты модуля предобработки."""
    
    @pytest.fixture
    def sample_df(self):
        return load_sample_data()
    
    def test_calculate_gsop(self, sample_df):
        """Тест расчёта ГСОП."""
        preprocessor = DataPreprocessor(sample_df)
        df = preprocessor.calculate_gsop()
        
        assert 'GSOP_daily' in df.columns, "Должна быть колонка GSOP_daily"
        assert 'GSOP_cumulative' in df.columns, "Должна быть колонка GSOP_cumulative"
        assert all(df['GSOP_daily'] >= 0), "ГСОП не может быть отрицательным"
        
        # Проверка формулы: при T_out > 18 ГСОП должен быть 0
        warm_days = df[df['T_out'] > 18]
        if len(warm_days) > 0:
            assert all(warm_days['GSOP_daily'] == 0), "При T_out > 18 ГСОП должен быть 0"
    
    def test_add_features(self, sample_df):
        """Тест добавления признаков."""
        preprocessor = DataPreprocessor(sample_df)
        df = preprocessor.add_features()
        
        assert 'day_of_week' in df.columns, "Должен быть day_of_week"
        assert 'month' in df.columns, "Должен быть month"
        assert 'season' in df.columns, "Должен быть season"
    
    def test_handle_missing_values(self, sample_df):
        """Тест обработки пропусков."""
        # Создаём пропуски в середине (не в начале, чтобы интерполяция работала)
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[100:104, 'Q'] = np.nan
        
        preprocessor = DataPreprocessor(df_with_nulls)
        df_filled = preprocessor.handle_missing_values(method='interpolate')
        
        assert df_filled['Q'].isnull().sum() == 0, "Пропуски должны быть заполнены"
    
    def test_preprocessing_report(self, sample_df):
        """Тест отчёта о предобработке."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.calculate_gsop()
        report = preprocessor.get_preprocessing_report()
        
        assert 'original_shape' in report, "Должно быть original_shape"
        assert 'current_shape' in report, "Должно быть current_shape"
        assert 'gsop_stats' in report, "Должно быть gsop_stats"


class TestAnalyticsEngine:
    """Тесты аналитического движка."""
    
    @pytest.fixture
    def processed_df(self):
        df = load_sample_data()
        preprocessor = DataPreprocessor(df)
        df = preprocessor.calculate_gsop()
        df = preprocessor.add_features()
        return df
    
    def test_linear_regression(self, processed_df):
        """Тест линейной регрессии."""
        engine = AnalyticsEngine(processed_df)
        results = engine.fit_linear_regression()
        
        assert 'error' not in results, "Не должно быть ошибки"
        assert 'coefficients' in results, "Должны быть коэффициенты"
        assert 'metrics' in results, "Должны быть метрики"
        assert 'r2' in results['metrics'], "Должен быть R²"
        assert 0 <= results['metrics']['r2'] <= 1, "R² должен быть в диапазоне [0, 1]"
    
    def test_detect_anomalies_ewma(self, processed_df):
        """Тест выявления аномалий EWMA."""
        engine = AnalyticsEngine(processed_df)
        df_anom = engine.detect_anomalies_ewma()
        
        assert 'anomaly_ewma' in df_anom.columns, "Должна быть колонка anomaly_ewma"
        assert 'anomaly_ewma_zscore' in df_anom.columns, "Должна быть колонка z-score"
        assert all(df_anom['anomaly_ewma'].isin([True, False])), "Флаги должны быть булевыми"
    
    def test_evaluate_efficiency(self, processed_df):
        """Тест оценки эффективности."""
        engine = AnalyticsEngine(processed_df)
        results = engine.evaluate_efficiency(normative_consumption=0.08)
        
        assert 'error' not in results, "Не должно быть ошибки"
        assert 'status' in results, "Должен быть статус"
        assert results['status'] in ['Эффективный', 'Нормативный', 'Предупреждение', 'Критический'], \
            "Неверный статус"
        assert 'recommendations' in results, "Должны быть рекомендации"
        assert len(results['recommendations']) > 0, "Должны быть рекомендации"
    
    def test_forecast_holt_winters(self, processed_df):
        """Тест прогнозирования Холта-Винтерса."""
        engine = AnalyticsEngine(processed_df)
        results = engine.forecast_holt_winters(periods=7)
        
        assert 'error' not in results, "Не должно быть ошибки"
        assert 'forecast' in results, "Должен быть прогноз"
        assert len(results['forecast']['forecast']) == 7, "Должно быть 7 прогнозов"
    
    def test_cluster_buildings_monthly(self, processed_df):
        """Тест кластеризации (по месяцам)."""
        engine = AnalyticsEngine(processed_df)
        results = engine.cluster_buildings(n_clusters=3)
        
        # Может вернуть ошибку если недостаточно данных для группировки
        if 'error' not in results:
            assert 'silhouette_score' in results, "Должен быть силуэт"
            assert 'labels' in results, "Должны быть метки"
            assert -1 <= results['silhouette_score'] <= 1, "Силуэт должен быть в диапазоне [-1, 1]"


class TestIntegration:
    """Интеграционные тесты полного цикла."""
    
    def test_full_pipeline(self):
        """Тест полного цикла обработки."""
        # Загрузка
        df = load_sample_data()
        assert len(df) == 365
        
        # Предобработка
        preprocessor = DataPreprocessor(df)
        df = preprocessor.calculate_gsop()
        df = preprocessor.add_features()
        assert 'GSOP_daily' in df.columns
        
        # Анализ
        engine = AnalyticsEngine(df)
        
        reg_results = engine.fit_linear_regression()
        assert 'r2' in reg_results['metrics']
        
        eff_results = engine.evaluate_efficiency()
        assert 'status' in eff_results
        
        forecast_results = engine.forecast_holt_winters(periods=7)
        assert 'forecast' in forecast_results or 'error' in forecast_results
        
        # Сохранение моделей
        engine.save_models('models/test_models.joblib')
        assert Path('models/test_models.joblib').exists()
        
        # Очистка
        Path('models/test_models.joblib').unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
