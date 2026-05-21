"""
Статистическое ядро системы аналитики.
Реализация регрессии, кластеризации, выявления аномалий и прогнозирования.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib


class AnalyticsEngine:
    """Основной движок статистического анализа данных МКД."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация аналитического движка.
        
        Args:
            df: DataFrame с предобработанными данными
        """
        self.df = df.copy()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        
        logger.info("Инициализация аналитического движка")
    
    # =========================================================================
    # 1. РЕГРЕССИОННЫЙ АНАЛИЗ
    # =========================================================================
    
    def fit_linear_regression(self, target: str = 'Q', features: List[str] = None) -> Dict:
        """
        Построение линейной регрессии потребления от температуры.
        Q = α + β·T_out + ε
        
        Args:
            target: Целевая переменная
            features: Список признаков (по умолчанию ['T_out'])
            
        Returns:
            Словарь с коэффициентами и метриками модели
        """
        from sklearn.linear_model import LinearRegression, HuberRegressor
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"Построение линейной регрессии: {target} ~ T_out")
        
        if features is None:
            features = ['T_out']
        
        # Подготовка данных
        df_clean = self.df.dropna(subset=[target] + features)
        
        if len(df_clean) < 10:
            logger.warning("Недостаточно данных для регрессии")
            return {'error': 'insufficient_data'}
        
        X = df_clean[features].values
        y = df_clean[target].values
        
        # Обычная линейная регрессия
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        
        # Робастная регрессия (Huber)
        hr = HuberRegressor()
        hr.fit(X, y)
        y_pred_robust = hr.predict(X)
        
        # Метрики
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
        
        results = {
            'model_type': 'linear_regression',
            'coefficients': {
                'intercept': float(lr.intercept_),
                **{f: float(c) for f, c in zip(features, lr.coef_)}
            },
            'robust_coefficients': {
                'intercept': float(hr.intercept_),
                **{f: float(c) for f, c in zip(features, hr.coef_)}
            },
            'metrics': {
                'r2': float(r2),
                'rmse': float(rmse),
                'cv_r2_mean': float(cv_scores.mean()),
                'cv_r2_std': float(cv_scores.std()),
            },
            'predictions': y_pred,
            'residuals': y - y_pred,
        }
        
        self.models['linear_regression'] = lr
        self.models['huber_regression'] = hr
        self.results['regression'] = results
        
        logger.info(f"Регрессия построена: R²={r2:.3f}, RMSE={rmse:.3f}")
        
        return results
    
    def fit_quantile_regression(self, target: str = 'Q', quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict:
        """
        Квантильная регрессия для построения доверительных интервалов.
        
        Args:
            target: Целевая переменная
            quantiles: Список квантилей
            
        Returns:
            Словарь с моделями квантильной регрессии
        """
        try:
            from sklearn.linear_model import QuantileRegressor
        except ImportError:
            logger.warning("QuantileRegressor требует scikit-learn >= 1.0")
            return {'error': 'version_mismatch'}
        
        logger.info(f"Построение квантильной регрессии для {quantiles}")
        
        df_clean = self.df.dropna(subset=['Q', 'T_out'])
        X = df_clean[['T_out']].values
        y = df_clean['Q'].values
        
        quantile_models = {}
        
        for q in quantiles:
            qr = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
            qr.fit(X, y)
            quantile_models[f'q_{int(q*100)}'] = {
                'model': qr,
                'coefficient': float(qr.coef_[0]),
                'intercept': float(qr.intercept_),
            }
        
        self.results['quantile_regression'] = quantile_models
        
        logger.info(f"Квантильная регрессия построена для {len(quantiles)} квантилей")
        
        return quantile_models
    
    # =========================================================================
    # 2. ВЫЯВЛЕНИЕ АНОМАЛИЙ
    # =========================================================================
    
    def detect_anomalies_ewma(self, column: str = 'Q', span: int = 7, threshold: float = 3.0) -> pd.DataFrame:
        """
        Выявление аномалий методом EWMA (Exponentially Weighted Moving Average).
        
        Args:
            column: Анализируемая колонка
            span: Период сглаживания
            threshold: Порог в стандартных отклонениях
            
        Returns:
            DataFrame с флагами аномалий
        """
        logger.info(f"EWMA-анализ аномалий для {column} (span={span})")
        
        df = self.df.copy()
        
        # EWMA статистика
        ewm_mean = df[column].ewm(span=span).mean()
        ewm_std = df[column].ewm(span=span).std()
        
        # Z-score на основе EWMA
        z_score = (df[column] - ewm_mean) / ewm_std.replace(0, np.nan)
        
        # Флаги аномалий
        df['anomaly_ewma'] = z_score.abs() > threshold
        df['anomaly_ewma_zscore'] = z_score
        
        n_anomalies = df['anomaly_ewma'].sum()
        logger.info(f"Найдено {n_anomalies} аномалий (EWMA)")
        
        self.results['anomalies_ewma'] = {
            'count': int(n_anomalies),
            'percentage': float(n_anomalies / len(df) * 100),
        }
        
        return df
    
    def detect_anomalies_isolation_forest(self, features: List[str] = None, contamination: float = 0.05) -> pd.DataFrame:
        """
        Выявление аномалий методом Isolation Forest.
        
        Args:
            features: Список признаков для анализа
            contamination: Ожидаемая доля аномалий
            
        Returns:
            DataFrame с флагами аномалий
        """
        from sklearn.ensemble import IsolationForest
        
        logger.info(f"Isolation Forest анализ (contamination={contamination})")
        
        if features is None:
            features = ['T_out', 'Q']
        
        df = self.df.copy()
        df_clean = df.dropna(subset=features)
        
        if len(df_clean) < 20:
            logger.warning("Недостаточно данных для Isolation Forest")
            df['anomaly_if'] = False
            return df
        
        X = df_clean[features].values
        
        # Обучение модели
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = iso_forest.fit_predict(X)
        
        # -1 = аномалия, 1 = норма
        df.loc[df_clean.index, 'anomaly_if'] = predictions == -1
        df['anomaly_if'] = df['anomaly_if'].fillna(False)
        
        n_anomalies = df['anomaly_if'].sum()
        logger.info(f"Найдено {n_anomalies} аномалий (Isolation Forest)")
        
        self.models['isolation_forest'] = iso_forest
        self.results['anomalies_if'] = {
            'count': int(n_anomalies),
            'percentage': float(n_anomalies / len(df) * 100),
        }
        
        return df
    
    def detect_anomalies_grubbs(self, column: str = 'Q', alpha: float = 0.05) -> pd.DataFrame:
        """
        Тест Граббса для выявления выбросов.
        
        Args:
            column: Анализируемая колонка
            alpha: Уровень значимости
            
        Returns:
            DataFrame с флагами аномалий
        """
        from scipy import stats
        
        logger.info(f"Тест Граббса для {column} (alpha={alpha})")
        
        df = self.df.copy()
        values = df[column].dropna().values
        
        if len(values) < 3:
            logger.warning("Недостаточно данных для теста Граббса")
            df['anomaly_grubbs'] = False
            return df
        
        # Статистика Граббса
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            df['anomaly_grubbs'] = False
            return df
        
        # G-статистика для каждого значения
        g_values = np.abs(values - mean) / std
        g_critical = stats.t.ppf(1 - alpha / (2 * len(values)), len(values) - 2)
        g_critical = g_critical * np.sqrt(len(values) - 1) / np.sqrt(len(values) - 2 + g_critical**2)
        
        # Флаги аномалий
        anomaly_mask = g_values > g_critical
        
        df.loc[df[column].notna(), 'anomaly_grubbs'] = anomaly_mask
        df['anomaly_grubbs'] = df['anomaly_grubbs'].fillna(False)
        
        n_anomalies = df['anomaly_grubbs'].sum()
        logger.info(f"Найдено {n_anomalies} аномалий (Граббс)")
        
        self.results['anomalies_grubbs'] = {
            'count': int(n_anomalies),
            'critical_value': float(g_critical),
        }
        
        return df
    
    # =========================================================================
    # 3. КЛАСТЕРИЗАЦИЯ МКД
    # =========================================================================
    
    def cluster_buildings(self, features: List[str] = None, n_clusters: int = 4) -> Dict:
        """
        Кластеризация МКД методом K-Means.
        
        Args:
            features: Признаки для кластеризации
            n_clusters: Количество кластеров
            
        Returns:
            Словарь с результатами кластеризации
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        logger.info(f"Кластеризация K-Means (k={n_clusters})")
        
        if features is None:
            features = ['avg_T_out', 'avg_Q', 'GSOP_total', 'consumption_per_degree']
        
        # Группировка по МКД (если есть несколько зданий)
        if 'mkd_id' in self.df.columns:
            df_grouped = self.df.groupby('mkd_id').agg({
                'T_out': 'mean',
                'Q': 'mean',
                'GSOP_daily': 'sum',
            }).rename(columns={
                'T_out': 'avg_T_out',
                'Q': 'avg_Q',
                'GSOP_daily': 'GSOP_total',
            })
            
            # Добавляем признак расхода на градус
            df_grouped['consumption_per_degree'] = (
                df_grouped['avg_Q'] / np.maximum(0.1, 18 - df_grouped['avg_T_out'])
            )
        else:
            # Если одно здание, создаём фиктивную группировку по месяцам
            logger.info("Группировка по месяцам (одно здание)")
            df_grouped = self.df.copy()
            if 'month' in df_grouped.columns:
                df_grouped = df_grouped.groupby('month').agg({
                    'T_out': 'mean',
                    'Q': 'mean',
                    'GSOP_daily': 'sum',
                }).rename(columns={
                    'T_out': 'avg_T_out',
                    'Q': 'avg_Q',
                    'GSOP_daily': 'GSOP_total',
                })
        
        # Подготовка данных
        available_features = [f for f in features if f in df_grouped.columns]
        
        if len(available_features) < 2:
            logger.warning("Недостаточно признаков для кластеризации")
            return {'error': 'insufficient_features'}
        
        X = df_grouped[available_features].dropna()
        
        if len(X) < n_clusters:
            logger.warning("Меньше объектов чем кластеров")
            return {'error': 'too_few_objects'}
        
        # Нормализация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Силуэт
        silhouette = silhouette_score(X_scaled, labels)
        
        # Центроиды
        centroids = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=available_features,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )
        
        results = {
            'labels': labels,
            'cluster_counts': pd.Series(labels).value_counts().to_dict(),
            'silhouette_score': float(silhouette),
            'centroids': centroids.to_dict(),
            'scaler': scaler,
        }
        
        self.models['kmeans'] = kmeans
        self.results['clustering'] = results
        
        logger.info(f"Кластеризация завершена: силуэт={silhouette:.3f}")
        
        return results
    
    # =========================================================================
    # 4. ПРОГНОЗИРОВАНИЕ (Holt-Winters)
    # =========================================================================
    
    def forecast_holt_winters(
        self,
        target: str = 'Q',
        periods: int = 14,
        seasonal: int = 7
    ) -> Dict:
        """
        Прогнозирование методом Холта-Винтерса (тройное экспоненциальное сглаживание).
        
        Args:
            target: Целевая переменная
            periods: Горизонт прогноза (дней)
            seasonal: Сезонность (7 для недельной)
            
        Returns:
            Словарь с прогнозом
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        logger.info(f"Прогноз Холта-Винтерса: {periods} дней, сезонность={seasonal}")
        
        df = self.df.copy()
        
        if len(df) < seasonal * 2:
            logger.warning("Недостаточно данных для сезонного прогноза")
            return {'error': 'insufficient_data'}
        
        # Временной ряд
        series = df[target].dropna()
        
        if len(series) < seasonal * 2:
            return {'error': 'insufficient_valid_data'}
        
        # Модель
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal,
        )
        
        fitted = model.fit()
        forecast = fitted.forecast(periods)
        
        # Доверительные интервалы (упрощённо)
        residuals = fitted.fittedvalues - series
        std_err = residuals.std()
        
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'lower': forecast - 1.96 * std_err,
            'upper': forecast + 1.96 * std_err,
        })
        
        results = {
            'forecast': forecast_df.to_dict('list'),
            'fitted_values': fitted.fittedvalues.tolist(),
            'parameters': {
                'level': float(fitted.params['smoothing_level']),
                'trend': float(fitted.params['smoothing_trend']),
                'seasonal': float(fitted.params['smoothing_seasonal']),
            },
            'aic': float(fitted.aic),
        }
        
        self.models['holt_winters'] = fitted
        self.results['forecast'] = results
        
        logger.info(f"Прогноз построен: AIC={results['aic']:.2f}")
        
        return results
    
    # =========================================================================
    # 5. ОЦЕНКА ЭФФЕКТИВНОСТИ
    # =========================================================================
    
    def evaluate_efficiency(self, normative_consumption: float = None) -> Dict:
        """
        Оценка эффективности теплопотребления.
        Нормализация к базовым ГСОП и сравнение с нормативом.
        
        Args:
            normative_consumption: Нормативный расход (Гкал/м² за отопительный период)
            
        Returns:
            Словарь со статусом эффективности
        """
        logger.info("Оценка эффективности теплопотребления")
        
        df = self.df.copy()
        
        # Расчёт удельного расхода
        total_GSOP = df['GSOP_daily'].sum()
        total_Q = df['Q'].sum()
        
        if total_GSOP == 0:
            return {'error': 'zero_gsop'}
        
        # Удельный расход на ГСОП
        specific_consumption = total_Q / total_GSOP
        
        # Статус
        if normative_consumption is not None:
            ratio = specific_consumption / normative_consumption
            
            if ratio < 0.9:
                status = 'Эффективный'
            elif ratio < 1.1:
                status = 'Нормативный'
            elif ratio < 1.3:
                status = 'Предупреждение'
            else:
                status = 'Критический'
        else:
            # Эвристическая оценка
            if specific_consumption < 0.05:
                status = 'Эффективный'
            elif specific_consumption < 0.08:
                status = 'Нормативный'
            elif specific_consumption < 0.12:
                status = 'Предупреждение'
            else:
                status = 'Критический'
            
            ratio = None
        
        results = {
            'total_GSOP': float(total_GSOP),
            'total_consumption': float(total_Q),
            'specific_consumption': float(specific_consumption),
            'normative_ratio': float(ratio) if ratio else None,
            'status': status,
            'recommendations': self._get_recommendations(status, specific_consumption),
        }
        
        self.results['efficiency'] = results
        
        logger.info(f"Статус эффективности: {status}")
        
        return results
    
    def _get_recommendations(self, status: str, specific_consumption: float) -> List[str]:
        """
        Генерация рекомендаций на основе статуса.
        """
        recommendations = []
        
        if status == 'Эффективный':
            recommendations.append("Продолжайте мониторинг, текущие показатели отличные.")
        elif status == 'Нормативный':
            recommendations.append("Показатели в норме. Рекомендуется плановое обслуживание.")
        elif status == 'Предупреждение':
            recommendations.extend([
                "Проверьте настройки элеваторного узла.",
                "Рекомендуется провести энергоаудит.",
                "Возможна разбалансировка системы отопления."
            ])
        elif status == 'Критический':
            recommendations.extend([
                "СРОЧНО: Проверьте наличие утечек в системе.",
                "Возможен перерасход тепла из-за неисправности оборудования.",
                "Требуется внеплановая диагностика всех узлов.",
                "Рассмотрите вопрос модернизации системы отопления."
            ])
        
        return recommendations
    
    # =========================================================================
    # 6. СОХРАНЕНИЕ/ЗАГРУЗКА МОДЕЛЕЙ
    # =========================================================================
    
    def save_models(self, filepath: str) -> None:
        """Сохранение обученных моделей."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, filepath)
        logger.info(f"Модели сохранены: {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Загрузка ранее обученных моделей."""
        self.models = joblib.load(filepath)
        logger.info(f"Модели загружены: {filepath}")
    
    def get_full_report(self) -> Dict:
        """Получение полного отчёта по анализу."""
        return {
            'data_shape': self.df.shape,
            'models_trained': list(self.models.keys()),
            'results': self.results,
        }


if __name__ == "__main__":
    # Демонстрация работы модуля
    import sys
    sys.path.insert(0, '/workspace/src')
    from loader import load_sample_data
    from preprocess import DataPreprocessor
    
    logger.add("logs/analytics.log", rotation="1 MB")
    
    # Загрузка и предобработка
    df = load_sample_data()
    preprocessor = DataPreprocessor(df)
    df = preprocessor.calculate_gsop()
    df = preprocessor.add_features()
    
    # Анализ
    engine = AnalyticsEngine(df)
    
    print("\n" + "="*60)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("="*60)
    
    # Регрессия
    reg_results = engine.fit_linear_regression()
    print(f"\nРЕГРЕССИЯ:")
    print(f"  R² = {reg_results['metrics']['r2']:.3f}")
    print(f"  Коэффициент при T_out: {reg_results['coefficients'].get('T_out', 'N/A')}")
    
    # Аномалии
    df_with_anomalies = engine.detect_anomalies_ewma()
    print(f"\nАНОМАЛИИ (EWMA): {engine.results['anomalies_ewma']['count']}")
    
    # Эффективность
    eff_results = engine.evaluate_efficiency()
    print(f"\nЭФФЕКТИВНОСТЬ:")
    print(f"  Статус: {eff_results['status']}")
    print(f"  Удельный расход: {eff_results['specific_consumption']:.4f}")
    
    # Прогноз
    forecast_results = engine.forecast_holt_winters(periods=7)
    if 'error' not in forecast_results:
        print(f"\nПРОГНОЗ на 7 дней:")
        print(f"  Среднее значение: {np.mean(forecast_results['forecast']['forecast']):.3f}")
    
    print("\n" + "="*60 + "\n")
    
    # Сохранение моделей
    engine.save_models('models/analytics_models.joblib')
    print("Модели сохранены в models/analytics_models.joblib")
