"""
Модуль предобработки данных.
Включает интерполяцию пропусков, расчёт ГСОП, синхронизацию частоты.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
from datetime import datetime


class DataPreprocessor:
    """Класс для предобработки и нормализации данных по МКД."""
    
    # Базовая температура для расчёта ГСОП (СП 50.13330.2012)
    BASE_TEMP = 18.0
    
    # Порог для длительных пропусков (часы)
    LONG_GAP_THRESHOLD = 24
    
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация препроцессора.
        
        Args:
            df: DataFrame с данными
        """
        self.df = df.copy()
        self.original_df = df.copy()
        logger.info("Инициализация препроцессора данных")
    
    def handle_missing_values(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        Обработка пропущенных значений.
        
        Args:
            method: Метод обработки ('interpolate', 'forward_fill', 'model')
            
        Returns:
            DataFrame с обработанными пропусками
        """
        logger.info(f"Обработка пропусков методом: {method}")
        
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        initial_nulls = df[numeric_cols].isnull().sum().sum()
        
        if method == 'interpolate':
            # Линейная интерполяция для числовых колонок
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit_area='inside')
                
        elif method == 'forward_fill':
            # Заполнение вперёд (для временных рядов)
            for col in numeric_cols:
                df[col] = df[col].fillna(method='ffill')
                
        elif method == 'model':
            # Модельная импутация (регрессия на основе других признаков)
            df = self._model_imputation(df)
        
        final_nulls = df[numeric_cols].isnull().sum().sum()
        logger.info(f"Пропусков до: {initial_nulls}, после: {final_nulls}")
        
        self.df = df
        return df
    
    def _model_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Модельная импутация пропусков на основе исторической регрессии.
        Используется для пропусков > 24 часов.
        """
        from sklearn.linear_model import LinearRegression
        
        logger.info("Выполнение модельной импутации")
        
        # Для простоты используем корреляцию с T_out для импутации Q
        if 'Q' in df.columns and 'T_out' in df.columns:
            mask = df['Q'].isnull()
            
            if mask.any():
                # Обучаем на валидных данных
                train_df = df.dropna(subset=['Q', 'T_out'])
                
                if len(train_df) > 10:
                    X_train = train_df[['T_out']]
                    y_train = train_df['Q']
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Предсказываем пропущенные значения
                    X_missing = df.loc[mask, ['T_out']]
                    df.loc[mask, 'Q'] = model.predict(X_missing)
                    
                    logger.info(f"Импортировано {mask.sum()} значений Q через регрессию")
        
        return df
    
    def calculate_gsop(self, temp_column: str = 'T_out') -> pd.DataFrame:
        """
        Расчёт Градусо-суток отопительного периода (ГСОП).
        По СП 50.13330.2012: ГСОП = Σ(max(0, T_base - T_out))
        
        Args:
            temp_column: Название колонки с температурой
            
        Returns:
            DataFrame с добавленной колонкой GSOP
        """
        logger.info(f"Расчёт ГСОП по колонке {temp_column} (база: {self.BASE_TEMP}°C)")
        
        if temp_column not in self.df.columns:
            raise ValueError(f"Колонка {temp_column} не найдена")
        
        df = self.df.copy()
        
        # Суточный ГСОП
        df['GSOP_daily'] = np.maximum(0, self.BASE_TEMP - df[temp_column])
        
        # Накопительный ГСОП (с начала отопительного периода)
        df['GSOP_cumulative'] = df['GSOP_daily'].cumsum()
        
        logger.info(f"ГСОП рассчитан: средний суточный = {df['GSOP_daily'].mean():.2f}")
        
        self.df = df
        return df
    
    def resample_to_daily(self, agg_methods: Optional[dict] = None) -> pd.DataFrame:
        """
        Синхронизация данных к суточной частоте.
        
        Args:
            agg_methods: Словарь методов агрегации по колонкам
            
        Returns:
            DataFrame с суточными данными
        """
        logger.info("Синхронизация к суточной частоте")
        
        df = self.df.copy()
        
        if 'date' not in df.columns:
            raise ValueError("Колонка 'date' обязательна для ресемплинга")
        
        df = df.set_index('date')
        
        # Методы агрегации по умолчанию
        if agg_methods is None:
            agg_methods = {
                'T_out': 'mean',
                'T_in': 'mean',
                'Q': 'sum',
                'GSOP_daily': 'sum',
            }
        
        # Агрегация только существующих колонок
        existing_aggs = {k: v for k, v in agg_methods.items() if k in df.columns}
        
        if existing_aggs:
            df_daily = df.resample('D').agg(existing_aggs)
        else:
            df_daily = df.resample('D').mean()
        
        df_daily = df_daily.reset_index()
        
        logger.info(f"Ресемплинг завершён: {len(df_daily)} суток")
        
        self.df = df_daily
        return df_daily
    
    def add_features(self) -> pd.DataFrame:
        """
        Добавление дополнительных признаков для анализа.
        
        Returns:
            DataFrame с новыми признаками
        """
        logger.info("Добавление дополнительных признаков")
        
        df = self.df.copy()
        
        if 'date' in df.columns:
            # Временные признаки
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day_of_year'] = df['date'].dt.dayofyear
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['season'] = ((df['month'] % 12 + 3) // 3).map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'})
        
        if 'T_out' in df.columns:
            # Температурные бины
            df['temp_bin'] = pd.cut(
                df['T_out'],
                bins=[-np.inf, -10, 0, 10, 18, np.inf],
                labels=['very_cold', 'cold', 'cool', 'mild', 'warm']
            )
        
        if 'Q' in df.columns and 'T_out' in df.columns:
            # Удельный расход на градус разницы
            df['Q_per_degree'] = df.apply(
                lambda row: row['Q'] / max(0.1, self.BASE_TEMP - row['T_out'])
                if row['T_out'] < self.BASE_TEMP else np.nan,
                axis=1
            )
        
        self.df = df
        logger.info(f"Добавлено признаков: {len(df.columns) - len(self.original_df.columns)}")
        
        return df
    
    def save_to_sqlite(self, db_path: str, table_name: str = 'mkd_data') -> None:
        """
        Сохранение обработанных данных в SQLite.
        
        Args:
            db_path: Путь к базе данных
            table_name: Имя таблицы
        """
        import sqlite3
        
        logger.info(f"Сохранение данных в SQLite: {db_path}")
        
        conn = sqlite3.connect(db_path)
        
        try:
            self.df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Создание индекса
            cursor = conn.cursor()
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_mkd_date ON {table_name}(mkd_id, date)")
            conn.commit()
            
            logger.info(f"Данные сохранены: {len(self.df)} записей в таблицу {table_name}")
            
        finally:
            conn.close()
    
    def get_preprocessing_report(self) -> dict:
        """
        Отчёт о предобработке.
        
        Returns:
            Словарь со статистикой предобработки
        """
        report = {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'columns_added': list(set(self.df.columns) - set(self.original_df.columns)),
            'null_counts_before': self.original_df.isnull().sum().to_dict(),
            'null_counts_after': self.df.isnull().sum().to_dict(),
        }
        
        if 'GSOP_daily' in self.df.columns:
            report['gsop_stats'] = {
                'total': self.df['GSOP_daily'].sum(),
                'daily_avg': self.df['GSOP_daily'].mean(),
                'daily_max': self.df['GSOP_daily'].max(),
            }
        
        return report


if __name__ == "__main__":
    # Демонстрация работы модуля
    import sys
    sys.path.insert(0, '/workspace/src')
    from loader import load_sample_data
    
    logger.add("logs/preprocess.log", rotation="1 MB")
    
    # Загрузка тестовых данных
    df = load_sample_data()
    
    # Предобработка
    preprocessor = DataPreprocessor(df)
    df_processed = preprocessor.handle_missing_values()
    df_processed = preprocessor.calculate_gsop()
    df_processed = preprocessor.add_features()
    
    # Отчёт
    report = preprocessor.get_preprocessing_report()
    
    print(f"\n{'='*50}")
    print("ОТЧЁТ О ПРЕДОБРАБОТКЕ")
    print(f"{'='*50}")
    print(f"Исходный размер: {report['original_shape']}")
    print(f"Итоговый размер: {report['current_shape']}")
    print(f"Добавленные колонки: {report['columns_added']}")
    
    if 'gsop_stats' in report:
        print(f"\nГСОП:")
        print(f"  Общий: {report['gsop_stats']['total']:.2f}")
        print(f"  Средний суточный: {report['gsop_stats']['daily_avg']:.2f}")
        print(f"  Максимальный суточный: {report['gsop_stats']['daily_max']:.2f}")
    
    print(f"{'='*50}\n")
    
    # Сохранение в SQLite
    preprocessor.save_to_sqlite('db/mkd_analytics.db')
    print("Данные сохранены в db/mkd_analytics.db")
