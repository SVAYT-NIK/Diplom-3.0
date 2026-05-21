"""
Модуль загрузки данных из Excel файлов.
Поддерживает чтение .xlsx/.xls файлов с данными по МКД.
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple


class DataLoader:
    """Класс для загрузки и первичной валидации данных из Excel."""
    
    # Допустимые диапазоны значений (СП 50.13330.2012)
    VALID_RANGES = {
        'T_out': (-40, 40),      # Температура наружного воздуха
        'T_in': (18, 26),        # Температура внутри помещения
        'Q': (0, 1000),          # Тепловая нагрузка (Гкал/сутки)
        'V': (0, 10000),         # Объем потребления (м³)
    }
    
    def __init__(self, filepath: str):
        """
        Инициализация загрузчика данных.
        
        Args:
            filepath: Путь к Excel файлу
        """
        self.filepath = Path(filepath)
        self.df: Optional[pd.DataFrame] = None
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {self.filepath}")
        
        logger.info(f"Инициализация загрузчика для файла: {self.filepath}")
    
    def load(self, sheet_name: int | str = 0) -> pd.DataFrame:
        """
        Загрузка данных из Excel файла.
        
        Args:
            sheet_name: Имя или номер листа
            
        Returns:
            DataFrame с загруженными данными
        """
        logger.info(f"Загрузка данных из листа: {sheet_name}")
        
        try:
            self.df = pd.read_excel(
                self.filepath,
                sheet_name=sheet_name,
                engine='openpyxl',
                parse_dates=['date'] if 'date' in pd.read_excel(self.filepath, nrows=1).columns else None
            )
            logger.info(f"Успешно загружено {len(self.df)} строк, {len(self.df.columns)} колонок")
            return self.df
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            raise
    
    def validate_ranges(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Валидация диапазонов значений.
        Помечает выбросы статусом 'invalid'.
        
        Args:
            df: DataFrame для валидации (по умолчанию self.df)
            
        Returns:
            Кортеж (валидные данные, невалидные данные)
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("Данные не загружены. Вызовите load() primero.")
        
        logger.info("Начало валидации диапазонов")
        
        # Создаем копию для маркировки
        df_validated = df.copy()
        df_validated['status'] = 'valid'
        
        invalid_mask = pd.Series([False] * len(df_validated))
        
        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col in df_validated.columns:
                col_invalid = (df_validated[col] < min_val) | (df_validated[col] > max_val)
                invalid_mask |= col_invalid
                
                if col_invalid.any():
                    logger.warning(f"Найдено {col_invalid.sum()} выбросов в колонке {col} (диапазон: {min_val}-{max_val})")
        
        df_validated.loc[invalid_mask, 'status'] = 'invalid'
        
        valid_df = df_validated[df_validated['status'] == 'valid'].copy()
        invalid_df = df_validated[df_validated['status'] == 'invalid'].copy()
        
        logger.info(f"Валидация завершена: {len(valid_df)} валидных, {len(invalid_df)} невалидных записей")
        
        return valid_df, invalid_df
    
    def get_summary(self) -> dict:
        """
        Получение сводной информации о загруженных данных.
        
        Returns:
            Словарь со статистикой
        """
        if self.df is None:
            return {}
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            'null_counts': self.df.isnull().sum().to_dict(),
        }
        
        # Статистика по числовым колонкам
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = self.df[numeric_cols].describe().to_dict()
        
        return summary


def load_sample_data() -> pd.DataFrame:
    """
    Создание тестовых данных для демонстрации.
    
    Returns:
        DataFrame с синтетическими данными
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.info("Генерация тестовых данных")
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Синтетические данные для одного МКД
    np.random.seed(42)
    
    # Температура наружного воздуха (сезонная зависимость)
    day_of_year = dates.dayofyear
    T_out = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 3, n_days)
    
    # Тепловая нагрузка (обратная зависимость от температуры)
    Q = np.maximum(0, 1.5 - 0.08 * T_out) + np.random.normal(0, 0.2, n_days)
    Q = np.clip(Q, 0, 3)
    
    # Температура внутри (стабильная)
    T_in = 22 + np.random.normal(0, 1, n_days)
    
    df = pd.DataFrame({
        'mkd_id': 'MKD-001',
        'date': dates,
        'T_out': T_out,
        'T_in': T_in,
        'Q': Q,
    })
    
    # Добавим несколько выбросов
    df.loc[df.sample(5).index, 'T_out'] = 50  # Аномальная жара
    df.loc[df.sample(3).index, 'Q'] = -0.5    # Отрицательная нагрузка
    
    logger.info(f"Сгенерировано {len(df)} записей для тестирования")
    
    return df


if __name__ == "__main__":
    # Демонстрация работы модуля
    logger.add("logs/loader.log", rotation="1 MB")
    
    # Генерация и сохранение тестовых данных
    test_df = load_sample_data()
    test_df.to_excel("data/sample_data.xlsx", index=False)
    logger.info("Тестовые данные сохранены в data/sample_data.xlsx")
    
    # Загрузка и валидация
    loader = DataLoader("data/sample_data.xlsx")
    df = loader.load()
    
    valid_df, invalid_df = loader.validate_ranges()
    
    print(f"\n{'='*50}")
    print(f"ВСЕГО ЗАПИСЕЙ: {len(df)}")
    print(f"ВАЛИДНЫХ: {len(valid_df)}")
    print(f"НЕВАЛИДНЫХ: {len(invalid_df)}")
    print(f"{'='*50}\n")
    
    if len(invalid_df) > 0:
        print("Примеры невалидных записей:")
        print(invalid_df.head())
