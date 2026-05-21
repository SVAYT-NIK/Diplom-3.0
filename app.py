"""
Веб-интерфейс системы аналитики МКД на Streamlit.
Интерактивная визуализация, отчёты и экспорт данных.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.loader import DataLoader, load_sample_data
from src.preprocess import DataPreprocessor
from src.analytics import AnalyticsEngine

# =============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# =============================================================================

st.set_page_config(
    page_title="Система аналитики МКД",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# БОКОВАЯ ПАНЕЛЬ
# =============================================================================

st.sidebar.title("⚙️ Настройки")
st.sidebar.markdown("---")

# Загрузка данных
upload_option = st.sidebar.radio(
    "Источник данных:",
    ["Тестовые данные", "Загрузить Excel", "Из базы данных"]
)

df = None

if upload_option == "Тестовые данные":
    if st.sidebar.button("🔄 Сгенерировать тестовые данные"):
        with st.spinner("Генерация данных..."):
            df = load_sample_data()
            st.sidebar.success(f"Сгенерировано {len(df)} записей")

elif upload_option == "Загрузить Excel":
    uploaded_file = st.sidebar.file_uploader(
        "Выберите Excel файл",
        type=["xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        # Сохраняем временный файл
        temp_path = Path("data/temp_upload.xlsx")
        temp_path.parent.mkdir(exist_ok=True)
        temp_path.write_bytes(uploaded_file.getvalue())
        
        try:
            loader = DataLoader(str(temp_path))
            df = loader.load()
            st.sidebar.success(f"Загружено {len(df)} строк")
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки: {e}")

elif upload_option == "Из базы данных":
    db_path = st.sidebar.text_input("Путь к БД:", value="db/mkd_analytics.db")
    if Path(db_path).exists():
        if st.sidebar.button("📊 Загрузить из БД"):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM mkd_data", conn)
                conn.close()
                st.sidebar.success(f"Загружено {len(df)} записей из БД")
            except Exception as e:
                st.sidebar.error(f"Ошибка: {e}")

# =============================================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# =============================================================================

st.title("🏢 Система анализа теплопотребления МКД")
st.markdown("""
**Версия:** 1.0 | **Стэк:** Python + Pandas + Scikit-learn + Plotly + Streamlit
""")

if df is None:
    st.info("👈 Выберите источник данных в боковой панели для начала работы")
    st.stop()

# =============================================================================
# ПРЕДОБРАБОТКА
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Предобработка")

run_preprocess = st.sidebar.checkbox("Выполнить предобработку", value=True)

if run_preprocess:
    preprocessor = DataPreprocessor(df)
    df = preprocessor.handle_missing_values()
    df = preprocessor.calculate_gsop()
    df = preprocessor.add_features()
    
    report = preprocessor.get_preprocessing_report()
    
    with st.expander("📋 Отчёт о предобработке"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Записей", report['current_shape'][0])
        col2.metric("Признаков", report['current_shape'][1])
        col3.metric("Добавлено", len(report['columns_added']))
        
        if 'gsop_stats' in report:
            st.write("**ГСОП:**")
            st.write(f"- Общий: {report['gsop_stats']['total']:.2f}")
            st.write(f"- Средний суточный: {report['gsop_stats']['daily_avg']:.2f}")

# =============================================================================
# ВКЛАДКИ
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Данные",
    "📈 Регрессия",
    "🔍 Аномалии",
    "🎯 Кластеризация",
    "🔮 Прогноз",
    "⭐ Эффективность"
])

# -----------------------------------------------------------------------------
# Вкладка 1: Просмотр данных
# -----------------------------------------------------------------------------

with tab1:
    st.header("📊 Исходные данные")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(df.head(100), use_container_width=True)
    
    with col2:
        st.subheader("Статистика")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.write(df[numeric_cols].describe())
    
    # Графики временных рядов
    st.subheader("📈 Временные ряды")
    
    if 'date' in df.columns and 'T_out' in df.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['T_out'],
            name='T наружная',
            line=dict(color='blue', width=2)
        ))
        
        if 'T_in' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['T_in'],
                name='T внутренняя',
                line=dict(color='red', width=2)
            ))
        
        if 'Q' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['Q'],
                name='Тепловая нагрузка (Q)',
                line=dict(color='green', width=2),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Динамика показателей",
            xaxis_title="Дата",
            yaxis_title="Температура (°C)",
            yaxis2=dict(title="Q (Гкал)", overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Вкладка 2: Регрессионный анализ
# -----------------------------------------------------------------------------

with tab2:
    st.header("📈 Регрессионный анализ")
    
    st.markdown("""
    **Модель:** Q = α + β·T_out + ε
    
    Где:
    - Q — тепловая нагрузка (Гкал/сутки)
    - T_out — температура наружного воздуха (°C)
    - α, β — коэффициенты регрессии
    - ε — остатки модели
    """)
    
    # Инициализация аналитического движка
    engine = AnalyticsEngine(df)
    
    # Линейная регрессия
    if st.button("🚀 Построить регрессию"):
        with st.spinner("Обучение моделей..."):
            reg_results = engine.fit_linear_regression()
            
            if 'error' not in reg_results:
                col1, col2, col3 = st.columns(3)
                
                metrics = reg_results['metrics']
                coeffs = reg_results['coefficients']
                
                col1.metric("R²", f"{metrics['r2']:.3f}")
                col2.metric("RMSE", f"{metrics['rmse']:.3f}")
                col3.metric("CV R² (среднее)", f"{metrics['cv_r2_mean']:.3f}")
                
                st.subheader("Коэффициенты")
                st.write(f"**Intercept (α):** {coeffs.get('intercept', 'N/A'):.4f}")
                st.write(f"**Coefficient при T_out (β):** {coeffs.get('T_out', 'N/A'):.4f}")
                
                # График регрессии
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['T_out'],
                    y=df['Q'],
                    mode='markers',
                    name='Фактические данные',
                    marker=dict(size=8, opacity=0.6)
                ))
                
                # Линия регрессии
                x_range = np.linspace(df['T_out'].min(), df['T_out'].max(), 100)
                y_pred = coeffs['intercept'] + coeffs['T_out'] * x_range
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name='Линейная регрессия',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title="Зависимость Q от T_out",
                    xaxis_title="T_out (°C)",
                    yaxis_title="Q (Гкал)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Квантильная регрессия
                st.subheader("Доверительные интервалы (квантильная регрессия)")
                
                if st.checkbox("Показать квантили"):
                    quant_results = engine.fit_quantile_regression()
                    
                    if 'error' not in quant_results:
                        fig = go.Figure()
                        
                        colors = ['blue', 'green', 'red']
                        for i, (key, val) in enumerate(quant_results.items()):
                            y_q = val['intercept'] + val['coefficient'] * x_range
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_q,
                                mode='lines',
                                name=f"Квантиль {key}",
                                line=dict(color=colors[i], dash='dash')
                            ))
                        
                        fig.update_layout(
                            title="Квантильная регрессия",
                            xaxis_title="T_out (°C)",
                            yaxis_title="Q (Гкал)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Вкладка 3: Выявление аномалий
# -----------------------------------------------------------------------------

with tab3:
    st.header("🔍 Выявление аномалий")
    
    st.markdown("""
    Методы обнаружения выбросов:
    - **EWMA** — экспоненциально взвешенное скользящее среднее
    - **Isolation Forest** — метод изоляции леса
    - **Тест Граббса** — статистический тест на выбросы
    """)
    
    engine = AnalyticsEngine(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Метод:",
            ["EWMA", "Isolation Forest", "Тест Граббса"]
        )
        
        if st.button("🔍 Найти аномалии"):
            with st.spinner("Анализ..."):
                if method == "EWMA":
                    df_anom = engine.detect_anomalies_ewma()
                    anom_col = 'anomaly_ewma'
                elif method == "Isolation Forest":
                    df_anom = engine.detect_anomalies_isolation_forest()
                    anom_col = 'anomaly_if'
                else:
                    df_anom = engine.detect_anomalies_grubbs()
                    anom_col = 'anomaly_grubbs'
                
                n_anomalies = df_anom[anom_col].sum()
                st.metric("Найдено аномалий", int(n_anomalies))
                
                if n_anomalies > 0:
                    st.subheader("Аномальные записи:")
                    anomalies = df_anom[df_anom[anom_col]]
                    st.dataframe(anomalies[['date', 'T_out', 'Q', anom_col]].head(20))
    
    with col2:
        if 'anomaly_ewma' in df.columns or 'anomaly_if' in df.columns or 'anomaly_grubbs' in df.columns:
            st.subheader("Визуализация аномалий")
            
            anom_col = next((c for c in ['anomaly_ewma', 'anomaly_if', 'anomaly_grubbs'] if c in df.columns), None)
            
            if anom_col:
                fig = go.Figure()
                
                normal_df = df[~df[anom_col]]
                anomaly_df = df[df[anom_col]]
                
                fig.add_trace(go.Scatter(
                    x=normal_df['date'],
                    y=normal_df['Q'],
                    mode='markers',
                    name='Норма',
                    marker=dict(size=6, color='blue', opacity=0.5)
                ))
                
                if len(anomaly_df) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_df['date'],
                        y=anomaly_df['Q'],
                        mode='markers',
                        name='Аномалия',
                        marker=dict(size=10, color='red', symbol='x')
                    ))
                
                fig.update_layout(
                    title="Аномалии на графике тепловой нагрузки",
                    xaxis_title="Дата",
                    yaxis_title="Q (Гкал)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Вкладка 4: Кластеризация
# -----------------------------------------------------------------------------

with tab4:
    st.header("🎯 Кластеризация МКД")
    
    st.markdown("""
    Группировка объектов по схожим характеристикам потребления тепла.
    
    **Признаки:** средняя температура, среднее потребление, ГСОП, удельный расход
    """)
    
    engine = AnalyticsEngine(df)
    
    n_clusters = st.slider("Количество кластеров:", 2, 6, 4)
    
    if st.button("🎲 Выполнить кластеризацию"):
        with st.spinner("Кластеризация..."):
            cluster_results = engine.cluster_buildings(n_clusters=n_clusters)
            
            if 'error' not in cluster_results:
                col1, col2 = st.columns(2)
                
                col1.metric("Силуэт (качество)", f"{cluster_results['silhouette_score']:.3f}")
                
                col2.write("**Распределение по кластерам:**")
                cluster_counts = pd.Series(cluster_results['cluster_counts'])
                col2.bar_chart(cluster_counts)
                
                # Центроиды
                st.subheader("Центроиды кластеров")
                centroids_df = pd.DataFrame(cluster_results['centroids'])
                st.dataframe(centroids_df)
                
                # Визуализация (если достаточно признаков)
                if 'avg_T_out' in centroids_df.index and 'avg_Q' in centroids_df.index:
                    fig = go.Figure()
                    
                    for i in range(n_clusters):
                        cluster_name = f'Cluster_{i}'
                        fig.add_trace(go.Scatter(
                            x=[centroids_df.loc['avg_T_out', cluster_name]],
                            y=[centroids_df.loc['avg_Q', cluster_name]],
                            mode='markers+text',
                            name=cluster_name,
                            marker=dict(size=20),
                            text=[f'{cluster_results["cluster_counts"].get(i, 0)} objs'],
                            textposition='top center'
                        ))
                    
                    fig.update_layout(
                        title="Центроиды кластеров",
                        xaxis_title="Средняя T_out (°C)",
                        yaxis_title="Среднее Q (Гкал)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Вкладка 5: Прогнозирование
# -----------------------------------------------------------------------------

with tab5:
    st.header("🔮 Прогнозирование (Холт-Винтерс)")
    
    st.markdown("""
    Тройное экспоненциальное сглаживание для прогноза потребления.
    
    Учитывает:
    - Уровень (level)
    - Тренд (trend)
    - Сезонность (seasonal)
    """)
    
    engine = AnalyticsEngine(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Горизонт прогноза (дней):", 7, 30, 14)
    
    with col2:
        seasonal_period = st.slider("Период сезонности:", 7, 30, 7)
    
    if st.button("📊 Построить прогноз"):
        with st.spinner("Прогнозирование..."):
            forecast_results = engine.forecast_holt_winters(
                periods=forecast_days,
                seasonal=seasonal_period
            )
            
            if 'error' not in forecast_results:
                params = forecast_results['parameters']
                
                st.subheader("Параметры модели")
                col1, col2, col3 = st.columns(3)
                col1.metric("Level", f"{params['level']:.3f}")
                col2.metric("Trend", f"{params['trend']:.3f}")
                col3.metric("Seasonal", f"{params['seasonal']:.3f}")
                
                st.write(f"**AIC:** {forecast_results['aic']:.2f}")
                
                # График прогноза
                fig = go.Figure()
                
                # Исторические данные
                historical = df['Q'].values
                dates_hist = df['date'].values if 'date' in df.columns else range(len(historical))
                
                fig.add_trace(go.Scatter(
                    x=dates_hist,
                    y=historical,
                    mode='lines',
                    name='История',
                    line=dict(color='blue', width=2)
                ))
                
                # Прогноз
                forecast_dates = pd.date_range(
                    start=df['date'].max() + pd.Timedelta(days=1),
                    periods=forecast_days
                ) if 'date' in df.columns else range(len(historical), len(historical) + forecast_days)
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_results['forecast']['forecast'],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                # Доверительные интервалы
                fig.add_trace(go.Scatter(
                    x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                    y=(forecast_results['forecast']['upper'] + 
                       forecast_results['forecast']['lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(0,255,0,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% ДИ'
                ))
                
                fig.update_layout(
                    title=f"Прогноз потребления на {forecast_days} дней",
                    xaxis_title="Дата",
                    yaxis_title="Q (Гкал)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Экспорт прогноза
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'forecast': forecast_results['forecast']['forecast'],
                    'lower': forecast_results['forecast']['lower'],
                    'upper': forecast_results['forecast']['upper'],
                })
                
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать прогноз (CSV)",
                    data=csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )

# -----------------------------------------------------------------------------
# Вкладка 6: Оценка эффективности
# -----------------------------------------------------------------------------

with tab6:
    st.header("⭐ Оценка эффективности")
    
    st.markdown("""
    Анализ соответствия фактического потребления нормативным значениям.
    
    **Статусы:**
    - 🟢 Эффективный (< 90% от норматива)
    - 🟡 Нормативный (90-110%)
    - 🟠 Предупреждение (110-130%)
    - 🔴 Критический (> 130%)
    """)
    
    engine = AnalyticsEngine(df)
    
    normative = st.number_input(
        "Нормативное потребление (Гкал/ГСОП):",
        min_value=0.01,
        max_value=1.0,
        value=0.08,
        step=0.01
    )
    
    if st.button("📊 Оценить эффективность"):
        with st.spinner("Анализ..."):
            eff_results = engine.evaluate_efficiency(normative_consumption=normative)
            
            if 'error' not in eff_results:
                status = eff_results['status']
                
                # Иконка статуса
                status_icons = {
                    'Эффективный': '🟢',
                    'Нормативный': '🟡',
                    'Предупреждение': '🟠',
                    'Критический': '🔴'
                }
                
                st.subheader(f"{status_icons.get(status, '')} {status}")
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "Общий ГСОП",
                    f"{eff_results['total_GSOP']:.1f}"
                )
                
                col2.metric(
                    "Общее потребление",
                    f"{eff_results['total_consumption']:.2f} Гкал"
                )
                
                col3.metric(
                    "Удельный расход",
                    f"{eff_results['specific_consumption']:.4f}"
                )
                
                if eff_results['normative_ratio']:
                    st.write(f"**Отношение к нормативу:** {eff_results['normative_ratio']*100:.1f}%")
                
                # Рекомендации
                st.subheader("📋 Рекомендации")
                for rec in eff_results['recommendations']:
                    st.write(f"- {rec}")
                
                # Визуализация
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Фактическое', 'Нормативное'],
                    y=[
                        eff_results['specific_consumption'],
                        normative
                    ],
                    marker_color=['steelblue', 'orange'],
                    text=[
                        f"{eff_results['specific_consumption']:.4f}",
                        f"{normative:.4f}"
                    ],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Сравнение с нормативом",
                    yaxis_title="Удельный расход (Гкал/ГСОП)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# ПОДВАЛ
# =============================================================================

st.markdown("---")
st.caption("""
**Система аналитики теплопотребления МКД v1.0**  
Разработано в рамках магистерской диссертации  
Технологии: Python, Pandas, Scikit-learn, Statsmodels, Plotly, Streamlit
""")

# Кнопка экспорта полного отчёта
if st.button("📥 Экспортировать полный отчёт (JSON)"):
    import json
    
    engine = AnalyticsEngine(df)
    full_report = engine.get_full_report()
    
    json_report = json.dumps(full_report, indent=2, default=str)
    
    st.download_button(
        label="Скачать JSON",
        data=json_report,
        file_name="analytics_report.json",
        mime="application/json"
    )
