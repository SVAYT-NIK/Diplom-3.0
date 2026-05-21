"""
Report generator for PDF and CSV exports.
"""
from typing import List, Any
from datetime import datetime
import io


def generate_pdf_report(building: Any, results: List[Any], run_id: str) -> bytes:
    """
    Generate a PDF report with analysis results.

    Args:
        building: Building ORM object
        results: List of AnalysisResult ORM objects
        run_id: Analysis run ID

    Returns:
        PDF file as bytes
    """
    # Simple HTML template for the report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Heat Analytics Report - {run_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
            .anomaly {{ color: #e74c3c; font-weight: bold; }}
            .normal {{ color: #27ae60; }}
        </style>
    </head>
    <body>
        <h1>🔥 Отчёт по анализу теплопотребления</h1>
        
        <div class="summary">
            <h2>Информация о здании</h2>
            <p><strong>Адрес:</strong> {building.address if building else 'N/A'}</p>
            <p><strong>Площадь:</strong> {building.area_m2 if building else 'N/A'} м²</p>
            <p><strong>Год постройки:</strong> {building.year_built if building and building.year_built else 'N/A'}</p>
            <p><strong>ID анализа:</strong> {run_id}</p>
            <p><strong>Дата генерации:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>

        <h2>Статистика анализа</h2>
        <table>
            <tr>
                <th>Метрика</th>
                <th>Значение</th>
            </tr>
            <tr>
                <td>Всего записей</td>
                <td>{len(results)}</td>
            </tr>
            <tr>
                <td>Аномалий обнаружено</td>
                <td>{sum(1 for r in results if r.anomaly_flag)}</td>
            </tr>
            <tr>
                <td>Процент аномалий</td>
                <td>{(sum(1 for r in results if r.anomaly_flag) / len(results) * 100) if results else 0:.2f}%</td>
            </tr>
        </table>

        <h2>Результаты анализа</h2>
        <table>
            <tr>
                <th>Модель</th>
                <th>Прогноз Q (Гкал)</th>
                <th>Остаток</th>
                <th>Скоринг аномалии</th>
                <th>Класс эффективности</th>
                <th>Отклонение от нормы (%)</th>
            </tr>
    """

    # Add sample rows (first 20)
    for r in results[:20]:
        anomaly_class = "anomaly" if r.anomaly_flag else "normal"
        html_content += f"""
            <tr>
                <td>{r.model_type}</td>
                <td>{r.predicted_q:.4f if r.predicted_q else 'N/A'}</td>
                <td>{r.residual:.4f if r.residual else 'N/A'}</td>
                <td class="{anomaly_class}">{r.anomaly_score:.4f if r.anomaly_score else 'N/A'}</td>
                <td>{r.efficiency_class or 'N/A'}</td>
                <td>{r.norm_deviation_pct:.2f if r.norm_deviation_pct else 'N/A'}</td>
            </tr>
        """

    html_content += """
        </table>

        <div style="margin-top: 40px; font-size: 12px; color: #7f8c8d;">
            <p>Отчёт сгенерирован автоматически системой Heat Analytics.</p>
            <p>Для получения полной информации обратитесь к веб-интерфейсу системы.</p>
        </div>
    </body>
    </html>
    """

    try:
        from weasyprint import HTML
        pdf_buffer = HTML(string=html_content).write_pdf()
        return pdf_buffer
    except ImportError:
        # Fallback if weasyprint is not installed
        return html_content.encode("utf-8")


def generate_csv_report(building: Any, results: List[Any], run_id: str) -> str:
    """
    Generate a CSV report with analysis results.

    Args:
        building: Building ORM object
        results: List of AnalysisResult ORM objects
        run_id: Analysis run ID

    Returns:
        CSV content as string
    """
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Heat Analytics Report"])
    writer.writerow([f"Run ID: {run_id}"])
    writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"])
    writer.writerow([])

    # Building info
    writer.writerow(["Building Information"])
    writer.writerow(["Address", building.address if building else "N/A"])
    writer.writerow(["Area (m²)", building.area_m2 if building else "N/A"])
    writer.writerow(["Year Built", building.year_built if building and building.year_built else "N/A"])
    writer.writerow([])

    # Statistics
    writer.writerow(["Analysis Statistics"])
    writer.writerow(["Total Records", len(results)])
    writer.writerow(["Anomalies Detected", sum(1 for r in results if r.anomaly_flag)])
    writer.writerow(
        ["Anomaly Percentage", f"{(sum(1 for r in results if r.anomaly_flag) / len(results) * 100) if results else 0:.2f}%"]
    )
    writer.writerow([])

    # Results data
    writer.writerow(["Analysis Results"])
    writer.writerow([
        "ID",
        "Model Type",
        "Predicted Q (Gcal)",
        "Residual",
        "Anomaly Score",
        "Anomaly Flag",
        "Cluster ID",
        "Efficiency Class",
        "Norm Deviation (%)",
    ])

    for r in results:
        writer.writerow([
            r.id,
            r.model_type,
            r.predicted_q if r.predicted_q is not None else "",
            r.residual if r.residual is not None else "",
            r.anomaly_score if r.anomaly_score is not None else "",
            r.anomaly_flag,
            r.cluster_id if r.cluster_id is not None else "",
            r.efficiency_class if r.efficiency_class else "",
            r.norm_deviation_pct if r.norm_deviation_pct is not None else "",
        ])

    return output.getvalue()
