"""
Excel file parser for heat consumption data.
Handles various date formats and data cleaning.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class ExcelParser:
    """Parser for Excel files with heat consumption data."""

    # Expected column names (Russian)
    EXPECTED_COLUMNS = [
        "Дата", "Время", "Состояние", "Отключение", "НС", "Время НС",
        "T1", "T2", "P1", "P2", "V1", "V2", "M1", "M2", "Q",
        "d T", "d V", "d M", "Небаланс", "Ти", "Тост", "Тхв",
        "Система сбора данных"
    ]

    # Rows to skip (summary rows)
    SKIP_ROWS = [
        "Итого за период штатной работы",
        "Итого за период НС",
        "Итого",
        "Нештатные ситуации"
    ]

    def __init__(self, file_path: str):
        """
        Initialize parser with file path.

        Args:
            file_path: Path to the Excel file
        """
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse the Excel file and return cleaned data.

        Returns:
            Tuple of (cleaned DataFrame, metadata dict)

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        logger.info("Parsing Excel file", file_path=str(self.file_path))

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Read Excel file, skipping header rows (1-4 contain metadata)
        # Data starts from row 6 (index 5)
        try:
            self.df = pd.read_excel(
                self.file_path,
                sheet_name=0,
                skiprows=5,  # Skip rows 1-5 (0-indexed: 0-4)
                engine="openpyxl"
            )
        except Exception as e:
            logger.error("Failed to read Excel file", error=str(e))
            raise ValueError(f"Failed to read Excel file: {str(e)}")

        if self.df.empty:
            raise ValueError("Excel file contains no data")

        # Extract metadata from header rows (optional)
        self._extract_metadata()

        # Clean and normalize data
        self._clean_data()

        # Validate required columns
        self._validate_columns()

        # Filter out summary rows
        self._filter_summary_rows()

        logger.info(
            "Excel parsing completed",
            rows=len(self.df),
            columns=list(self.df.columns)
        )

        return self.df, self.metadata

    def _extract_metadata(self):
        """Extract metadata from header rows if available."""
        try:
            # Try to read first few rows for metadata
            meta_df = pd.read_excel(
                self.file_path,
                sheet_name=0,
                nrows=4,
                header=None,
                engine="openpyxl"
            )

            # Look for common metadata patterns
            for idx, row in meta_df.iterrows():
                row_str = " ".join(str(v) for v in row.values if pd.notna(v))

                if "отчёт сформирован" in row_str.lower():
                    self.metadata["report_generated"] = row_str
                elif "тепловычислитель" in row_str.lower():
                    self.metadata["heat_computer"] = row_str
                elif "потребитель" in row_str.lower():
                    self.metadata["consumer"] = row_str
                elif "схема" in row_str.lower():
                    self.metadata["scheme"] = row_str

        except Exception as e:
            logger.warning("Failed to extract metadata", error=str(e))

    def _clean_data(self):
        """Clean and normalize the dataframe."""
        # Rename columns to standard names
        column_mapping = {}
        for col in self.df.columns:
            if isinstance(col, str):
                # Strip whitespace
                clean_col = col.strip()
                column_mapping[col] = clean_col

        self.df.rename(columns=column_mapping, inplace=True)

        # Normalize date column
        if "Дата" in self.df.columns:
            self.df["Дата"] = self.df["Дата"].apply(self._normalize_date)

        # Normalize numeric columns - replace comma with dot, handle NaN
        numeric_cols = [
            "T1", "T2", "P1", "P2", "V1", "V2", "M1", "M2", "Q",
            "d T", "d V", "d M", "Небаланс", "Ти", "Тост", "Тхв"
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._normalize_numeric)

        # Parse NS codes (non-standard situations)
        if "НС" in self.df.columns:
            self.df["НС"] = self.df["НС"].apply(self._parse_ns_codes)

        # Convert system type to string
        if "Система сбора данных" in self.df.columns:
            self.df["Система сбора данных"] = self.df["Система сбора данных"].astype(str)

    def _normalize_date(self, date_value) -> Optional[datetime]:
        """
        Normalize date to YYYY-MM-DD format.

        Handles:
        - DD.MM.YYYY
        - MM/DD/YY
        - datetime objects
        - Excel serial dates
        """
        if pd.isna(date_value):
            return None

        # Already a datetime
        if isinstance(date_value, datetime):
            return date_value

        # Excel serial date (float)
        if isinstance(date_value, (int, float)):
            try:
                # Excel epoch is 1899-12-30
                return datetime(1899, 12, 30) + pd.Timedelta(days=date_value)
            except Exception:
                return None

        # String date
        if isinstance(date_value, str):
            date_str = date_value.strip()

            # Try DD.MM.YYYY
            try:
                return datetime.strptime(date_str, "%d.%m.%Y")
            except ValueError:
                pass

            # Try MM/DD/YY
            try:
                return datetime.strptime(date_str, "%m/%d/%y")
            except ValueError:
                pass

            # Try MM/DD/YYYY
            try:
                return datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                pass

        return None

    def _normalize_numeric(self, value) -> Optional[float]:
        """
        Normalize numeric value.

        Handles:
        - Strings with comma as decimal separator
        - NaN/None values
        - Regular numbers
        """
        if pd.isna(value) or value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in ["nan", "null", "none", "-"]:
                return None

            # Replace comma with dot
            value = value.replace(",", ".")

            try:
                return float(value)
            except ValueError:
                return None

        return None

    def _parse_ns_codes(self, value) -> Optional[str]:
        """
        Parse NS (non-standard situation) codes.

        Handles multiple codes separated by commas.
        Returns comma-separated string of codes.
        """
        if pd.isna(value) or value is None:
            return None

        if isinstance(value, (int, float)):
            return str(int(value)) if not pd.isna(value) else None

        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in ["nan", "null", "none", "-"]:
                return None

            # Split by comma and clean
            codes = [str(c).strip() for c in value.split(",")]
            codes = [c for c in codes if c and c not in ["nan", "null", "none"]]

            return ",".join(codes) if codes else None

        return None

    def _validate_columns(self):
        """Validate that required columns are present."""
        required = ["Дата", "Q"]

        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _filter_summary_rows(self):
        """Filter out summary/aggregation rows."""
        if "Дата" not in self.df.columns:
            return

        # Filter rows where Date is not a valid datetime
        initial_len = len(self.df)
        self.df = self.df[self.df["Дата"].apply(lambda x: isinstance(x, datetime))]

        filtered_count = initial_len - len(self.df)
        if filtered_count > 0:
            logger.info("Filtered summary rows", count=filtered_count)

        # Also filter by checking for known summary text patterns
        for col in self.df.columns:
            if self.df[col].dtype == object:
                for skip_text in self.SKIP_ROWS:
                    mask = self.df[col].astype(str).str.contains(skip_text, na=False)
                    self.df = self.df[~mask]

    def get_building_info(self) -> Dict[str, Any]:
        """
        Extract building information from metadata.

        Returns:
            Dict with address, area, year_built, etc.
        """
        info = {}

        # Try to extract from consumer metadata
        consumer = self.metadata.get("consumer", "")
        if consumer:
            # Simple heuristic: look for address patterns
            # This would need to be customized based on actual data format
            info["address"] = consumer

        return info


def parse_excel_file(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to parse an Excel file.

    Args:
        file_path: Path to the Excel file

    Returns:
        Tuple of (DataFrame, metadata)
    """
    parser = ExcelParser(file_path)
    return parser.parse()
