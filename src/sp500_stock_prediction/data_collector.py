"""
MODULE 1: Data Collection & Ingestion Pipeline
===============================================
Purpose: Fetch real-time and historical stock data for S&P 500 constituents
using yfinance with retry logic, caching, and validation.

Features:
- Multi-stock batch fetching (S&P 500 tickers)
- Real-time + historical data modes
- Data validation (schema, null checks, date range)
- Local caching with SQLite to avoid API rate limits
- Retry with exponential backoff

Input:  List of ticker symbols, date range, interval
Output: Cleaned pandas DataFrame with OHLCV data indexed by Date.
"""

from __future__ import annotations

import functools
import io
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

REQUIRED_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume"]
DATE_TOLERANCE_DAYS = 5

PERIOD_DAYS: Dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 91,
    "6mo": 182,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
}

F = TypeVar("F", bound=Callable[..., Any])


class DataFetchError(Exception):
    """Raised when data cannot be fetched from the remote source."""


class ValidationError(Exception):
    """Raised when fetched or supplied data fails validation rules."""


class CacheError(Exception):
    """Raised when reading from or writing to the local cache fails."""


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> Callable[[F], F]:
    """Retry a function with exponential backoff.

    Catches network-related exceptions as well as DataFetchError raised by
    this module. The delay between attempts grows as
    ``base_delay * (2 ** attempt)``.

    Args:
        max_retries: Total number of attempts before giving up.
        base_delay: Base delay in seconds for the first retry.

    Returns:
        A decorator that wraps the target callable with retry logic.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    DataFetchError,
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ) as exc:
                    last_exc = exc
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "retry_attempt=%d/%d func=%s delay=%.2fs error=%s",
                        attempt + 1,
                        max_retries,
                        getattr(func, "__name__", repr(func)),
                        delay,
                        exc,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise DataFetchError(
                f"{getattr(func, '__name__', repr(func))} failed after "
                f"{max_retries} attempts: {last_exc}"
            ) from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


_SP500_STATIC_TICKERS_RAW = """
AAPL MSFT NVDA AMD AVGO INTC QCOM TXN ADI MU MRVL NXPI MPWR SWKS QRVO ON TER ENTG
AMAT LRCX KLAC SNPS CDNS ADSK ANSS ANET FFIV NTAP JBL GLW ZBRA KEYS GRMN TRMB TDY IT
IBM ORCL CRM ADBE INTU NOW WDAY VEEV APP SMCI DELL HPE WDC STX ANET AKAM GDDY PANW
GOOGL GOOG META NFLX DIS CMCSA CHTR WBD PARA FOXA FOX OMC IPG T TMUS VZ EA TTWO MTCH LYV
AMZN TSLA HD LOW TJX ROST ORLY AZO LKQ NKE LULU RL PVH BBWI DG DLTR COST WMT TGT ULTA
YUM DRI CMG SBUX MCD DPZ DASH ABNB UBER BKNG EXPE MAR HLT HST WYNN LVS MGM CZR RCL
CCL NCLH LYV EBAY F GM APTV BWA GPC LKQ KMX CPRT ORLY AZO AAP
KO PEP KDP MNST K KHC GIS CAG HRL TSN LW HSY MKC SJM CHD CL KMB CLX EL KVUE PG
STZ BF.B MO PM MDLZ COST WMT TGT DG DLTR
UNH LLY JNJ PFE BMY AMGN GILD BIIB VRTX REGN MRNA ALNY INCY CI CNC HUM MOH MCK HSIC
DGX LH IQV ABT MDT BSX EW SYK ISRG RMD STE ZBH BDX WST COO HOLX TFX A MTD WAT RVTY
PODD DXCM ZTS HCA UHS TMO CRL
BRK.B JPM BAC WFC C GS MS SCHW RJF TROW BEN IVZ STT BK NTRS PFG PRU MET AFL UNM GL
AIG TRV ALL PGR HIG CINF CB ACGL AJG MMC AON WTW AIZ ERIE ICE CME CBOE NDAQ MSCI COF
DFS AXP SYF KEY HBAN RF CFG MTB FITB TFC PNC USB CMA ZION EWBC ALLY AMP RJF KKR BX
XOM CVX COP EOG FANG DVN APA HAL BKR SLB TRGP OKE WMB KMI LNG EQT CTRA PSX VLO MPC
OXY HES NRG VST CEG AEE
DUK SO D AEP ED EXC XEL PEG PPL WEC ES EIX PCG SRE DTE FE CNP ETR EVRG LNT NI ATO NJR AES
BA GE GEV RTX LMT NOC GD HWM TDG TXT CAT DE URI ETN EMR AME ROK ROP DOV FTV IEX GNRC
XYL WSO PH SWK MAS ALLE MBC CCK IP WRK SEE AVY BALL AMCR PKG MMM HUBB PWR EME ACM J
ODFL JBHT EXPD CHRW FDX UPS DAL UAL LUV TXT HII AXON TEL APH GLW OTIS CARR JCI TT VRT
GWW FAST TSCO POOL BLDR LEN PHM DHI NVR TOL MHK
LIN APD SHW PPG ECL DD DOW LYB CE ALB FCX NEM AEM SCCO MLM VMC EXPD CTVA MOS NUE STLD CLF AA
PLTR SPGI LOW HON UPS BA ELV PLD GS AXP SBUX BKNG AMAT MDT ADI MMC AON TJX NEM SLB
IR GTLS DOV FTV IEX GNRC XYL WSO PH SWK MAS ALLE MBC
"""

_SP500_STATIC_TICKERS: List[str] = sorted(
    {t.strip() for t in _SP500_STATIC_TICKERS_RAW.split() if t.strip()}
)


class StockDataCollector:
    """Collects, validates, and caches OHLCV stock data via yfinance."""

    def __init__(self, cache_db: str = "stock_cache.db", max_retries: int = 3) -> None:
        self.cache_db = Path(cache_db)
        self.max_retries = max_retries
        try:
            if self.cache_db.parent != Path(""):
                self.cache_db.parent.mkdir(parents=True, exist_ok=True)
            self._init_cache()
        except sqlite3.Error as exc:
            raise CacheError(f"failed to initialise cache at {self.cache_db}: {exc}") from exc
        self._download = retry_with_backoff(max_retries=self.max_retries, base_delay=0.25)(
            self._download_once
        )

    def _init_cache(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    ticker TEXT PRIMARY KEY,
                    saved_at TEXT NOT NULL,
                    payload BLOB NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.cache_db))

    def _download_once(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval)
        except Exception as exc:
            raise DataFetchError(f"yfinance request failed for {ticker}: {exc}") from exc
        if raw is None or len(raw) == 0:
            raise DataFetchError(f"no data returned for ticker={ticker!r}")
        return raw

    def fetch_single(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV history for one ticker.

        Args:
            ticker: Stock symbol, e.g. ``"AAPL"``.
            period: yfinance period string, e.g. ``"2y"``.
            interval: Candle interval, e.g. ``"1d"``.
            use_cache: When True, serve fresh cached data when available.

        Returns:
            Cleaned DataFrame with OHLCV columns and a DatetimeIndex.

        Raises:
            DataFetchError: If no data could be retrieved after retries.
            ValidationError: If retrieved data does not pass validation.
        """
        ticker = ticker.strip().upper()
        if use_cache:
            cached = self.load_from_cache(ticker)
            if cached is not None:
                logger.info("ticker=%s status=cache_hit rows=%d", ticker, len(cached))
                return cached

        df = self._clean(self._download(ticker, period, interval))
        if not self.validate_data(df, period=period):
            raise ValidationError(f"fetched data for {ticker} failed validation")
        self.cache_data(ticker, df)
        logger.info("ticker=%s status=fetched rows=%d period=%s", ticker, len(df), period)
        return df

    def fetch_batch(
        self, tickers: List[str], period: str = "2y", interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV history for many tickers, skipping failures.

        Args:
            tickers: Iterable of ticker symbols.
            period: yfinance period string applied to every ticker.
            interval: Candle interval applied to every ticker.

        Returns:
            Mapping of successfully fetched tickers to their DataFrames.
        """
        results: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_single(ticker, period=period, interval=interval)
            except (DataFetchError, ValidationError) as exc:
                logger.error("ticker=%s status=batch_fetch_failed error=%s", ticker, exc)
        logger.info(
            "status=batch_complete succeeded=%d failed=%d",
            len(results),
            len(tickers) - len(results),
        )
        return results

    @staticmethod
    def validate_data(df: Optional[pd.DataFrame], period: Optional[str] = None) -> bool:
        """Validate an OHLCV DataFrame against schema and quality rules.

        Rules checked:
        - Required columns present (Open, High, Low, Close, Volume).
        - Index is a sorted DatetimeIndex.
        - No null values in Close; Close values are positive.
        - Volume is numeric and strictly positive.
        - High >= Low and High/Low bracket Open/Close.
        - Date span covers the requested period within +/- 5 days tolerance
          (only when ``period`` maps to a known calendar length).

        Returns:
            True when all rules pass, otherwise False (errors are logged).
        """
        if df is None or df.empty:
            logger.error("validation_failed reason=empty_dataframe")
            return False

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.error("validation_failed reason=missing_columns missing=%s", missing)
            return False

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("validation_failed reason=index_not_datetime")
            return False

        if df["Close"].isnull().any():
            logger.error("validation_failed reason=null_close")
            return False

        if (df["Close"] <= 0).any():
            logger.error("validation_failed reason=nonpositive_close")
            return False

        if not pd.api.types.is_numeric_dtype(df["Volume"]) or (df["Volume"] <= 0).any():
            logger.error("validation_failed reason=invalid_volume")
            return False

        bad_bars = (
            (df["High"] < df["Low"])
            | (df["High"] < df["Open"])
            | (df["High"] < df["Close"])
            | (df["Low"] > df["Open"])
            | (df["Low"] > df["Close"])
        )
        if bad_bars.any():
            logger.error("validation_failed reason=inconsistent_ohlcv rows=%d", int(bad_bars.sum()))
            return False

        if period is not None and period in PERIOD_DAYS:
            expected_days = PERIOD_DAYS[period]
            span_days = (df.index.max() - df.index.min()).days + 1
            if span_days < expected_days - DATE_TOLERANCE_DAYS:
                logger.error(
                    "validation_failed reason=date_range_too_short span=%dd expected>=%dd",
                    span_days,
                    expected_days - DATE_TOLERANCE_DAYS,
                )
                return False

        return True

    def cache_data(self, ticker: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame to the SQLite cache.

        Raises:
            CacheError: If the write fails.
        """
        buffer = io.BytesIO()
        df.to_pickle(buffer)
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO ohlcv_cache (ticker, saved_at, payload)
                    VALUES (?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        saved_at = excluded.saved_at,
                        payload = excluded.payload
                    """,
                    (ticker.upper(), now, buffer.getvalue()),
                )
        except sqlite3.Error as exc:
            raise CacheError(f"cache write failed for {ticker}: {exc}") from exc
        logger.info("ticker=%s status=cached rows=%d", ticker, len(df))

    def load_from_cache(self, ticker: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Load a cached DataFrame if it exists and is fresh enough.

        Args:
            ticker: Stock symbol.
            max_age_hours: Maximum age of the cached entry in hours.

        Returns:
            The cached DataFrame, or None when absent/stale/unreadable.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT saved_at, payload FROM ohlcv_cache WHERE ticker = ?",
                    (ticker.strip().upper(),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise CacheError(f"cache read failed for {ticker}: {exc}") from exc

        if row is None:
            return None

        saved_at_raw, payload = row
        try:
            saved_at = datetime.fromisoformat(saved_at_raw)
            if saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
            age_hours = (
                datetime.now(timezone.utc) - saved_at
            ).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                logger.info("ticker=%s status=cache_expired age_h=%.1f", ticker, age_hours)
                return None
            df = pd.read_pickle(io.BytesIO(payload))
        except (ValueError, TypeError, OSError) as exc:
            raise CacheError(f"cached payload unreadable for {ticker}: {exc}") from exc
        logger.info("ticker=%s status=cache_loaded rows=%d age_h=%.1f", ticker, len(df), age_hours)
        return df

    @staticmethod
    def get_sp500_tickers() -> List[str]:
        """Return current S&P 500 constituent tickers.

        Attempts to scrape Wikipedia's constituent list first and falls back
        to an embedded static list when scraping is unavailable.

        Returns:
            Sorted list of unique ticker symbols (>400 entries).
        """
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            table = next(t for t in tables if "Symbol" in t.columns)
            tickers = (
                table["Symbol"]
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .unique()
                .tolist()
            )
            if len(tickers) >= 400:
                return sorted(tickers)
            logger.warning("wikipedia scrape returned only %d tickers", len(tickers))
        except Exception as exc:
            logger.warning("wikipedia scrape failed (%s); using static fallback", exc)
        return sorted(_SP500_STATIC_TICKERS)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        cleaned = cleaned.sort_index()
        cleaned = cleaned.dropna(subset=["Close"])
        return cleaned
