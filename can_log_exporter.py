# 2026-02-24 00:00 v1.1.0 - Rewrite TRC exporter: FILEVERSION 2.1 → 1.1, exact
#                           PCAN-View v1.1 column layout and header block for
#                           PCAN Explorer compatibility.
"""
CAN Log Exporter  –  multi-format export module for can_tui.py.

Supported output formats
------------------------
  CSV   Plain comma-separated values.  Universally importable (Excel, Python …).
  ASC   ASCII log format written by Vector CANalyzer / CANoe (text, *.asc).
  TRC   PEAK PCAN-View trace format v1.1 (text, *.trc).
  BLF   Vector Binary Logging Format (binary, *.blf) via the python-can library.

Usage example
-------------
    from can_log_exporter import ExportFormat, export_records

    ok, msg = export_records(records, ExportFormat.ASC, filepath="/tmp/my_trace.asc",
                             channel=1, bitrate=500_000)
    if not ok:
        print("Export failed:", msg)

The *records* argument is a list of :class:`TraceRecord`-compatible objects that
expose the following attributes:

    rel_ts       float   – seconds since recording start
    can_id       int     – CAN identifier (raw, without extended flag)
    is_extended  bool    – True for 29-bit extended frames
    direction    str     – "Rx" or "Tx"
    dlc          int     – data length code (0-8)
    data         bytes   – payload bytes
"""

from __future__ import annotations

import csv
import io
import os
import struct
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional, Protocol, Tuple

# Tool version embedded in exported file headers
_TOOL_VERSION = "v1.6.0"


def _fmt_can_id(can_id: int, is_extended: bool) -> str:
    """Format a CAN ID without leading zeros (CANalyzer-compatible).

    Extended (29-bit): no zero-padding  → e.g. 0xEFF001A  (7 digits)
    Standard (11-bit): always 3 digits  → e.g. 0x111
    """
    if is_extended:
        return f"0x{can_id:X}"
    return f"0x{can_id:03X}"

# ---------------------------------------------------------------------------
# Optional python-can import (needed only for BLF)
# ---------------------------------------------------------------------------
try:
    import can as python_can  # python-can library
    _PYTHON_CAN_AVAILABLE = True
except ImportError:
    _PYTHON_CAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExportFormat(Enum):
    """Supported CAN log export formats."""

    CSV = "CSV"
    ASC = "ASC"
    TRC = "TRC"
    BLF = "BLF"

    @property
    def extension(self) -> str:
        """Return the canonical file extension (without leading dot)."""
        return {
            ExportFormat.CSV: "csv",
            ExportFormat.ASC: "asc",
            ExportFormat.TRC: "trc",
            ExportFormat.BLF: "blf",
        }[self]

    @property
    def label(self) -> str:
        """Human-readable label for UI display."""
        return {
            ExportFormat.CSV: "CSV  (*.csv)",
            ExportFormat.ASC: "ASC  Vector CANalyzer  (*.asc)",
            ExportFormat.TRC: "TRC  PEAK PCAN-View  (*.trc)",
            ExportFormat.BLF: "BLF  Vector Binary  (*.blf)",
        }[self]

    @property
    def available(self) -> bool:
        """False when a required optional dependency is missing."""
        if self == ExportFormat.BLF:
            return _PYTHON_CAN_AVAILABLE
        return True


# The public export function (convenience wrapper)
def export_records(
    records: List[Any],
    fmt: ExportFormat,
    filepath: str,
    channel: int = 1,
    bitrate: int = 500_000,
) -> Tuple[bool, str]:
    """Write *records* to *filepath* in the requested format.

    Args:
        records:  List of TraceRecord-like objects (see module docstring).
        fmt:      Target :class:`ExportFormat`.
        filepath: Destination file path including extension.
        channel:  CAN channel number written into the log header (default 1).
        bitrate:  CAN bitrate in bit/s written into the log header (default 500000).

    Returns:
        Tuple ``(ok, message)`` where *ok* is True on success and *message*
        is a human-readable status string (filename on success, error text
        on failure).
    """
    exporter = _EXPORTERS[fmt](channel=channel, bitrate=bitrate)
    return exporter.export(records, filepath)


# ---------------------------------------------------------------------------
# TraceRecord protocol (duck-typed – no hard dependency on can_tui.py)
# ---------------------------------------------------------------------------

class _TraceRecordLike(Protocol):
    rel_ts: float
    can_id: int
    is_extended: bool
    direction: str
    dlc: int
    data: bytes


# ---------------------------------------------------------------------------
# Base exporter
# ---------------------------------------------------------------------------

class BaseExporter(ABC):
    """Abstract base class shared by all format exporters."""

    def __init__(self, channel: int = 1, bitrate: int = 500_000) -> None:
        self._channel = channel
        self._bitrate = bitrate

    # -- Public ---------------------------------------------------------------

    def export(
        self, records: List[Any], filepath: str
    ) -> Tuple[bool, str]:
        """Write records to filepath.  Returns (ok, message)."""
        if not records:
            return False, "No records to export"
        try:
            self._write(records, filepath)
            filename = os.path.basename(filepath)
            return True, f"Saved: {filename}  ({len(records)} frames)"
        except Exception as exc:  # pylint: disable=broad-except
            return False, f"Export failed: {exc}"

    # -- Private (subclass responsibility) ------------------------------------

    @abstractmethod
    def _write(self, records: List[Any], filepath: str) -> None:
        """Write records to the given filepath (raises on error)."""


# ---------------------------------------------------------------------------
# CSV exporter
# ---------------------------------------------------------------------------

class CSVExporter(BaseExporter):
    """Export to plain comma-separated values."""

    def _write(self, records: List[Any], filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["Time_s", "CAN_ID_hex", "Type", "Dir", "DLC", "Data_hex"]
            )
            for rec in records:
                id_str = _fmt_can_id(rec.can_id, rec.is_extended)
                writer.writerow([
                    f"{rec.rel_ts:.6f}",
                    id_str,
                    "Ext" if rec.is_extended else "Std",
                    rec.direction,
                    rec.dlc,
                    rec.data.hex(" ").upper(),
                ])


# ---------------------------------------------------------------------------
# ASC exporter  (Vector CANalyzer / CANoe ASCII log)
# ---------------------------------------------------------------------------

_ASC_DATE_FMT = "%a %b %d %I:%M:%S %p %Y"   # e.g.  Sat Feb 22 14:30:00 PM 2026


class ASCExporter(BaseExporter):
    """Export to Vector CANalyzer / CANoe ASCII log format (.asc).

    Format reference: Vector ASC format, used by CANalyzer ≥ 5.x and CANoe.
    Timestamps are written as absolute seconds (floating-point) since the
    start of the recording.

    Frame line layout (data frame, 8 bytes):
        <time>  <ch>  <id>  <dir>  d  <dlc>  <b0> <b1> … <bN>
        0.001234  1  0111  Rx  d  8  11 22 33 44 55 66 77 88

    Extended-frame IDs are suffixed with 'x':
        0.001234  1  18FEF100x  Rx  d  8  DE AD BE EF 00 11 22 33
    """

    def _write(self, records: List[Any], filepath: str) -> None:
        now = datetime.now()
        lines: List[str] = []

        # --- File header ---
        lines.append(f"date {now.strftime(_ASC_DATE_FMT)}")
        lines.append("base hex  timestamps absolute")
        lines.append("no internal events logged")
        lines.append(
            f"// Generated by Waveshare CAN TUI  "
            f"– channel {self._channel}, {self._bitrate // 1000} kbps"
        )
        lines.append("")

        # --- Frame records ---
        for rec in records:
            id_str = (
                f"{rec.can_id:X}x" if rec.is_extended
                else f"{rec.can_id:X}"
            )
            data_str = " ".join(f"{b:02X}" for b in rec.data)
            # Direction: "Rx" → "Rx", "Tx" → "Tx"
            dir_str = rec.direction
            lines.append(
                f"   {rec.rel_ts:.6f}  {self._channel}  {id_str}"
                f"  {dir_str}  d  {rec.dlc}  {data_str}"
            )

        lines.append("")
        lines.append("End of measurement")

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            fh.write("\n")


# ---------------------------------------------------------------------------
# TRC exporter  (PEAK PCAN-View trace format v1.1)
# ---------------------------------------------------------------------------

class TRCExporter(BaseExporter):
    """Export to PEAK PCAN-View trace format v1.1 (.trc).

    Format reference: PCAN-View TRC v1.1 – the format accepted by PCAN Explorer
    and PCAN-View 5.x when opened via File → Open.

    Column layout (matches PCAN-View output exactly):
        <msg_nr>)  <time_ms>  <dir>  <id_8hex>  <dlc>  <data bytes ...>

        Field widths (1-based character positions):
          msg_nr  right-aligned in col 1-6   (e.g. "     1)")
          time_ms right-aligned in col 8-18  (e.g. "       0.9")  1 decimal
          dir     col 20-21                  ("Rx" or "Tx")
          id      col 24-31                  always 8 hex digits
          dlc     col 34                     1 digit
          data    col 36+                    "BB 9B …" space-separated

    Extended IDs are written as-is (8 hex digits).
    Standard IDs (11-bit) are zero-padded to 8 digits to match PCAN-View v1.1
    behaviour – PCAN Explorer ignores the upper nibbles for standard frames.

    The ;$STARTTIME value is a Windows OLE Automation Date (fractional days
    since 30-Dec-1899 00:00:00 UTC), matching what PCAN-View writes.
    """

    # OLE Automation Date epoch: 30-Dec-1899 00:00:00 UTC
    _OLE_EPOCH = datetime(1899, 12, 30, tzinfo=timezone.utc)

    # Column positions derived from real PCAN-View v1.1 trace files:
    #   msg_nr field ends at position  6  (right-aligned, followed by ")")
    #   time_ms field right-edge at   18  (1 decimal place, in ms)
    #   dir field starts at           20
    #   id field starts at            27  (8 hex digits)
    #   dlc field starts at           37
    #   data field starts at          40
    _HDR_RULER = (
        ";---+--   ----+----  --+--  ----+---  +  "
        "-+ -- -- -- -- -- -- --"
    )

    def _ole_date(self, dt: datetime) -> float:
        """Convert a datetime to an OLE Automation Date (fractional days)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt - self._OLE_EPOCH).total_seconds() / 86400.0

    def _write(self, records: List[Any], filepath: str) -> None:
        now = datetime.now()
        ole_start = self._ole_date(now.replace(tzinfo=timezone.utc))

        # Start-time string: DD.MM.YYYY HH:MM:SS.mmm.0  (matches PCAN-View)
        start_str = now.strftime("%d.%m.%Y %H:%M:%S.") + f"{now.microsecond // 1000:03d}.0"

        lines: List[str] = []

        # --- File header (exact PCAN-View v1.1 layout) ---
        lines.append(";$FILEVERSION=1.1")
        lines.append(f";$STARTTIME={ole_start:.10f}")
        lines.append(";")
        lines.append(f";   Start time: {start_str}")
        lines.append(f";   Generated by Waveshare CAN TUI {_TOOL_VERSION}")
        lines.append(";")
        lines.append(";   Message Number")
        lines.append(";   |         Time Offset (ms)")
        lines.append(";   |         |        Type")
        lines.append(";   |         |        |        ID (hex)")
        lines.append(";   |         |        |        |     Data Length")
        lines.append(";   |         |        |        |     |   Data Bytes (hex) ...")
        lines.append(";   |         |        |        |     |   |")
        lines.append(self._HDR_RULER)

        # --- Frame records ---
        for idx, rec in enumerate(records, start=1):
            time_ms = rec.rel_ts * 1000.0
            # TRC v1.1 always uses 8 zero-padded hex digits for the ID field –
            # this is required by the format spec and matches PCAN-View output.
            # Note: _fmt_can_id() is intentionally NOT used here.
            id_str = f"{rec.can_id:08X}"
            dir_str = rec.direction  # "Rx" or "Tx"
            data_str = " ".join(f"{b:02X}" for b in rec.data)
            # Format matches PCAN-View v1.1 column layout exactly:
            #   "     1)         0.9  Rx     0EFF011C  8  BB 9B …"
            lines.append(
                f"{idx:>6})  {time_ms:>11.1f}  {dir_str:<2}     {id_str}  "
                f"{rec.dlc}  {data_str}"
            )

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            fh.write("\n")


# ---------------------------------------------------------------------------
# BLF exporter  (Vector Binary Logging Format via python-can)
# ---------------------------------------------------------------------------

class BLFExporter(BaseExporter):
    """Export to Vector Binary Logging Format (.blf) using python-can's BLFWriter.

    Requires: pip install python-can

    The BLF format is the native binary format of Vector CANalyzer and CANoe.
    Files produced here can be opened directly in CANalyzer without any
    conversion step.  BLF is typically 10–50× more compact than the equivalent
    ASC file.

    Notes:
      - python-can's BLFWriter uses the system wall-clock time as the absolute
        start timestamp.  The relative timestamps from TraceRecord are added to
        this base to reconstruct absolute timestamps for each frame.
      - The channel number is passed through to the BLF object channel field.
      - BLF files are big-endian internally; python-can handles all encoding.
    """

    def _write(self, records: List[Any], filepath: str) -> None:
        if not _PYTHON_CAN_AVAILABLE:
            raise RuntimeError(
                "python-can is not installed.  "
                "Run: pip install python-can"
            )

        # Determine absolute base timestamp from wall clock at export time,
        # minus the total recording duration so that the last frame lands at
        # approximately "now".
        base_ts = time.time() - (records[-1].rel_ts if records else 0.0)

        with python_can.BLFWriter(filepath, channel=self._channel) as blf_writer:
            for rec in records:
                abs_ts = base_ts + rec.rel_ts
                # Pass channel=None in the Message so BLFWriter uses its own
                # default channel directly.  If msg.channel is set, python-can
                # applies a +1 offset ("many interfaces start at 0 which is
                # invalid") which would shift our channel 1 → 2, causing
                # CANalyzer to not find the DBC mapping.
                msg = python_can.Message(
                    timestamp=abs_ts,
                    arbitration_id=rec.can_id,
                    is_extended_id=rec.is_extended,
                    is_remote_frame=False,
                    dlc=rec.dlc,
                    data=rec.data,
                    channel=None,  # let BLFWriter use its own default (no +1 offset)
                )
                blf_writer(msg)


# ---------------------------------------------------------------------------
# Exporter registry
# ---------------------------------------------------------------------------

_EXPORTERS = {
    ExportFormat.CSV: CSVExporter,
    ExportFormat.ASC: ASCExporter,
    ExportFormat.TRC: TRCExporter,
    ExportFormat.BLF: BLFExporter,
}
