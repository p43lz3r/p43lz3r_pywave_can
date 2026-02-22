#!/usr/bin/env python3
# 2026-02-22 15:00 v1.0.0 - Multi-format trace export: CSV, ASC (Vector CANalyzer),
#                           TRC (PEAK PCAN-View), BLF (Vector Binary Logging Format).
#                           Format selector dropdown added to Trace tab.
#                           Requires can_log_exporter.py v1.0.0.
#                           Based on v0.9.1: Timer optimisations, blinking REC indicator.
# 2026-02-22 v1.1.0 - Sprint 1: Signal Discovery Screen (F7) + Stale Value Highlighting.
#                     F7 opens DiscoveryScreen: capture two CAN bus snapshots and inspect
#                     byte-level diffs per CAN-ID with DBC status, sort and unknown filter.
#                     Stale Value Highlighting: DLC + Data cells turn red after 10 s without
#                     a data change on the main monitor table (STALE_TIMEOUT_S = 10.0).
# 2026-02-22 v1.3.0 - Signal Discovery: Observe-Phase (Noise-Baseline) implementiert.
#                     Neuer State OBSERVING vor CAPTURING: [o] startet Observe, Bus-Aktivität
#                     wird als Noise-IDs akkumuliert. [c] beendet Observe und startet Capture.
#                     Noise-IDs werden in _compute_deltas() gefiltert. Header zeigt live
#                     Noise-Count während OBSERVING und gefilterte Anzahl in RESULTS.
#                     [c] direkt ohne [o] funktioniert weiterhin ohne Noise-Filter.
#                     Frame Type selector in ConnectionPanel (Auto-Detect / Extended / Standard).
#                     Auto-Detection on connect: listens 3 s for Extended frames, then
#                     switches to Standard if none received. Result shown in StatusPanel.
#                     Fixes silent loss of 11-bit Standard CAN frames (hardware limitation:
#                     Waveshare dongle accepts only one frame type at a time per protocol spec).
"""
CAN Bus TUI - Textual-based Terminal UI for Waveshare USB-CAN-A.

Requires waveshare_can.py v1.0.2 (corrected CANSpeed enum + setup command).
Requires cantools >= 39.0  (pip install cantools)
Requires can_log_exporter.py v1.0.0  (multi-format log export)
Optional: python-can  (pip install python-can)  – required for BLF export only.
"""

import argparse
import os
import platform
import glob
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.text import Text

from textual.app import App, ComposeResult
from textual.screen import Screen, ModalScreen
from textual.containers import Horizontal, Container, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Select, Label,
    Log, DataTable, Input, Checkbox, TabbedContent, TabPane,
)
from textual.reactive import reactive
from textual.timer import Timer

from waveshare_can import WaveshareCAN, CANFrame, CANSpeed, CANMode, CANFrameType

try:
    import cantools
    import cantools.database
    CANTOOLS_AVAILABLE = True
except ImportError:
    CANTOOLS_AVAILABLE = False

from can_log_exporter import ExportFormat, export_records


# ---------------------------------------------------------------------------
# CAN ID range constants
# ---------------------------------------------------------------------------
CAN_STD_MAX = 0x7FF        # 11-bit standard CAN ID maximum
CAN_EXT_MAX = 0x1FFFFFFF   # 29-bit extended CAN ID maximum

# ---------------------------------------------------------------------------
# Timer intervals (seconds)
# ---------------------------------------------------------------------------
TIMER_STATUS_S = 0.5    # Status display refresh
TIMER_FPS_S = 1.0       # FPS / RX counter snapshot
TIMER_MONITOR_S = 0.2   # Monitor table refresh
TIMER_STATS_S = 2.0     # Statistics panel refresh
TIMER_TRACE_S = 0.2     # Trace flush to DataTable
TIMER_DBC_S = 1.0       # DBC decoded signals refresh

# ---------------------------------------------------------------------------
# Misc tuning constants
# ---------------------------------------------------------------------------
HIGHLIGHT_FADE_S = 1.0          # Seconds a changed byte cell stays highlighted
STALE_TIMEOUT_S = 10.0          # Seconds before a non-changing frame is marked stale
TIMER_DISCOVERY_S = 0.5         # Discovery screen live-counter + blink refresh
THREAD_JOIN_TIMEOUT_S = 1.0     # Timeout when joining background threads
LOG_HISTORY_LINES = 200         # Max log lines replayed when Details opens
STATS_TOP_N = 30                # Rows shown in the Statistics table
TRACE_BATCH_SIZE = 50           # Max trace rows added per UI tick
TRACE_WARN_THRESHOLD = 100_000  # Frame count that triggers the warning label

# Theoretical max frames/s = bit_rate / bits_per_frame
# ~108 bits per standard frame with 8 data bytes and worst-case bit-stuffing
_BITS_PER_FRAME = 108


# ---------------------------------------------------------------------------
# Port detection
# ---------------------------------------------------------------------------
def detect_serial_ports() -> List[str]:
    """Return a list of available serial port device paths for the current OS."""
    ports: List[str] = []
    system = platform.system()
    if system == "Linux":
        ports.extend(sorted(glob.glob("/dev/ttyUSB*")))
        ports.extend(sorted(glob.glob("/dev/ttyACM*")))
    elif system == "Windows":
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
    elif system == "Darwin":
        ports.extend(sorted(glob.glob("/dev/tty.usb*")))
        ports.extend(sorted(glob.glob("/dev/cu.usb*")))
    return ports if ports else ["(no ports found)"]


# ---------------------------------------------------------------------------
# Speed / Mode option lists
# ---------------------------------------------------------------------------
SPEED_OPTIONS: List[Tuple[str, CANSpeed]] = [
    ("5 kbps", CANSpeed.SPEED_5K),
    ("10 kbps", CANSpeed.SPEED_10K),
    ("20 kbps", CANSpeed.SPEED_20K),
    ("50 kbps", CANSpeed.SPEED_50K),
    ("100 kbps", CANSpeed.SPEED_100K),
    ("125 kbps", CANSpeed.SPEED_125K),
    ("200 kbps", CANSpeed.SPEED_200K),
    ("250 kbps", CANSpeed.SPEED_250K),
    ("400 kbps", CANSpeed.SPEED_400K),
    ("500 kbps", CANSpeed.SPEED_500K),
    ("800 kbps", CANSpeed.SPEED_800K),
    ("1 Mbps", CANSpeed.SPEED_1M),
]

MODE_OPTIONS: List[Tuple[str, CANMode]] = [
    ("Normal", CANMode.NORMAL),
    ("Silent", CANMode.SILENT),
    ("Loopback", CANMode.LOOPBACK),
    ("Loopback-Silent", CANMode.LOOPBACK_SILENT),
]

# "Auto" is represented as None – triggers Auto-Detection on connect
FRAME_TYPE_OPTIONS: List[Tuple[str, Optional[CANFrameType]]] = [
    ("Auto-Detect", None),
    ("Extended (29-bit)", CANFrameType.EXTENDED),
    ("Standard (11-bit)", CANFrameType.STANDARD),
]

# Seconds to wait for frames before switching frame type during Auto-Detection
AUTO_DETECT_TIMEOUT_S = 3.0

SORT_MODES: List[Tuple[str, str, bool]] = [
    ("ID \u2191", "id", False),
    ("Rate \u2193", "rate", True),
    ("Count \u2193", "count", True),
]

# CAN bus theoretical max frames/s per speed setting
SPEED_MAX_FPS: Dict[CANSpeed, float] = {
    CANSpeed.SPEED_5K: 5_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_10K: 10_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_20K: 20_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_50K: 50_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_100K: 100_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_125K: 125_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_200K: 200_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_250K: 250_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_400K: 400_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_500K: 500_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_800K: 800_000 / _BITS_PER_FRAME,
    CANSpeed.SPEED_1M: 1_000_000 / _BITS_PER_FRAME,
}

# Fallback max FPS used when the current speed is not found in the dict
_DEFAULT_MAX_FPS = 500_000 / _BITS_PER_FRAME


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
# Map human-readable speed strings to CANSpeed enum values
_SPEED_STR_MAP: Dict[str, CANSpeed] = {
    "5K":    CANSpeed.SPEED_5K,
    "10K":   CANSpeed.SPEED_10K,
    "20K":   CANSpeed.SPEED_20K,
    "50K":   CANSpeed.SPEED_50K,
    "100K":  CANSpeed.SPEED_100K,
    "125K":  CANSpeed.SPEED_125K,
    "200K":  CANSpeed.SPEED_200K,
    "250K":  CANSpeed.SPEED_250K,
    "400K":  CANSpeed.SPEED_400K,
    "500K":  CANSpeed.SPEED_500K,
    "800K":  CANSpeed.SPEED_800K,
    "1M":    CANSpeed.SPEED_1M,
}


@dataclass
class StartupArgs:
    """Parsed CLI arguments passed into CANBusTUI at construction time."""

    port: Optional[str] = None          # --port / -p
    speed: Optional[CANSpeed] = None    # --speed / -s
    dbc_path: Optional[str] = None      # --dbc / -d
    auto_connect: bool = False          # --connect / -c


def parse_cli_args() -> StartupArgs:
    """Parse sys.argv and return a StartupArgs instance."""
    parser = argparse.ArgumentParser(
        prog="can_tui.py",
        description="Waveshare CAN Bus TUI  v1.0.0",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p", "--port",
        metavar="PORT",
        help="Serial port  (e.g. /dev/ttyUSB0  or  COM3)",
    )
    parser.add_argument(
        "-s", "--speed",
        metavar="SPEED",
        help=(
            "CAN bus bitrate.  Valid values:\n"
            "  5K 10K 20K 50K 100K 125K 200K 250K 400K 500K 800K 1M\n"
            "  (default: 500K)"
        ),
    )
    parser.add_argument(
        "-d", "--dbc",
        metavar="FILE",
        help="Path to a .dbc file to load on startup",
    )
    parser.add_argument(
        "-c", "--connect",
        action="store_true",
        help="Auto-connect on startup (requires --port)",
    )

    args = parser.parse_args()

    # Validate and convert speed string
    speed: Optional[CANSpeed] = None
    if args.speed:
        key = args.speed.upper().replace("BPS", "").replace("KBPS", "K").strip()
        speed = _SPEED_STR_MAP.get(key)
        if speed is None:
            valid = ", ".join(_SPEED_STR_MAP.keys())
            parser.error(f"Unknown speed '{args.speed}'.  Valid: {valid}")

    # Validate DBC path early so the user gets a clear error before the TUI starts
    dbc_path: Optional[str] = None
    if args.dbc:
        if not os.path.isfile(args.dbc):
            parser.error(f"DBC file not found: {args.dbc}")
        dbc_path = args.dbc

    if args.connect and not args.port:
        parser.error("--connect requires --port")

    return StartupArgs(
        port=args.port,
        speed=speed,
        dbc_path=dbc_path,
        auto_connect=args.connect,
    )


# ---------------------------------------------------------------------------
# Thread-safe CAN frame store
# ---------------------------------------------------------------------------
class CANFrameStore:
    """Thread-safe store for the latest CAN frame per CAN-ID with change tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Dict[int, dict] = {}
        self._dirty = False

    def update(self, frame: CANFrame) -> None:
        """Insert or update a frame entry and set the dirty flag."""
        with self._lock:
            now = frame.timestamp
            can_id = frame.can_id
            if can_id not in self._rows:
                self._rows[can_id] = {
                    "frame": frame,
                    "count": 1,
                    "prev_data": frame.data,
                    "changed_mask": [False] * len(frame.data),
                    "first_ts": now,
                    "last_ts": now,
                }
            else:
                e = self._rows[can_id]
                prev = e["prev_data"]
                new = frame.data
                ml = max(len(prev), len(new))
                mask = [
                    (prev[i] if i < len(prev) else -1) != (new[i] if i < len(new) else -1)
                    for i in range(ml)
                ]
                e["frame"] = frame
                e["count"] += 1
                e["prev_data"] = frame.data
                e["changed_mask"] = mask
                e["last_ts"] = now
            self._dirty = True

    def snapshot(self) -> Tuple[bool, dict]:
        """Consume the dirty flag and return a shallow copy of all rows.

        Used by the monitor table update loop.
        """
        with self._lock:
            dirty = self._dirty
            self._dirty = False
            return dirty, dict(self._rows)

    def read(self) -> dict:
        """Return a shallow copy of all rows without touching the dirty flag.

        Used by the statistics panel which has its own timer.
        """
        with self._lock:
            return dict(self._rows)

    def clear(self) -> None:
        """Remove all stored frames and set the dirty flag."""
        with self._lock:
            self._rows.clear()
            self._dirty = True


# ---------------------------------------------------------------------------
# Trace buffer
# ---------------------------------------------------------------------------
class TraceState(Enum):
    """State machine states for the trace recorder."""

    IDLE = "IDLE"
    RECORDING = "REC"
    PAUSED = "PAUSED"


@dataclass
class TraceRecord:
    """A single recorded CAN frame entry in the trace buffer."""

    rel_ts: float       # Seconds elapsed since recording start
    can_id: int
    is_extended: bool
    direction: str      # "Rx" or "Tx"
    dlc: int
    data: bytes


class DiscoveryState(Enum):
    """State machine states for the Signal Discovery screen."""

    IDLE = "IDLE"
    OBSERVING = "OBSERVING"   # Phase 1: building noise baseline
    CAPTURING = "CAPTURING"   # Phase 2: waiting for user action
    RESULTS = "RESULTS"       # Phase 3: showing delta table


@dataclass
class ChangeDelta:
    """Represents the change between two CAN frame snapshots for one CAN-ID.

    Attributes:
        can_id:          The CAN identifier.
        is_extended:     True if this is a 29-bit extended frame.
        before_data:     Bytes from Snapshot 1 (empty if frame is newly appeared).
        after_data:      Bytes from Snapshot 2 (empty if frame disappeared).
        changed_indices: Byte positions that differ between snapshots.
        dbc_name:        Message name from the loaded DBC, or None if unknown.
    """

    can_id: int
    is_extended: bool
    before_data: bytes
    after_data: bytes
    changed_indices: List[int]
    dbc_name: Optional[str]


class TraceBuffer:
    """Unbounded, thread-safe trace buffer with a pending queue for the UI thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[TraceRecord] = []
        self._pending: List[TraceRecord] = []
        self._state = TraceState.IDLE
        self._start_ts: Optional[float] = None

    # -- State property -------------------------------------------------------

    @property
    def state(self) -> TraceState:
        """Current recorder state (thread-safe read)."""
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: TraceState) -> None:
        with self._lock:
            self._state = value

    # -- Record / drain -------------------------------------------------------

    def record(self, frame: CANFrame, direction: str = "Rx") -> None:
        """Append a frame to the buffer if recording is active."""
        with self._lock:
            if self._state != TraceState.RECORDING:
                return
            if self._start_ts is None:
                self._start_ts = frame.timestamp
            rec = TraceRecord(
                rel_ts=frame.timestamp - self._start_ts,
                can_id=frame.can_id,
                is_extended=frame.is_extended,
                direction=direction,
                dlc=len(frame.data),
                data=frame.data,
            )
            self._records.append(rec)
            self._pending.append(rec)

    def drain_pending(self) -> List[TraceRecord]:
        """Return and clear all pending (not-yet-displayed) records."""
        with self._lock:
            out = self._pending[:]
            self._pending.clear()
        return out

    @property
    def pending_count(self) -> int:
        """Number of frames buffered but not yet written to the display."""
        with self._lock:
            return len(self._pending)

    def prepend_pending(self, records: List[TraceRecord]) -> None:
        """Push records back to the front of the pending queue (overflow handling)."""
        with self._lock:
            self._pending[:0] = records

    def snapshot_records(self) -> List[TraceRecord]:
        """Return a copy of all recorded frames (for CSV export)."""
        with self._lock:
            return list(self._records)

    # -- State transitions ----------------------------------------------------

    def start(self, ts: Optional[float] = None) -> None:
        """Begin or resume recording; optionally supply an explicit start timestamp."""
        with self._lock:
            if self._state == TraceState.IDLE:
                self._start_ts = ts
            self._state = TraceState.RECORDING

    def pause(self) -> None:
        """Pause the trace display – new frames keep buffering in _pending.

        The table stops scrolling because _flush_trace() checks the state before
        writing rows.  On Resume all buffered frames are flushed at once, so no
        data is lost.  This matches the behaviour of professional CAN tools such
        as PCAN-View and CANalyzer.
        """
        with self._lock:
            if self._state == TraceState.RECORDING:
                self._state = TraceState.PAUSED
                # Do NOT clear _pending – frames continue to accumulate

    def resume(self) -> None:
        """Resume a paused recording."""
        with self._lock:
            if self._state == TraceState.PAUSED:
                self._state = TraceState.RECORDING

    def stop(self) -> None:
        """Stop recording; accumulated data is retained for export / review.

        Any frames still pending display are discarded so the trace table stops
        scrolling immediately.  Call clear() to also discard the stored data.
        """
        with self._lock:
            self._state = TraceState.IDLE
            self._pending.clear()   # stop table from receiving more rows

    def clear(self) -> None:
        """Discard all data and reset to IDLE state."""
        with self._lock:
            self._records.clear()
            self._pending.clear()
            self._start_ts = None
            self._state = TraceState.IDLE

    # -- Info -----------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of recorded frames."""
        with self._lock:
            return len(self._records)

    @property
    def warning(self) -> bool:
        """True when the frame count exceeds TRACE_WARN_THRESHOLD."""
        with self._lock:
            return len(self._records) >= TRACE_WARN_THRESHOLD


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------
def parse_id_list(text: str) -> Set[int]:
    """Parse a whitespace- or comma-separated string of CAN IDs into a set of ints."""
    ids: Set[int] = set()
    for token in text.replace(",", " ").split():
        try:
            ids.add(int(token, 0))
        except ValueError:
            pass
    return ids


# ---------------------------------------------------------------------------
# DBC database wrapper
# ---------------------------------------------------------------------------
@dataclass
class DBCSignalValue:
    """A single decoded signal value."""

    name: str
    value: float
    unit: str
    raw: int


@dataclass
class DBCMessageInfo:
    """Metadata about a DBC message."""

    can_id: int           # Raw CAN ID (without J1939 ext bit)
    name: str
    dlc: int
    signals: List[str]    # Signal names in this message


class DBCDatabase:
    """Thread-safe wrapper around a cantools database."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db: Optional[Any] = None          # cantools.database.Database
        self._path: str = ""
        self._msg_by_id: Dict[int, Any] = {}    # can_id → cantools Message object

    @property
    def loaded(self) -> bool:
        """True when a DBC file is loaded."""
        with self._lock:
            return self._db is not None

    @property
    def path(self) -> str:
        """Path of the currently loaded DBC file."""
        with self._lock:
            return self._path

    @property
    def message_count(self) -> int:
        """Number of messages in the loaded database."""
        with self._lock:
            return len(self._msg_by_id)

    def load(self, path: str) -> Tuple[bool, str]:
        """Load a DBC file. Returns (ok, error_message)."""
        if not CANTOOLS_AVAILABLE:
            return False, "cantools not installed  (pip install cantools)"
        try:
            db = cantools.database.load_file(path)
            msg_by_id: Dict[int, Any] = {}
            for msg in db.messages:
                # cantools stores the raw 29-bit ID without the extended flag
                msg_by_id[msg.frame_id] = msg
            with self._lock:
                self._db = db
                self._path = path
                self._msg_by_id = msg_by_id
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def unload(self) -> None:
        """Clear the loaded database."""
        with self._lock:
            self._db = None
            self._path = ""
            self._msg_by_id = {}

    def get_messages(self) -> List[DBCMessageInfo]:
        """Return a list of all messages sorted by CAN ID."""
        with self._lock:
            out: List[DBCMessageInfo] = []
            for can_id, msg in sorted(self._msg_by_id.items()):
                out.append(DBCMessageInfo(
                    can_id=can_id,
                    name=msg.name,
                    dlc=msg.length,
                    signals=[s.name for s in msg.signals],
                ))
            return out

    def lookup_name(self, can_id: int) -> Optional[str]:
        """Return the message name for a CAN ID, or None if not in DBC."""
        with self._lock:
            msg = self._msg_by_id.get(can_id)
            return msg.name if msg else None

    def decode(self, can_id: int, data: bytes) -> Optional[List[DBCSignalValue]]:
        """Decode a CAN frame. Returns a list of signal values or None."""
        with self._lock:
            msg = self._msg_by_id.get(can_id)
            if msg is None:
                return None
            try:
                decoded = msg.decode(data, decode_choices=False)
                out: List[DBCSignalValue] = []
                for sig in msg.signals:
                    val = decoded.get(sig.name)
                    if val is None:
                        continue
                    out.append(DBCSignalValue(
                        name=sig.name,
                        value=float(val),
                        unit=sig.unit or "",
                        raw=0,  # raw extraction would need more work; omit for clarity
                    ))
                return out
            except Exception:
                return None


def _format_signal_value(value: float) -> str:
    """Format a decoded signal value avoiding scientific notation for normal ranges.

    Rules:
    - Integer result → no decimal point  (e.g. 42)
    - abs(value) >= 10000 → 0 decimal places  (e.g. 38700)
    - abs(value) >= 100   → 1 decimal place   (e.g. 387.0)
    - abs(value) >= 1     → 2 decimal places  (e.g. 3.87)
    - smaller             → up to 4 sig-figs, no sci notation
    """
    if value == 0.0:
        return "0"
    if value == int(value) and abs(value) < 1_000_000:
        return str(int(value))
    abs_v = abs(value)
    if abs_v >= 10_000:
        return f"{value:.0f}"
    if abs_v >= 100:
        return f"{value:.1f}"
    if abs_v >= 1:
        return f"{value:.2f}"
    if abs_v >= 0.01:
        return f"{value:.4f}"
    # Very small: use fixed notation with enough precision
    return f"{value:.6f}"


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "midnight": {
        "bg": "#000080", "fg": "#00ffff", "border": "#00ffff",
        "title_bg": "#00aaaa", "title_fg": "#000000",
        "accent": "#ffffff", "ok": "#00ff00", "err": "#ff5555",
        "btn_bg": "#004488", "btn_fg": "#ffffff",
        "log_bg": "#000060", "table_bg": "#000060",
        "header_bg": "#00aaaa", "header_fg": "#000000",
        "footer_bg": "#00aaaa", "footer_fg": "#000000",
        "highlight": "#ffff00", "paused": "#ff8800",
        "input_bg": "#001a4d",
        "tab_active_bg": "#00aaaa", "tab_active_fg": "#000000",
        "tab_inactive_bg": "#000060", "tab_inactive_fg": "#00ffff",
        "load_low": "#00ff00", "load_mid": "#ffaa00", "load_high": "#ff5555",
        "modal_bg": "#001a4d", "modal_border": "#00ffff",
    },
    "norton": {
        "bg": "#0000aa", "fg": "#ffff55", "border": "#ffff55",
        "title_bg": "#aa0000", "title_fg": "#ffffff",
        "accent": "#ffffff", "ok": "#55ff55", "err": "#ff5555",
        "btn_bg": "#555555", "btn_fg": "#ffffff",
        "log_bg": "#000088", "table_bg": "#000088",
        "header_bg": "#aa0000", "header_fg": "#ffffff",
        "footer_bg": "#aa0000", "footer_fg": "#ffffff",
        "highlight": "#ff5555", "paused": "#ff8800",
        "input_bg": "#00006a",
        "tab_active_bg": "#aa0000", "tab_active_fg": "#ffffff",
        "tab_inactive_bg": "#000088", "tab_inactive_fg": "#ffff55",
        "load_low": "#55ff55", "load_mid": "#ffaa00", "load_high": "#ff5555",
        "modal_bg": "#00006a", "modal_border": "#ffff55",
    },
    "amber": {
        "bg": "#1a1100", "fg": "#ffaa00", "border": "#ffaa00",
        "title_bg": "#aa7700", "title_fg": "#000000",
        "accent": "#ffcc00", "ok": "#ffcc00", "err": "#ff4400",
        "btn_bg": "#443300", "btn_fg": "#ffaa00",
        "log_bg": "#110a00", "table_bg": "#110a00",
        "header_bg": "#aa7700", "header_fg": "#000000",
        "footer_bg": "#aa7700", "footer_fg": "#000000",
        "highlight": "#ff4400", "paused": "#ff8800",
        "input_bg": "#2a1a00",
        "tab_active_bg": "#aa7700", "tab_active_fg": "#000000",
        "tab_inactive_bg": "#110a00", "tab_inactive_fg": "#ffaa00",
        "load_low": "#ffcc00", "load_mid": "#ff8800", "load_high": "#ff4400",
        "modal_bg": "#2a1a00", "modal_border": "#ffaa00",
    },
    "green": {
        "bg": "#001a00", "fg": "#00ff00", "border": "#00ff00",
        "title_bg": "#008800", "title_fg": "#000000",
        "accent": "#00ff00", "ok": "#00ff00", "err": "#ff4444",
        "btn_bg": "#003300", "btn_fg": "#00ff00",
        "log_bg": "#001100", "table_bg": "#001100",
        "header_bg": "#008800", "header_fg": "#000000",
        "footer_bg": "#008800", "footer_fg": "#000000",
        "highlight": "#ffff00", "paused": "#ff8800",
        "input_bg": "#002200",
        "tab_active_bg": "#008800", "tab_active_fg": "#000000",
        "tab_inactive_bg": "#001100", "tab_inactive_fg": "#00ff00",
        "load_low": "#00ff00", "load_mid": "#ffaa00", "load_high": "#ff4444",
        "modal_bg": "#002200", "modal_border": "#00ff00",
    },
    "modern": {
        "bg": "#1e1e2e", "fg": "#cdd6f4", "border": "#6c7086",
        "title_bg": "#585b70", "title_fg": "#cdd6f4",
        "accent": "#89b4fa", "ok": "#a6e3a1", "err": "#f38ba8",
        "btn_bg": "#313244", "btn_fg": "#cdd6f4",
        "log_bg": "#181825", "table_bg": "#181825",
        "header_bg": "#585b70", "header_fg": "#cdd6f4",
        "footer_bg": "#585b70", "footer_fg": "#cdd6f4",
        "highlight": "#f9e2af", "paused": "#fab387",
        "input_bg": "#11111b",
        "tab_active_bg": "#585b70", "tab_active_fg": "#cdd6f4",
        "tab_inactive_bg": "#181825", "tab_inactive_fg": "#6c7086",
        "load_low": "#a6e3a1", "load_mid": "#f9e2af", "load_high": "#f38ba8",
        "modal_bg": "#11111b", "modal_border": "#89b4fa",
    },
}
THEME_NAMES: List[str] = ["midnight", "norton", "amber", "green", "modern"]


# ---------------------------------------------------------------------------
# Keyboard shortcuts modal
# ---------------------------------------------------------------------------
# Shortcuts are split by context so each screen shows only relevant keys.
SHORTCUTS_MAIN: List[Tuple[str, str]] = [
    ("Main Screen", ""),
    ("F1", "Help"),
    ("F2", "Connect"),
    ("F3", "Disconnect"),
    ("F5", "Refresh Ports"),
    ("F6", "Fullscreen Monitor"),
    ("F4", "Details (Log / Stats / DBC / Trace)"),
    ("Space", "Pause / Resume Monitor"),
    ("Del", "Clear Monitor"),
    ("s", "Cycle Sort Mode"),
    ("f", "Focus Filter Input"),
    ("t", "Cycle Theme"),
    ("q", "Quit"),
]

SHORTCUTS_DETAILS: List[Tuple[str, str]] = [
    ("Details Screen", ""),
    ("Esc / q", "Back to Main"),
    ("t", "Cycle Theme"),
    ("", ""),
    ("DBC Decoder tab", ""),
    ("Load button / Enter", "Load .dbc file from path input"),
    ("Unload button", "Clear loaded DBC database"),
    ("", ""),
    ("Trace tab", ""),
    ("Record", "Start recording frames"),
    ("Pause", "Freeze display; frames keep recording, shown on Resume"),
    ("Stop", "End session; data kept for export"),
    ("Clear", "Discard all trace data"),
    ("Export", "Save trace in selected format (CSV / ASC / TRC / BLF)"),
]

SHORTCUTS_FULLSCREEN: List[Tuple[str, str]] = [
    ("Fullscreen Monitor", ""),
    ("Esc / q", "Back to Main"),
    ("Left click", "Toggle ID selection"),
    ("t", "Cycle Theme"),
    ("F1", "Help"),
]

SHORTCUTS_DISCOVERY: List[Tuple[str, str]] = [
    ("Signal Discovery  (F7)", ""),
    ("Esc / q", "Back to Main"),
    ("o", "Observe Start  (Noise-Baseline aufbauen)"),
    ("c", "Capture Start  (IDLE, OBSERVING oder RESULTS)"),
    ("x", "Capture Stop   (CAPTURING)"),
    ("s", "Cycle sort: ID → Δ-Bytes → Status"),
    ("u", "Toggle filter: All ↔ Unknown only"),
    ("t", "Cycle Theme"),
    ("F1", "Help"),
    ("", ""),
    ("Workflow mit Noise-Filter", ""),
    ("1.", "[o] Observe Start – Bus beobachten"),
    ("2.", "Warten bis Noise-IDs erkannt (Header zeigt Anzahl)"),
    ("3.", "[c] Capture Start – Snapshot 1 wird gezogen"),
    ("4.", "Aktion durchführen (Tür, Knopf, …)"),
    ("5.", "[x] Capture Stop – Ergebnisse ohne Noise-IDs"),
    ("", ""),
    ("Schnell-Workflow (ohne Noise-Filter)", ""),
    ("1.", "[c] direkt drücken → Aktion → [x] Stop"),
]

SHORTCUTS_CSS = """
ShortcutsScreen {
    align: center middle;
}
#shortcuts-dialog {
    width: 60;
    height: auto;
    border: double #00ffff;
    background: #001a4d;
    padding: 1 2;
}
#shortcuts-title {
    text-style: bold;
    color: #000000;
    background: #00aaaa;
    width: 100%;
    text-align: center;
    margin-bottom: 1;
}
#shortcuts-table {
    height: auto;
    background: #001a4d;
    color: #00ffff;
}
#shortcuts-close {
    margin-top: 1;
    width: 100%;
    background: #004488;
    color: #ffffff;
}
"""


class ShortcutsScreen(ModalScreen):
    """Modal overlay displaying context-sensitive keyboard shortcuts."""

    BINDINGS = [("escape", "dismiss", "Close"), ("f1", "dismiss", "Close")]
    CSS = SHORTCUTS_CSS

    def __init__(self, context: str = "main", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context = context

    def _shortcuts_for_context(self) -> List[Tuple[str, str]]:
        """Return the shortcut list matching the active screen context."""
        if self._context == "details":
            return SHORTCUTS_DETAILS
        if self._context == "fullscreen":
            return SHORTCUTS_FULLSCREEN
        if self._context == "discovery":
            return SHORTCUTS_DISCOVERY
        return SHORTCUTS_MAIN

    def compose(self) -> ComposeResult:
        """Build the shortcuts dialog layout."""
        with Vertical(id="shortcuts-dialog"):
            yield Static("KEYBOARD SHORTCUTS", id="shortcuts-title")
            tbl = DataTable(id="shortcuts-table", show_cursor=False, show_header=False)
            yield tbl
            yield Button("Close  [Esc]", id="shortcuts-close", variant="default")

    def on_mount(self) -> None:
        """Populate the shortcuts table after mount."""
        tbl = self.query_one("#shortcuts-table", DataTable)
        tbl.add_columns("Key", "Action")
        for key, action in self._shortcuts_for_context():
            if key == "" and action == "":
                tbl.add_row("", "")
            elif action == "":
                tbl.add_row(Text(key, style="bold #ffff00"), Text("", style=""))
            else:
                tbl.add_row(
                    Text(key, style="bold #ffff00"),
                    Text(action, style="#00ffff"),
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Close button."""
        if event.button.id == "shortcuts-close":
            self.dismiss()

    def apply_theme(self, t: Dict[str, str]) -> None:
        """Apply a theme colour dict to all styled elements of this modal."""
        try:
            dlg = self.query_one("#shortcuts-dialog")
            dlg.styles.background = t["modal_bg"]
            dlg.styles.border = ("double", t["modal_border"])
        except Exception:
            pass
        try:
            self.query_one("#shortcuts-title").styles.background = t["title_bg"]
            self.query_one("#shortcuts-title").styles.color = t["title_fg"]
        except Exception:
            pass
        try:
            tbl = self.query_one("#shortcuts-table", DataTable)
            tbl.styles.background = t["modal_bg"]
            tbl.styles.color = t["fg"]
        except Exception:
            pass
        try:
            btn = self.query_one("#shortcuts-close", Button)
            btn.styles.background = t["btn_bg"]
            btn.styles.color = t["btn_fg"]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reusable panel widgets
# ---------------------------------------------------------------------------
class ConnectionPanel(Container):
    """Panel holding port, speed, mode selectors and Connect/Disconnect buttons."""

    def compose(self) -> ComposeResult:
        """Build the connection panel layout."""
        yield Static("CONNECTION", classes="panel-title")
        with Horizontal(classes="form-row"):
            yield Label("Port:", classes="form-label")
            ports = detect_serial_ports()
            yield Select(
                [(p, p) for p in ports],
                id="port-select",
                value=ports[0] if ports else Select.BLANK,
                allow_blank=False,
            )
            yield Button("Refresh [F5]", id="btn-refresh", variant="default")
        with Horizontal(classes="form-row"):
            yield Label("Speed:", classes="form-label")
            yield Select(SPEED_OPTIONS, id="speed-select",
                         value=CANSpeed.SPEED_500K, allow_blank=False)
        with Horizontal(classes="form-row"):
            yield Label("Mode:", classes="form-label")
            yield Select(MODE_OPTIONS, id="mode-select",
                         value=CANMode.NORMAL, allow_blank=False)
        with Horizontal(classes="form-row"):
            yield Label("Frames:", classes="form-label")
            yield Select(
                FRAME_TYPE_OPTIONS,
                id="frame-type-select",
                value=None,   # Auto-Detect default
                allow_blank=False,
            )
        with Horizontal(classes="button-row"):
            yield Button("Connect [F2]", id="btn-connect", variant="success")
            yield Button("Disconnect [F3]", id="btn-disconnect",
                         variant="error", disabled=True)


class StatusPanel(Container):
    """Panel showing connection state, RX counters, bus load and queue depth."""

    def compose(self) -> ComposeResult:
        """Build the status panel layout."""
        yield Static("STATUS", classes="panel-title")
        yield Static("Disconnected", id="status-connection",
                     classes="status-disconnected")
        with Horizontal(classes="status-row"):
            yield Static("Port: -", id="status-port")
            yield Static("Speed: -", id="status-speed")
            yield Static("Mode: -", id="status-mode")
        with Horizontal(classes="status-row"):
            yield Static("RX Frames: 0", id="status-rx-count")
            yield Static("Frames/s: 0.0", id="status-fps")
            yield Static("Queue: 0", id="status-queue")
        with Horizontal(classes="status-row"):
            yield Static("Bus Load: --%", id="status-busload")
            yield Static("Unique IDs: 0", id="status-unique-ids")
            yield Static("", id="status-busload-bar")
        with Horizontal(classes="status-row"):
            yield Static("Frame: -", id="status-frame-type")


class MonitorPanel(Container):
    """Panel containing the live CAN frame DataTable."""

    COL_LABELS: Tuple[str, ...] = ("ID", "Type", "DLC", "Rate Hz", "Count", "Data")

    def compose(self) -> ComposeResult:
        """Build the monitor panel layout."""
        yield Static("LIVE MONITOR", id="monitor-title", classes="panel-title")
        yield DataTable(id="monitor-table", show_cursor=False)

    def on_mount(self) -> None:
        """Add table columns and store column key references."""
        table = self.query_one("#monitor-table", DataTable)
        col_keys = table.add_columns(*self.COL_LABELS)
        self.col_keys: Dict[str, object] = dict(zip(self.COL_LABELS, col_keys))


class FilterPanel(Container):
    """Panel with ID filter input, whitelist/blacklist toggle and sort button."""

    def compose(self) -> ComposeResult:
        """Build the filter and sort panel layout."""
        yield Static("FILTER & SORT", classes="panel-title")
        with Horizontal(classes="filter-row"):
            yield Label("IDs:", classes="form-label")
            yield Input(
                placeholder="0x123 0x7FFE  (space/comma, Enter to apply)",
                id="filter-input",
            )
            yield Button("Whitelist", id="btn-filter-mode", variant="default")
            yield Button("Clear", id="btn-filter-clear", variant="default")
            yield Button("Sort: ID \u2191", id="btn-sort", variant="default")


class SendPanel(Container):
    """Panel for single-shot and cyclic CAN frame transmission."""

    def compose(self) -> ComposeResult:
        """Build the transmit panel layout."""
        yield Static("TRANSMIT", classes="panel-title")
        with Horizontal(classes="send-row"):
            yield Label("ID:", classes="send-label")
            yield Input(placeholder="0x123", id="send-id", classes="send-input-id")
            yield Label("Ext", classes="send-label-sm")
            yield Checkbox("", id="send-extended", value=True)
        with Horizontal(classes="send-row"):
            yield Label("Data:", classes="send-label")
            yield Input(placeholder="DE AD BE EF  (hex bytes)", id="send-data",
                        classes="send-input-data")
        with Horizontal(classes="send-row"):
            yield Label("Period:", classes="send-label")
            yield Input(placeholder="100  (ms, 0 = single shot)", id="send-period",
                        classes="send-input-period")
            yield Label("Name:", classes="send-label-sm")
            yield Input(placeholder="task1", id="send-name",
                        classes="send-input-name")
        with Horizontal(classes="button-row"):
            yield Button("Send", id="btn-send", variant="success")
            yield Button("Start Cyclic", id="btn-cyclic-start", variant="default")
            yield Button("Stop Cyclic", id="btn-cyclic-stop",
                         variant="error", disabled=True)
        yield Static("", id="send-status", classes="send-status")


# ---------------------------------------------------------------------------
# Statistics tab widget
# ---------------------------------------------------------------------------
class StatisticsPanel(Container):
    """Statistics tab: top IDs by rate/count plus bus-load summary line."""

    COL_LABELS: Tuple[str, ...] = (
        "Rank", "ID", "Type", "DLC", "Rate Hz", "Count", "% of Bus"
    )

    def compose(self) -> ComposeResult:
        """Build the statistics panel layout."""
        yield Static("TOP IDs BY RATE", classes="panel-title")
        yield DataTable(id="stats-table", show_cursor=False)
        yield Static("", id="stats-summary", classes="stats-summary")

    def on_mount(self) -> None:
        """Add table columns."""
        table = self.query_one("#stats-table", DataTable)
        table.add_columns(*self.COL_LABELS)


# ---------------------------------------------------------------------------
# Trace tab widget
# ---------------------------------------------------------------------------
class TracePanel(Container):
    """Chronological frame trace with Record/Pause/Stop/Export controls."""

    COL_LABELS: Tuple[str, ...] = ("Time (s)", "CAN-ID", "Dir", "DLC", "Data")

    # Export format options shown in the Select widget
    _FMT_OPTIONS: List[Tuple[str, ExportFormat]] = [
        (ExportFormat.CSV.label, ExportFormat.CSV),
        (ExportFormat.ASC.label, ExportFormat.ASC),
        (ExportFormat.TRC.label, ExportFormat.TRC),
        (ExportFormat.BLF.label, ExportFormat.BLF),
    ]

    def compose(self) -> ComposeResult:
        """Build the trace panel layout."""
        yield Static("TRACE", classes="panel-title")
        with Horizontal(classes="trace-ctrl-row"):
            yield Button("Record", id="btn-trace-record", variant="success")
            yield Button("Pause", id="btn-trace-pause",
                         variant="default", disabled=True)
            yield Button("Stop", id="btn-trace-stop",
                         variant="error", disabled=True)
            yield Button("Clear", id="btn-trace-clear", variant="default")
            yield Static("IDLE", id="trace-state", classes="trace-state-idle")
        with Horizontal(classes="trace-export-row"):
            yield Select(
                options=self._FMT_OPTIONS,
                value=ExportFormat.CSV,
                id="trace-fmt-select",
            )
            yield Button("Export", id="btn-trace-export", variant="default")
        with Horizontal(classes="trace-info-row"):
            yield Static("Frames: 0", id="trace-count")
            yield Static("Elapsed: --", id="trace-elapsed")
            yield Static("", id="trace-warning", classes="trace-warning")
            yield Static("\u21a7 AUTO-SCROLL", id="trace-scroll-ind",
                         classes="trace-scroll-on")
        yield DataTable(id="trace-table", show_cursor=True)

    def on_mount(self) -> None:
        """Add table columns and initialise auto-scroll flag."""
        table = self.query_one("#trace-table", DataTable)
        table.add_columns(*self.COL_LABELS)
        self._auto_scroll = True

    def on_data_table_scroll(self, event) -> None:  # type: ignore[override]
        """Toggle auto-scroll when the user scrolls the trace table manually."""
        table = self.query_one("#trace-table", DataTable)
        at_end = table.scroll_y >= table.max_scroll_y - 1
        if self._auto_scroll and not at_end:
            self._auto_scroll = False
            self._update_scroll_indicator()
        elif not self._auto_scroll and at_end:
            self._auto_scroll = True
            self._update_scroll_indicator()

    def _update_scroll_indicator(self) -> None:
        """Refresh the AUTO-SCROLL / SCROLL LOCKED label."""
        ind = self.query_one("#trace-scroll-ind", Static)
        if self._auto_scroll:
            ind.update("\u21a7 AUTO-SCROLL")
            ind.remove_class("trace-scroll-off")
            ind.add_class("trace-scroll-on")
        else:
            ind.update("SCROLL LOCKED")
            ind.remove_class("trace-scroll-on")
            ind.add_class("trace-scroll-off")


# ---------------------------------------------------------------------------
# DBC Decoder tab widget
# ---------------------------------------------------------------------------
class DBCPanel(Container):
    """DBC Decoder tab: load a .dbc file, browse messages, show live signals."""

    MSG_COLS: Tuple[str, ...] = ("CAN-ID", "Message Name", "DLC", "Signals")

    def compose(self) -> ComposeResult:
        """Build the DBC panel layout."""
        yield Static("DBC FILE LOADER", classes="panel-title")
        with Horizontal(classes="dbc-load-row"):
            yield Input(placeholder="Path to .dbc file  (e.g. /home/user/my.dbc)",
                        id="dbc-path-input")
            yield Button("Load", id="btn-dbc-load", variant="success")
            yield Button("Unload", id="btn-dbc-unload", variant="error")
        yield Static("No DBC loaded", id="dbc-status")
        yield Static("MESSAGES IN DATABASE", classes="panel-title")
        yield DataTable(id="dbc-msg-table", show_cursor=True)

    def on_mount(self) -> None:
        """Add table columns."""
        msg_tbl = self.query_one("#dbc-msg-table", DataTable)
        msg_tbl.add_columns(*self.MSG_COLS)
        self._selected_ids: Set[int] = set()


# ---------------------------------------------------------------------------
# Details screen CSS
# ---------------------------------------------------------------------------
DETAILS_CSS = """
Screen {
    layout: grid;
    grid-size: 1 1;
    padding: 1;
    background: #000080;
    color: #00ffff;
}

Header { background: #00aaaa; color: #000000; }
Footer { dock: bottom; background: #00aaaa; color: #000000; }

TabbedContent { height: 1fr; }

TabPane {
    padding: 1;
    background: #000080;
}

.panel-title {
    text-style: bold;
    color: #000000;
    background: #00aaaa;
    padding: 0 1;
    width: 100%;
    text-align: center;
    margin-bottom: 1;
}

#event-log {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}

#stats-table {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}

.stats-summary {
    height: 3;
    color: #00ffff;
    padding: 0 1;
    margin-top: 1;
}

.dbc-placeholder {
    height: 1fr;
    color: #00aaaa;
    text-align: center;
    padding: 4 2;
}

/* ---- DBC tab ---- */
DBCPanel {
    height: 1fr;
    layout: vertical;
}

.dbc-load-row {
    height: 3;
    align-vertical: middle;
    margin-bottom: 0;
}
.dbc-load-row Input  { width: 1fr; }
.dbc-load-row Button { margin: 0 1; width: auto; min-width: 10; }

#dbc-status {
    height: 1;
    color: #00ffff;
    padding: 0 1;
    margin-bottom: 1;
}

#dbc-msg-table {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}

/* ---- Trace tab ---- */
TracePanel {
    height: 1fr;
    layout: vertical;
}

.trace-ctrl-row {
    height: 3;
    align-vertical: middle;
    margin-bottom: 0;
}
.trace-ctrl-row Button { margin: 0 1; }

.trace-export-row {
    height: 3;
    align-vertical: middle;
    margin-bottom: 0;
}
.trace-export-row Select { width: 1fr; }
.trace-export-row Button { margin: 0 1; width: auto; min-width: 12; }

.trace-info-row {
    height: 1;
    align-vertical: middle;
    margin-bottom: 1;
}
.trace-info-row Static { width: 1fr; }

.trace-state-idle      { color: #888888; text-style: bold; padding-left: 1; }
.trace-state-recording { color: #ff4444; text-style: bold; padding-left: 1; }
.trace-state-paused    { color: #ffaa00; text-style: bold; padding-left: 1; }

.trace-warning    { color: #ff4444; text-style: bold; }
.trace-scroll-on  { color: #00ff00; }
.trace-scroll-off { color: #ff8800; }

#trace-table {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}
#trace-count   { color: #00ffff; }
#trace-elapsed { color: #00ffff; }
"""


class DetailsScreen(Screen):
    """Secondary screen with TabbedContent: Event Log, Statistics, DBC, Trace."""

    # Context-sensitive bindings shown only while this screen is active
    BINDINGS = [
        ("escape", "pop_screen", "Back to Main"),
        ("q", "pop_screen", "Back"),
        ("t", "cycle_theme", "Theme"),
        ("f1", "show_shortcuts", "Help"),
    ]
    CSS = DETAILS_CSS

    def compose(self) -> ComposeResult:
        """Build the details screen layout."""
        yield Header()
        with TabbedContent(id="details-tabs"):
            with TabPane("Event Log", id="tab-eventlog"):
                yield Static("EVENT LOG", classes="panel-title")
                yield Log(id="event-log", max_lines=500)
            with TabPane("Statistics", id="tab-stats"):
                yield StatisticsPanel()
            with TabPane("DBC Decoder", id="tab-dbc"):
                yield DBCPanel()
            with TabPane("Trace", id="tab-trace"):
                yield TracePanel()
        yield Footer()

    def apply_theme(self, t: Dict[str, str]) -> None:
        """Apply a theme colour dict to all styled elements of this screen."""
        self.screen.styles.background = t["bg"]
        try:
            self.query_one(Header).styles.background = t["header_bg"]
            self.query_one(Header).styles.color = t["header_fg"]
            self.query_one(Footer).styles.background = t["footer_bg"]
            self.query_one(Footer).styles.color = t["footer_fg"]
        except Exception:
            pass  # Header/Footer not yet mounted - intentional
        try:
            self.query_one(TabbedContent).styles.background = t["bg"]
        except Exception:
            pass
        for pane in self.query(TabPane):
            pane.styles.background = t["bg"]
        for el in self.query(".panel-title"):
            el.styles.background = t["title_bg"]
            el.styles.color = t["title_fg"]
        try:
            log = self.query_one("#event-log", Log)
            log.styles.background = t["log_bg"]
            log.styles.color = t["fg"]
        except Exception:
            pass
        try:
            st = self.query_one("#stats-table", DataTable)
            st.styles.background = t["table_bg"]
            st.styles.color = t["fg"]
        except Exception:
            pass
        for wid_id in ("#stats-summary",):
            try:
                w = self.query_one(wid_id)
                w.styles.color = t["fg"]
                w.styles.background = t["bg"]
            except Exception:
                pass
        # DBC panel
        try:
            self.query_one(DBCPanel).styles.background = t["bg"]
        except Exception:
            pass
        for wid_id in ("#dbc-status", "#dbc-path-input"):
            try:
                w = self.query_one(wid_id)
                w.styles.color = t["fg"]
                w.styles.background = t["bg"]
            except Exception:
                pass
        for wid_id in ("#dbc-msg-table",):
            try:
                w = self.query_one(wid_id, DataTable)
                w.styles.background = t["table_bg"]
                w.styles.color = t["fg"]
            except Exception:
                pass
        try:
            self.query_one(TracePanel).styles.background = t["bg"]
        except Exception:
            pass
        try:
            tt = self.query_one("#trace-table", DataTable)
            tt.styles.background = t["table_bg"]
            tt.styles.color = t["fg"]
        except Exception:
            pass
        for wid_id in ("#trace-count", "#trace-elapsed"):
            try:
                self.query_one(wid_id).styles.color = t["fg"]
            except Exception:
                pass

    def action_pop_screen(self) -> None:
        """Return to the main screen, resume paused timers and clear details ref."""
        app = self.app  # type: ignore[attr-defined]
        app._details_screen = None
        app._details_active = False
        if app._monitor_timer:
            app._monitor_timer.resume()
        if app._status_timer:
            app._status_timer.resume()
        app.pop_screen()

    def action_cycle_theme(self) -> None:
        """Delegate theme cycling to the main app while Details is active."""
        self.app.action_cycle_theme()  # type: ignore[attr-defined]

    def action_show_shortcuts(self) -> None:
        """Delegate shortcut overlay to the main app with details context."""
        self.app.action_show_shortcuts(context="details")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Monitor Fullscreen CSS + Screen
# ---------------------------------------------------------------------------
MONITOR_FULLSCREEN_CSS = """
MonitorFullScreen {
    layout: horizontal;
    padding: 1;
    background: #000080;
    color: #00ffff;
}

Header { background: #00aaaa; color: #000000; }
Footer { dock: bottom; background: #00aaaa; color: #000000; }

#fs-left {
    width: 40;
    height: 100%;
    layout: vertical;
    border: solid #00ffff;
    padding: 1;
    margin-right: 1;
}

#fs-right {
    width: 1fr;
    height: 100%;
    layout: vertical;
    border: solid #00ffff;
    padding: 1;
}

.fs-title {
    text-style: bold;
    color: #000000;
    background: #00aaaa;
    padding: 0 1;
    width: 100%;
    text-align: center;
    margin-bottom: 1;
}

#fs-id-table {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}

#fs-signal-table {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}

#fs-hint {
    height: 1;
    color: #888888;
    text-align: center;
    margin-top: 1;
}
"""


class MonitorFullScreen(Screen):
    """Fullscreen split-view: left = ID list with selection, right = decoded signals."""

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("q", "pop_screen", "Back"),
        ("t", "cycle_theme", "Theme"),
        ("f1", "show_shortcuts", "Help"),
    ]
    CSS = MONITOR_FULLSCREEN_CSS

    # Column definitions
    ID_COLS: Tuple[str, ...] = ("Sel", "CAN-ID", "Name", "Rate Hz", "Count")
    SIG_COLS: Tuple[str, ...] = ("CAN-ID", "Name", "Signal", "Value", "Unit")

    def compose(self) -> ComposeResult:
        """Build the fullscreen monitor layout."""
        yield Header()
        with Container(id="fs-left"):
            yield Static("ID LIST  (Space = toggle)", classes="fs-title")
            yield DataTable(id="fs-id-table", show_cursor=True)
            yield Static("Space/Click to select  |  multiple allowed",
                         id="fs-hint")
        with Container(id="fs-right"):
            yield Static("LIVE DECODED SIGNALS", classes="fs-title")
            yield DataTable(id="fs-signal-table", show_cursor=False)
        yield Footer()

    def on_mount(self) -> None:
        """Initialise tables."""
        id_tbl = self.query_one("#fs-id-table", DataTable)
        id_tbl.add_columns(*self.ID_COLS)
        id_tbl.cursor_type = "row"
        id_tbl.show_cursor = True

        sig_tbl = self.query_one("#fs-signal-table", DataTable)
        sig_tbl.add_columns(*self.SIG_COLS)

        self._selected_ids: Set[int] = set()
        self._id_row_keys: Dict[int, str] = {}   # can_id → row_key string

        # References injected by the main app after push
        self._store: Optional[CANFrameStore] = None
        self._dbc: Optional[DBCDatabase] = None
        self._theme_idx: int = 0

        self._refresh_timer = self.set_interval(TIMER_DBC_S, self._refresh_tables)

        # Give focus to the ID table so Space works immediately without a click
        self.call_after_refresh(lambda: id_tbl.focus())

    def _refresh_tables(self) -> None:
        """Update the ID list and signal table from the live frame store."""
        if self._store is None:
            return
        rows = self._store.read()
        t = THEMES[THEME_NAMES[self._theme_idx]]
        fg = t["fg"]
        highlight = t["highlight"]
        self._refresh_id_table(rows, fg, highlight)
        self._refresh_signal_table(rows, fg)

    def _refresh_id_table(self, rows: dict, fg: str, highlight: str) -> None:
        """Rebuild the left ID list table."""
        try:
            id_tbl = self.query_one("#fs-id-table", DataTable)
        except Exception:
            return

        for can_id in sorted(rows.keys()):
            e = rows[can_id]
            frame = e["frame"]
            elapsed = e["last_ts"] - e["first_ts"]
            count = e["count"]
            rate = (count - 1) / elapsed if elapsed > 0 and count > 1 else 0.0

            id_str = (
                f"0x{can_id:08X}" if frame.is_extended else f"0x{can_id:03X}"
            )
            sel_str = "[X]" if can_id in self._selected_ids else "[ ]"
            sel_color = highlight if can_id in self._selected_ids else fg
            name_str = ""
            if self._dbc is not None and self._dbc.loaded:
                name_str = self._dbc.lookup_name(can_id) or ""

            row_key = str(can_id)
            if can_id not in self._id_row_keys:
                id_tbl.add_row(
                    Text(sel_str, style=sel_color),
                    Text(id_str, style=fg),
                    Text(name_str, style=fg),
                    Text(f"{rate:6.1f}", style=fg),
                    Text(str(count), style=fg),
                    key=row_key,
                )
                self._id_row_keys[can_id] = row_key
            else:
                col_keys = list(id_tbl.columns)
                id_tbl.update_cell(row_key, col_keys[0],
                                   Text(sel_str, style=sel_color),
                                   update_width=False)
                id_tbl.update_cell(row_key, col_keys[2],
                                   Text(name_str, style=fg), update_width=False)
                id_tbl.update_cell(row_key, col_keys[3],
                                   Text(f"{rate:6.1f}", style=fg), update_width=False)
                id_tbl.update_cell(row_key, col_keys[4],
                                   Text(str(count), style=fg), update_width=False)

    def _refresh_signal_table(self, rows: dict, fg: str) -> None:
        """Rebuild the right decoded signal table."""
        if not self._selected_ids:
            return
        try:
            sig_tbl = self.query_one("#fs-signal-table", DataTable)
        except Exception:
            return

        # Build all signal rows for selected IDs
        new_rows: List[Tuple] = []  # (row_key, id_str, name, sig_name, val_str, unit)
        for can_id in sorted(self._selected_ids):
            e = rows.get(can_id)
            if e is None:
                continue
            frame = e["frame"]
            id_str = (
                f"0x{can_id:08X}" if frame.is_extended else f"0x{can_id:03X}"
            )
            decoded_signals: Optional[List[DBCSignalValue]] = None
            if self._dbc is not None and self._dbc.loaded:
                decoded_signals = self._dbc.decode(can_id, bytes(frame.data))

            if decoded_signals:
                for sig in decoded_signals:
                    new_rows.append((
                        f"{can_id}_{sig.name}",
                        id_str, "",
                        sig.name,
                        (_format_signal_value(sig.value)),
                        sig.unit,
                    ))
            else:
                # Raw hex fallback
                hex_str = bytes(frame.data).hex(" ").upper()
                msg_name = ""
                if self._dbc is not None and self._dbc.loaded:
                    msg_name = self._dbc.lookup_name(can_id) or "(not in DBC)"
                new_rows.append((
                    f"{can_id}_raw",
                    id_str,
                    msg_name,
                    "raw",
                    hex_str,
                    "",
                ))

        existing_keys = {str(r.key.value) for r in sig_tbl.rows.values()}
        desired_keys = {r[0] for r in new_rows}

        for k in existing_keys - desired_keys:
            try:
                sig_tbl.remove_row(k)
            except Exception:
                pass

        col_keys = list(sig_tbl.columns)
        for row_key, id_str, msg_name, sig_name, val_str, unit in new_rows:
            if row_key in existing_keys:
                sig_tbl.update_cell(row_key, col_keys[3],
                                    Text(val_str, style=fg), update_width=False)
            else:
                sig_tbl.add_row(
                    Text(id_str, style=fg),
                    Text(msg_name, style=fg),
                    Text(sig_name, style=fg),
                    Text(val_str, style=fg),
                    Text(unit, style=fg),
                    key=row_key,
                )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter key on a row → toggle selection in the ID table."""
        if event.data_table.id == "fs-id-table":
            self._toggle_row_by_key(str(event.row_key.value))

    def on_click(self, event) -> None:  # type: ignore[override]
        """Single left-click anywhere in the ID table → toggle the clicked row.

        We resolve which DataTable row was clicked by comparing the mouse Y
        position against the table's content offset.  This avoids the
        double-click problem caused by waiting for RowSelected.
        """
        try:
            id_tbl = self.query_one("#fs-id-table", DataTable)
        except Exception:
            return

        # Only act on clicks inside the ID table widget
        try:
            region = id_tbl.content_region
        except Exception:
            return
        if not region.contains(event.screen_x, event.screen_y):
            return

        # Calculate which row was clicked from the Y offset within the table
        # content_region.y is the absolute top of the scrollable content area
        relative_y = event.screen_y - region.y + int(id_tbl.scroll_y)
        if relative_y < 0:
            return

        row_index = relative_y  # each row is 1 cell high in Textual DataTable
        row_keys = list(id_tbl.rows.keys())
        if row_index >= len(row_keys):
            return

        self._toggle_row_by_key(str(row_keys[row_index]))

    def _toggle_row_by_key(self, row_key: str) -> None:
        """Toggle the selected state of the row identified by row_key."""
        try:
            can_id = int(row_key)
        except ValueError:
            return
        if can_id in self._selected_ids:
            self._selected_ids.discard(can_id)
        else:
            self._selected_ids.add(can_id)
        # Force immediate visual refresh
        self._refresh_tables()

    def apply_theme(self, t: Dict[str, str]) -> None:
        """Apply theme colours to this screen."""
        try:
            self.screen.styles.background = t["bg"]
            self.screen.styles.color = t["fg"]
            self.query_one(Header).styles.background = t["header_bg"]
            self.query_one(Header).styles.color = t["header_fg"]
            self.query_one(Footer).styles.background = t["footer_bg"]
            self.query_one(Footer).styles.color = t["footer_fg"]
        except Exception:
            pass
        for wid_id in ("#fs-left", "#fs-right"):
            try:
                w = self.query_one(wid_id)
                w.styles.background = t["bg"]
                w.styles.border = ("solid", t["border"])
            except Exception:
                pass
        for el in self.query(".fs-title"):
            el.styles.background = t["title_bg"]
            el.styles.color = t["title_fg"]
        for wid_id in ("#fs-id-table", "#fs-signal-table"):
            try:
                w = self.query_one(wid_id, DataTable)
                w.styles.background = t["table_bg"]
                w.styles.color = t["fg"]
            except Exception:
                pass

    def action_pop_screen(self) -> None:
        """Return to main screen, resume paused timers."""
        app = self.app  # type: ignore[attr-defined]
        app._fullscreen_monitor = None
        app._details_active = False
        if app._monitor_timer:
            app._monitor_timer.resume()
        if app._status_timer:
            app._status_timer.resume()
        app.pop_screen()

    def action_cycle_theme(self) -> None:
        """Delegate theme cycling to the main app."""
        self.app.action_cycle_theme()  # type: ignore[attr-defined]

    def action_show_shortcuts(self) -> None:
        """Delegate shortcut overlay to the main app with fullscreen context."""
        self.app.action_show_shortcuts(context="fullscreen")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Signal Discovery Screen CSS + Screen
# ---------------------------------------------------------------------------
DISCOVERY_CSS = """
DiscoveryScreen {
    layout: vertical;
    padding: 0 1;
    background: #000080;
    color: #00ffff;
}

Header { background: #00aaaa; color: #000000; }
Footer { dock: bottom; background: #00aaaa; color: #000000; }

#disc-control {
    height: auto;
    layout: vertical;
    border: solid #00ffff;
    padding: 1;
    margin-bottom: 1;
    background: #000080;
}

#disc-control-title {
    text-style: bold;
    color: #000000;
    background: #00aaaa;
    padding: 0 1;
    width: 100%;
    text-align: center;
    margin-bottom: 1;
}

#disc-btn-row {
    height: 3;
    align-vertical: middle;
    margin-bottom: 1;
}

#disc-btn-row Button {
    margin-right: 2;
}

#disc-hint {
    height: 1;
    color: #888888;
}

#disc-results {
    height: 1fr;
    layout: vertical;
    border: solid #00ffff;
    padding: 1;
    background: #000080;
}

#disc-results-title {
    text-style: bold;
    color: #000000;
    background: #00aaaa;
    padding: 0 1;
    width: 100%;
    text-align: center;
    margin-bottom: 1;
}

#disc-filter-row {
    height: 3;
    align-vertical: middle;
    margin-bottom: 1;
}

#disc-filter-row Button {
    margin-right: 1;
}

#disc-table {
    height: 1fr;
    background: #000060;
    color: #00ffff;
}
"""

# Discovery table column definitions
_DISC_COLS: Tuple[str, ...] = (
    "CAN-ID", "Status", "Δ Bytes", "VORHER", "NACHHER"
)

# Sort modes for the discovery table: (label, key, reverse)
_DISC_SORT_MODES: List[Tuple[str, str, bool]] = [
    ("ID ↑", "id", False),
    ("Δ-Bytes ↓", "delta", True),
    ("Status", "status", False),
]


class DiscoveryScreen(Screen):
    """F7 – Signal Discovery: compare two CAN bus snapshots and show changed IDs."""

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("q", "pop_screen", "Back"),
        ("o", "observe_start", "o Observe"),
        ("c", "capture_start", "c Capture"),
        ("x", "capture_stop", "x Stop"),
        ("s", "cycle_sort", "s Sort"),
        ("u", "toggle_unknown_filter", "u Unknown only"),
        ("t", "cycle_theme", "t Theme"),
        ("f1", "show_shortcuts", "F1 Help"),
    ]
    CSS = DISCOVERY_CSS

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Shared references injected by the main app after push_screen
        self._store: Optional[CANFrameStore] = None
        self._dbc: Optional[DBCDatabase] = None
        self._theme_idx: int = 0

        # Discovery state machine
        self._disc_state: DiscoveryState = DiscoveryState.IDLE
        self._snap1: Dict[int, bytes] = {}          # Baseline snapshot (Capture Start)
        self._snap2: Dict[int, bytes] = {}          # Post-action snapshot (Capture Stop)
        self._snap1_ext: Dict[int, bool] = {}       # is_extended per id in snap1
        self._snap2_ext: Dict[int, bool] = {}       # is_extended per id in snap2
        self._results: List[ChangeDelta] = []       # Computed deltas
        self._blink_state: bool = False
        self._timer: Optional[Timer] = None

        # Noise baseline (Observe phase)
        # IDs that changed during OBSERVING are excluded from RESULTS.
        self._noise_ids: Set[int] = set()
        self._observe_snap: Dict[int, bytes] = {}   # Data at Observe Start

        # UI filter / sort state
        self._sort_idx: int = 0
        self._unknown_only: bool = False
        self._table_row_keys: List[int] = []        # CAN-IDs in table order

        # Warning flag: store was cleared during capture
        self._cleared_during_capture: bool = False

    # -----------------------------------------------------------------------
    # Composition
    # -----------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        """Build the Signal Discovery screen layout."""
        yield Header()

        with Container(id="disc-control"):
            yield Static("CAPTURE CONTROL", id="disc-control-title")
            with Horizontal(id="disc-btn-row"):
                yield Button(
                    "👁 OBSERVE  [o]",
                    id="btn-disc-observe",
                    variant="warning",
                )
                yield Button(
                    "▶ CAPTURE START  [c]",
                    id="btn-disc-start",
                    variant="success",
                )
                yield Button(
                    "■ CAPTURE STOP  [x]",
                    id="btn-disc-stop",
                    variant="error",
                    disabled=True,
                )
            yield Static(
                "Tipp: [o] Observe (Noise lernen) → [c] Capture → Aktion → [x] Stop",
                id="disc-hint",
            )

        with Container(id="disc-results"):
            yield Static("ERGEBNISSE", id="disc-results-title")
            with Horizontal(id="disc-filter-row"):
                yield Button("Sort: ID ↑", id="btn-disc-sort", variant="default")
                yield Button("Alle", id="btn-disc-filter", variant="default")
            yield DataTable(id="disc-table", show_cursor=True)

        yield Footer()

    def on_mount(self) -> None:
        """Initialise the results table columns."""
        tbl = self.query_one("#disc-table", DataTable)
        tbl.add_columns(*_DISC_COLS)
        self._set_header_idle()

    # -----------------------------------------------------------------------
    # Header helpers
    # -----------------------------------------------------------------------
    def _set_header_idle(self) -> None:
        """Set the app sub_title for the IDLE state."""
        self.app.sub_title = (  # type: ignore[attr-defined]
            "Signal Discovery  –  [o] Observe (Noise), dann [c] Capture starten."
        )

    def _set_header_observing(self, noise: int) -> None:
        """Set the app sub_title for the OBSERVING state with live noise counter."""
        dot = "\u25cf" if self._blink_state else "\u25cb"  # ● / ○
        self.app.sub_title = (  # type: ignore[attr-defined]
            f"{dot} OBSERVING...  Noise erkannt: {noise} IDs  │  "
            "[c] drücken um Capture zu starten"
        )

    def _set_header_capturing(self, baseline: int, delta: int) -> None:
        """Set the app sub_title for the CAPTURING state with live counters."""
        dot = "\u25cf" if self._blink_state else "\u25cb"  # ● / ○
        noise_str = f"  │  Noise-Filter: {len(self._noise_ids)} IDs" if self._noise_ids else ""
        self.app.sub_title = (  # type: ignore[attr-defined]
            f"{dot} CAPTURING...  Baseline: {baseline} IDs  │  "
            f"Δ seit Start: {delta} IDs geändert{noise_str}"
        )

    def _set_header_results(self, changed: int, unknown: int, filtered: int) -> None:
        """Set the app sub_title for the RESULTS state."""
        noise_str = f"  │  {filtered} Noise gefiltert" if filtered > 0 else ""
        if changed == 0:
            if filtered > 0:
                self.app.sub_title = (  # type: ignore[attr-defined]
                    f"Signal Discovery  –  ⓘ Alle Änderungen waren Noise ({filtered} IDs). "
                    "[o] Observe verkürzen oder [c] ohne Observe starten."
                )
            else:
                self.app.sub_title = (  # type: ignore[attr-defined]
                    "Signal Discovery  –  ⓘ Keine Änderungen gefunden.  "
                    "[c] für neuen Capture."
                )
        else:
            self.app.sub_title = (  # type: ignore[attr-defined]
                f"Signal Discovery  –  ✓ Fertig.  "
                f"{changed} geändert  │  {unknown} unbekannt"
                f"{noise_str}  │  [c] für neuen Capture."
            )

    # -----------------------------------------------------------------------
    # Capture state machine
    # -----------------------------------------------------------------------
    def action_observe_start(self) -> None:
        """Start the Observe phase to build a noise baseline (IDLE or RESULTS).

        During OBSERVING the timer continuously compares current store data
        against the observe snapshot and accumulates any changing CAN-ID into
        _noise_ids.  Press [c] to end Observe and immediately start Capture.
        """
        if self._disc_state == DiscoveryState.CAPTURING:
            return
        if self._disc_state == DiscoveryState.OBSERVING:
            return  # already observing
        if self._store is None:
            self.app.sub_title = (  # type: ignore[attr-defined]
                "Signal Discovery  –  ⚠ Kein Store verfügbar."
            )
            return

        snap = self._store.read()
        if not snap:
            self.app.sub_title = (  # type: ignore[attr-defined]
                "Signal Discovery  –  ⚠ Keine Frames im Monitor – "
                "erst verbinden und Frames empfangen."
            )
            return

        # Reset noise state and take observe baseline snapshot
        self._noise_ids = set()
        self._observe_snap = {cid: dict(e)["frame"].data for cid, e in snap.items()}

        # Clear results table (new session starting)
        tbl = self.query_one("#disc-table", DataTable)
        tbl.clear()
        self._table_row_keys = []
        self._results = []

        # Transition to OBSERVING
        self._disc_state = DiscoveryState.OBSERVING
        self.query_one("#btn-disc-observe", Button).disabled = True
        self.query_one("#btn-disc-start", Button).disabled = False   # [c] ends Observe
        self.query_one("#btn-disc-stop", Button).disabled = True

        self._blink_state = False
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(TIMER_DISCOVERY_S, self._tick_observe)
        self._set_header_observing(noise=0)

    def action_capture_start(self) -> None:
        """Start Capture phase (valid in IDLE, OBSERVING and RESULTS).

        If called from OBSERVING: stops the observe timer, preserves _noise_ids,
        and immediately begins Capture.
        If called from IDLE or RESULTS: no noise filter is active (_noise_ids empty).
        """
        if self._disc_state == DiscoveryState.CAPTURING:
            return
        if self._store is None:
            self.app.sub_title = (  # type: ignore[attr-defined]
                "Signal Discovery  –  ⚠ Kein Store verfügbar."
            )
            return

        # Stop observe timer if transitioning from OBSERVING
        if self._disc_state == DiscoveryState.OBSERVING:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None

        # If jumping straight from IDLE/RESULTS without Observe → clear noise
        if self._disc_state in (DiscoveryState.IDLE, DiscoveryState.RESULTS):
            self._noise_ids = set()

        # Guard: store must have frames
        snap = self._store.read()
        if not snap:
            self.app.sub_title = (  # type: ignore[attr-defined]
                "Signal Discovery  –  ⚠ Keine Frames im Monitor – "
                "erst verbinden und Frames empfangen."
            )
            return

        # Take Snapshot 1
        self._snap1 = {cid: dict(e)["frame"].data for cid, e in snap.items()}
        self._snap1_ext = {cid: dict(e)["frame"].is_extended for cid, e in snap.items()}
        self._snap2 = {}
        self._snap2_ext = {}
        self._results = []
        self._cleared_during_capture = False

        # Clear results table
        tbl = self.query_one("#disc-table", DataTable)
        tbl.clear()
        self._table_row_keys = []

        # Transition to CAPTURING
        self._disc_state = DiscoveryState.CAPTURING
        self.query_one("#btn-disc-observe", Button).disabled = False
        self.query_one("#btn-disc-start", Button).disabled = True
        self.query_one("#btn-disc-stop", Button).disabled = False

        # Start live-counter timer
        self._blink_state = False
        self._timer = self.set_interval(TIMER_DISCOVERY_S, self._tick_capture)
        self._set_header_capturing(baseline=len(self._snap1), delta=0)

    def action_capture_stop(self) -> None:
        """Stop the capture, compute deltas (minus noise) and populate results."""
        if self._disc_state != DiscoveryState.CAPTURING:
            return
        if self._store is None:
            return

        # Stop the live-counter timer
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        # Take Snapshot 2
        snap2 = self._store.read()
        self._snap2 = {cid: dict(e)["frame"].data for cid, e in snap2.items()}
        self._snap2_ext = {
            cid: dict(e)["frame"].is_extended for cid, e in snap2.items()
        }

        # Compute deltas (noise IDs are filtered inside _compute_deltas)
        all_deltas, filtered_count = self._compute_deltas()
        self._results = all_deltas

        # Transition to RESULTS
        self._disc_state = DiscoveryState.RESULTS
        self.query_one("#btn-disc-observe", Button).disabled = False
        self.query_one("#btn-disc-start", Button).disabled = False
        self.query_one("#btn-disc-stop", Button).disabled = True

        self._render_results()

        unknown = sum(1 for d in self._results if d.dbc_name is None)
        self._set_header_results(
            changed=len(self._results),
            unknown=unknown,
            filtered=filtered_count,
        )

        # Warn if store was cleared during capture
        if self._cleared_during_capture:
            self.app.sub_title = (  # type: ignore[attr-defined]
                "Signal Discovery  –  ⚠ Monitor wurde während Capture geleert – "
                "Ergebnisse möglicherweise unvollständig."
            )

    def _tick_observe(self) -> None:
        """Timer callback during OBSERVING: detect changing IDs and update noise set."""
        if self._disc_state != DiscoveryState.OBSERVING or self._store is None:
            return
        self._blink_state = not self._blink_state

        current = self._store.read()
        for cid, entry in current.items():
            data_now = dict(entry)["frame"].data
            # A CAN-ID is noise if it changed since observe start OR appeared new
            if cid not in self._observe_snap or data_now != self._observe_snap[cid]:
                self._noise_ids.add(cid)
                # Update observe baseline so continued change is tracked correctly
                self._observe_snap[cid] = data_now

        self._set_header_observing(noise=len(self._noise_ids))

    def _tick_capture(self) -> None:
        """Timer callback during CAPTURING: update blink + live Δ-counter."""
        if self._disc_state != DiscoveryState.CAPTURING or self._store is None:
            return
        self._blink_state = not self._blink_state

        current = self._store.read()
        delta = 0
        for cid, entry in current.items():
            data_now = dict(entry)["frame"].data
            if cid not in self._snap1:
                delta += 1  # new frame appeared
            elif data_now != self._snap1[cid]:
                delta += 1

        self._set_header_capturing(baseline=len(self._snap1), delta=delta)

    # -----------------------------------------------------------------------
    # Delta computation
    # -----------------------------------------------------------------------
    def _compute_deltas(self) -> Tuple[List[ChangeDelta], int]:
        """Compare Snapshot 1 and Snapshot 2, filter noise IDs, return deltas.

        Returns:
            Tuple of (list of ChangeDelta, number of noise-filtered IDs).

        Handles three cases per CAN-ID:
        - Frame changed: present in both snapshots with different data.
        - Frame appeared: present only in Snapshot 2.
        - Frame disappeared: present only in Snapshot 1.
        Frames with identical data and frames in _noise_ids are excluded.
        """
        all_ids: Set[int] = set(self._snap1.keys()) | set(self._snap2.keys())
        deltas: List[ChangeDelta] = []
        filtered_count: int = 0

        for cid in all_ids:
            before = self._snap1.get(cid, b"")
            after = self._snap2.get(cid, b"")

            if before == after:
                continue  # no change – skip

            # Skip noise IDs (changed during Observe phase)
            if cid in self._noise_ids:
                filtered_count += 1
                continue

            # Compute changed byte indices
            max_len = max(len(before), len(after))
            changed: List[int] = []
            for i in range(max_len):
                b_val = before[i] if i < len(before) else -1
                a_val = after[i] if i < len(after) else -1
                if b_val != a_val:
                    changed.append(i)

            is_ext = self._snap2_ext.get(cid, self._snap1_ext.get(cid, False))
            dbc_name: Optional[str] = None
            if self._dbc is not None and self._dbc.loaded:
                dbc_name = self._dbc.lookup_name(cid)

            deltas.append(ChangeDelta(
                can_id=cid,
                is_extended=is_ext,
                before_data=before,
                after_data=after,
                changed_indices=changed,
                dbc_name=dbc_name,
            ))

        return deltas, filtered_count

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------
    def _render_results(self) -> None:
        """Populate the results DataTable from self._results respecting filters/sort."""
        tbl = self.query_one("#disc-table", DataTable)
        tbl.clear()
        self._table_row_keys = []

        sorted_results = self._sorted_results()

        for delta in sorted_results:
            if self._unknown_only and delta.dbc_name is not None:
                continue

            id_str = (
                f"0x{delta.can_id:08X}" if delta.is_extended
                else f"0x{delta.can_id:03X}"
            )
            status_str = (
                f"✓ {delta.dbc_name}" if delta.dbc_name else "⚠ Unbekannt"
            )
            delta_str = " ".join(f"[{i}]" for i in delta.changed_indices)
            before_cell = self._build_diff_cell(
                delta.before_data, delta.changed_indices, "before"
            )
            after_cell = self._build_diff_cell(
                delta.after_data, delta.changed_indices, "after"
            )

            tbl.add_row(
                id_str, status_str, delta_str, before_cell, after_cell,
                key=str(delta.can_id),
            )
            self._table_row_keys.append(delta.can_id)

    def _build_diff_cell(
        self,
        data: bytes,
        changed_indices: List[int],
        side: str,
    ) -> Text:
        """Build a Rich Text object for the VORHER or NACHHER column.

        Changed bytes are coloured:
        - VORHER (before): theme 'err' colour (red / orange)
        - NACHHER (after): theme 'ok' colour (green)
        Missing bytes shown as '??' when the other snapshot was longer.

        Args:
            data:            Raw byte data for this snapshot.
            changed_indices: Byte positions that differ.
            side:            'before' or 'after' – determines highlight colour.
        """
        t = THEMES[THEME_NAMES[self._theme_idx]]
        fg = t["fg"]
        change_color = t["err"] if side == "before" else t["ok"]

        # Maximum length = length of data or highest changed index + 1
        max_len = max(
            len(data),
            (max(changed_indices) + 1) if changed_indices else 0,
        )

        text = Text()
        for i in range(max_len):
            if i < len(data):
                byte_str = f"{data[i]:02X}"
            else:
                byte_str = "??"  # byte missing in this snapshot

            color = change_color if i in changed_indices else fg
            text.append(byte_str, style=color)
            if i < max_len - 1:
                text.append(" ", style=fg)

        return text

    # -----------------------------------------------------------------------
    # Sort and filter
    # -----------------------------------------------------------------------
    def _sorted_results(self) -> List[ChangeDelta]:
        """Return self._results sorted according to the current sort mode."""
        _, key, reverse = _DISC_SORT_MODES[self._sort_idx]
        if key == "id":
            return sorted(self._results, key=lambda d: d.can_id, reverse=reverse)
        if key == "delta":
            return sorted(
                self._results, key=lambda d: len(d.changed_indices), reverse=reverse
            )
        # key == "status": known (dbc_name set) first, then unknown
        return sorted(
            self._results,
            key=lambda d: (0 if d.dbc_name else 1),
            reverse=reverse,
        )

    def action_cycle_sort(self) -> None:
        """Advance to the next sort mode and re-render the results table."""
        self._sort_idx = (self._sort_idx + 1) % len(_DISC_SORT_MODES)
        label, _, _ = _DISC_SORT_MODES[self._sort_idx]
        self.query_one("#btn-disc-sort", Button).label = f"Sort: {label}"
        if self._disc_state == DiscoveryState.RESULTS:
            self._render_results()

    def action_toggle_unknown_filter(self) -> None:
        """Toggle between showing all results and unknown-only results."""
        self._unknown_only = not self._unknown_only
        self.query_one("#btn-disc-filter", Button).label = (
            "Nur ⚠ Unbekannte" if self._unknown_only else "Alle"
        )
        if self._disc_state == DiscoveryState.RESULTS:
            self._render_results()

    # -----------------------------------------------------------------------
    # Button handler
    # -----------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button presses to the appropriate action."""
        bid = event.button.id
        if bid == "btn-disc-observe":
            self.action_observe_start()
        elif bid == "btn-disc-start":
            self.action_capture_start()
        elif bid == "btn-disc-stop":
            self.action_capture_stop()
        elif bid == "btn-disc-sort":
            self.action_cycle_sort()
        elif bid == "btn-disc-filter":
            self.action_toggle_unknown_filter()

    # -----------------------------------------------------------------------
    # Theme
    # -----------------------------------------------------------------------
    def apply_theme(self, t: Dict[str, str]) -> None:
        """Apply a theme colour dict to all styled elements of this screen."""
        try:
            self.screen.styles.background = t["bg"]
            self.screen.styles.color = t["fg"]
        except Exception:
            pass
        for wid_id in ("#disc-control", "#disc-results"):
            try:
                w = self.query_one(wid_id)
                w.styles.background = t["bg"]
                w.styles.border = ("solid", t["border"])
            except Exception:
                pass
        for wid_id in ("#disc-control-title", "#disc-results-title"):
            try:
                w = self.query_one(wid_id)
                w.styles.background = t["title_bg"]
                w.styles.color = t["title_fg"]
            except Exception:
                pass
        try:
            tbl = self.query_one("#disc-table", DataTable)
            tbl.styles.background = t["table_bg"]
            tbl.styles.color = t["fg"]
        except Exception:
            pass
        try:
            self.query_one(Header).styles.background = t["header_bg"]
            self.query_one(Header).styles.color = t["header_fg"]
            self.query_one(Footer).styles.background = t["footer_bg"]
            self.query_one(Footer).styles.color = t["footer_fg"]
        except Exception:
            pass
        for btn in self.query(Button):
            try:
                btn.styles.background = t["btn_bg"]
                btn.styles.color = t["btn_fg"]
            except Exception:
                pass
        # Re-render table so diff colours match new theme
        if self._disc_state == DiscoveryState.RESULTS:
            self._render_results()

    # -----------------------------------------------------------------------
    # Navigation / delegation
    # -----------------------------------------------------------------------
    def action_pop_screen(self) -> None:
        """Return to main screen, stop any running timer, resume main timers."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        app = self.app  # type: ignore[attr-defined]
        app._discovery_screen = None
        app._details_active = False
        app.sub_title = "v1.3.0"
        if app._monitor_timer:
            app._monitor_timer.resume()
        if app._status_timer:
            app._status_timer.resume()
        app.pop_screen()

    def action_cycle_theme(self) -> None:
        """Delegate theme cycling to the main app."""
        self.app.action_cycle_theme()  # type: ignore[attr-defined]

    def action_show_shortcuts(self) -> None:
        """Show the keyboard shortcuts overlay for the discovery context."""
        self.app.action_show_shortcuts(context="discovery")  # type: ignore[attr-defined]

    def notify_store_cleared(self) -> None:
        """Called by the main app when Del clears the store during an active phase."""
        if self._disc_state == DiscoveryState.CAPTURING:
            self._cleared_during_capture = True
        elif self._disc_state == DiscoveryState.OBSERVING:
            # Reset observe baseline – noise accumulation continues from fresh data
            self._observe_snap = {}
            self._noise_ids = set()


# ---------------------------------------------------------------------------
# Main screen CSS
# ---------------------------------------------------------------------------
MAIN_CSS = """
Screen {
    layout: horizontal;
    padding: 1;
    background: #000080;
    color: #00ffff;
}

Header { background: #00aaaa; color: #000000; }
Footer { dock: bottom; background: #00aaaa; color: #000000; }

#left-col {
    width: 1fr;
    height: 100%;
    layout: vertical;
}

#right-col {
    width: 90;
    height: 100%;
    layout: vertical;
    margin-left: 1;
}

ConnectionPanel {
    border: solid #00ffff; padding: 1;
    height: auto; background: #000080; color: #00ffff;
    margin-bottom: 1;
}
MonitorPanel {
    border: solid #00ffff; padding: 1;
    height: 1fr; background: #000080;
}
StatusPanel {
    border: solid #00ffff; padding: 1;
    height: auto; background: #000080; color: #00ffff;
    margin-bottom: 1;
}
SendPanel {
    border: solid #00ffff; padding: 1;
    height: auto; background: #000080; color: #00ffff;
    margin-bottom: 1;
}
FilterPanel {
    border: solid #00ffff; padding: 0 1;
    height: auto; background: #000080; color: #00ffff;
}

.panel-title {
    text-style: bold; color: #000000; background: #00aaaa;
    padding: 0 1; width: 100%; text-align: center; margin-bottom: 1;
}
.form-row    { height: 3; align-vertical: middle; margin-bottom: 0; }
.filter-row  { height: 3; align-vertical: middle; }
.send-row    { height: 3; align-vertical: middle; margin-bottom: 0; }

.form-label    { width: 8; padding-top: 1; color: #ffffff; }
.send-label    { width: 7; padding-top: 1; color: #ffffff; }
.send-label-sm { width: 5; padding-top: 1; color: #ffffff; }

.button-row        { height: auto; margin-top: 1; align-horizontal: center; }
.button-row Button { margin: 0 1; background: #004488; color: #ffffff; }

.status-row        { height: auto; }
.status-row Static { width: 1fr; }

.status-disconnected {
    color: #ff5555; text-style: bold; text-align: center; margin-bottom: 1;
}
.status-connected {
    color: #00ff00; text-style: bold; text-align: center; margin-bottom: 1;
}

#port-select  { width: 1fr; }
#speed-select { width: 1fr; }
#mode-select  { width: 1fr; }

#btn-refresh      { width: auto; min-width: 13; background: #004488; color: #ffffff; }
#filter-input     { width: 1fr; background: #000060; color: #00ffff; }
#btn-filter-mode  { width: auto; min-width: 12; background: #004488; color: #ffffff; }
#btn-filter-clear { width: auto; min-width: 8;  background: #004488; color: #ffffff; }
#btn-sort         { width: auto; min-width: 12; background: #004488; color: #ffffff; }

.send-input-id     { width: 12; background: #001a4d; color: #00ffff; }
.send-input-data   { width: 1fr; background: #001a4d; color: #00ffff; }
.send-input-period { width: 10; background: #001a4d; color: #00ffff; }
.send-input-name   { width: 1fr; background: #001a4d; color: #00ffff; }
.send-status       { height: 1; color: #00ff00; text-align: center; }

#monitor-table { height: 1fr; background: #000060; color: #00ffff; }

#status-busload     { width: 1fr; }
#status-unique-ids  { width: 1fr; }
#status-busload-bar { width: 1fr; }

Select { background: #004488; color: #00ffff; }
"""


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class CANBusTUI(App):
    """Waveshare CAN Bus TUI - main application class."""

    TITLE = "Waveshare CAN Bus Tool"
    SUB_TITLE = "v1.3.0"
    CSS = MAIN_CSS

    # Context-sensitive bindings for the main screen only.
    # F-key actions map to named action methods; the footer displays only these.
    BINDINGS = [
        ("f1", "show_shortcuts", "F1 Help"),
        ("f2", "connect", "F2 Connect"),
        ("f3", "disconnect", "F3 Disconnect"),
        ("f5", "refresh_ports", "F5 Refresh"),
        ("f6", "show_fullscreen_monitor", "F6 Monitor"),
        ("f7", "show_discovery", "F7 Discovery"),
        ("space", "toggle_pause", "Space Pause"),
        ("delete", "clear_monitor", "Del Clear"),
        ("s", "cycle_sort", "s Sort"),
        ("f", "focus_filter", "f Filter"),
        ("t", "cycle_theme", "t Theme"),
        ("f4", "show_details", "F4 Details"),
        ("q", "quit", "q Quit"),
    ]

    is_connected: reactive = reactive(False)
    rx_count: reactive = reactive(0)
    fps: reactive = reactive(0.0)
    paused: reactive = reactive(False)

    def __init__(self, startup_args: Optional[StartupArgs] = None) -> None:
        super().__init__()
        self._startup_args: StartupArgs = startup_args or StartupArgs()
        self.can: Optional[WaveshareCAN] = None
        self._fps_timer: Optional[Timer] = None
        self._status_timer: Optional[Timer] = None
        self._monitor_timer: Optional[Timer] = None
        self._stats_timer: Optional[Timer] = None
        self._trace_timer: Optional[Timer] = None
        self._theme_idx: int = 0

        self._store = CANFrameStore()
        self._table_rows: Dict[int, bool] = {}
        self._change_ts: Dict[int, float] = {}
        # Tracks the last timestamp when a CAN-ID's data *actually changed*.
        # Distinct from _change_ts (1 s highlight); used for 10 s stale colouring.
        self._last_actual_change_ts: Dict[int, float] = {}

        # Counters written by RX background thread - protected by lock
        self._counters_lock = threading.Lock()
        self._rx_count_raw: int = 0
        self._fps_raw: int = 0

        self._sort_idx: int = 0
        self._filter_ids: Set[int] = set()
        self._filter_mode: str = "whitelist"

        self._active_cyclics: Set[str] = set()
        self._local_cyclic_threads: Dict[
            str, Tuple[threading.Event, threading.Thread]
        ] = {}
        self._log_lines: List[str] = []

        # Direct reference to DetailsScreen while pushed (None otherwise)
        self._details_screen: Optional[DetailsScreen] = None
        self._fullscreen_monitor: Optional[MonitorFullScreen] = None
        self._discovery_screen: Optional[DiscoveryScreen] = None
        self._stats_updating: bool = False

        # DBC database (shared across screens)
        self._dbc = DBCDatabase()

        # Trace state
        self._trace_buf = TraceBuffer()
        self._trace_elapsed_start: Optional[float] = None

        self._current_speed: CANSpeed = CANSpeed.SPEED_500K
        self._current_frame_type: CANFrameType = CANFrameType.EXTENDED
        self._auto_detect_thread: Optional[threading.Thread] = None

        # Blink state for REC indicator in header (toggled by _fps_timer)
        self._blink_state: bool = False
        # Tracks whether a secondary screen (Details/Fullscreen) is active
        self._details_active: bool = False

    # -----------------------------------------------------------------------
    # Composition & lifecycle
    # -----------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        """Build the main screen layout."""
        yield Header()
        with Container(id="left-col"):
            yield ConnectionPanel()
            yield MonitorPanel()
        with Container(id="right-col"):
            yield StatusPanel()
            yield SendPanel()
            yield FilterPanel()
        yield Footer()

    def on_mount(self) -> None:
        """Start background timers after the DOM is ready."""
        self._log("CAN Bus TUI v1.3.0 started")
        self._log(f"Platform: {platform.system()} {platform.release()}")
        self._log(
            "F1=Help  F2=Connect  F3=Disconnect  F4=Details  "
            "F6=Fullscreen Monitor  F7=Signal Discovery  Space=Pause  q=Quit"
        )

        self._status_timer = self.set_interval(TIMER_STATUS_S, self._update_status_display)
        self._fps_timer = self.set_interval(TIMER_FPS_S, self._calculate_fps)
        self._monitor_timer = self.set_interval(TIMER_MONITOR_S, self._update_monitor_table)
        self._stats_timer = self.set_interval(TIMER_STATS_S, self._update_statistics)
        self._trace_timer = self.set_interval(TIMER_TRACE_S, self._flush_trace)
        self._dbc_timer = self.set_interval(TIMER_DBC_S, self._update_dbc_signals)

        # Give keyboard focus to the monitor table so arrow keys work immediately
        self.call_after_refresh(self._focus_monitor_table)

        # Apply CLI startup arguments after first render
        if (self._startup_args.port or self._startup_args.speed
                or self._startup_args.dbc_path or self._startup_args.auto_connect):
            self.call_after_refresh(self._apply_startup_args)

    def _focus_monitor_table(self) -> None:
        """Move keyboard focus to the monitor DataTable."""
        try:
            self.query_one("#monitor-table", DataTable).focus()
        except Exception:
            pass

    def _apply_startup_args(self) -> None:
        """Apply CLI startup arguments to the UI after first render."""
        args = self._startup_args

        # -- Port --
        if args.port:
            try:
                sel = self.query_one("#port-select", Select)
                # Add the port to the options if it isn't already present
                current_opts = [v for _, v in sel._options]  # type: ignore[attr-defined]
                if args.port not in current_opts:
                    sel.set_options(
                        [(args.port, args.port)]
                        + [(p, p) for p in detect_serial_ports() if p != args.port]
                    )
                sel.value = args.port
                self._log(f"Startup: port set to {args.port}")
            except Exception as exc:
                self._log(f"WARNING: could not set startup port: {exc}")

        # -- Speed --
        if args.speed:
            try:
                self.query_one("#speed-select", Select).value = args.speed
                speed_label = next(
                    (lbl for lbl, v in SPEED_OPTIONS if v == args.speed), str(args.speed)
                )
                self._current_speed = args.speed
                self._log(f"Startup: speed set to {speed_label}")
            except Exception as exc:
                self._log(f"WARNING: could not set startup speed: {exc}")

        # -- DBC --
        if args.dbc_path:
            self._dbc_load(args.dbc_path)

        # -- Auto-connect --
        if args.auto_connect:
            self._log("Startup: auto-connecting …")
            self._do_connect()

    def on_click(self, event) -> None:  # type: ignore[override]
        """Restore keyboard focus to the monitor table after any bare click.

        Clicking on non-focusable areas (labels, panels, empty space) causes
        Textual to drop focus entirely, which breaks arrow-key navigation.
        We re-focus the monitor table unless the click landed on a widget that
        is itself focusable (Input, Button, Select, DataTable, Checkbox).
        """
        focusable_types = (DataTable, Input, Button, Select, Checkbox)
        if not isinstance(event.widget, focusable_types):
            self._focus_monitor_table()

    def on_unmount(self) -> None:
        """Ensure all hardware and threads are cleaned up on exit."""
        if self.is_connected and self.can:
            self.can.close()
        for name in list(self._active_cyclics):
            if name in self._local_cyclic_threads:
                ev, thr = self._local_cyclic_threads.pop(name)
                ev.set()
                thr.join(timeout=THREAD_JOIN_TIMEOUT_S)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def _log(self, message: str) -> None:
        """Append a timestamped line to the internal log and the event-log widget."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self._log_lines.append(line)
        if self._details_screen is not None:
            try:
                self._details_screen.query_one("#event-log", Log).write_line(line)
            except Exception:
                pass  # DetailsScreen not yet fully mounted - intentional

    # -----------------------------------------------------------------------
    # Shortcuts overlay
    # -----------------------------------------------------------------------
    def action_show_shortcuts(self, context: str = "main") -> None:
        """Push the keyboard shortcuts modal with the given screen context."""
        screen = ShortcutsScreen(context=context)
        t = THEMES[THEME_NAMES[self._theme_idx]]

        def _style() -> None:
            screen.apply_theme(t)

        self.push_screen(screen)
        self.call_after_refresh(_style)

    # -----------------------------------------------------------------------
    # Details screen
    # -----------------------------------------------------------------------
    def action_show_details(self) -> None:
        """Push the Details screen, pause main-screen-only timers."""
        self._details_active = True
        if self._monitor_timer:
            self._monitor_timer.pause()
        if self._status_timer:
            self._status_timer.pause()

        ds = DetailsScreen(name="details")
        self._details_screen = ds
        self.push_screen(ds)

        def _populate() -> None:
            try:
                log_w = ds.query_one("#event-log", Log)
                for line in self._log_lines[-LOG_HISTORY_LINES:]:
                    log_w.write_line(line)
                ds.apply_theme(THEMES[THEME_NAMES[self._theme_idx]])
                self._update_statistics()
                self._update_trace_controls()
                # Populate DBC tab if a DB is already loaded
                if self._dbc.loaded:
                    self._populate_dbc_msg_table()
                    self._update_dbc_status_label()
            except Exception as exc:
                self._log(f"WARNING: DetailsScreen populate failed: {exc}")

        self.call_after_refresh(_populate)

    # -----------------------------------------------------------------------
    # Fullscreen Monitor
    # -----------------------------------------------------------------------
    def action_show_fullscreen_monitor(self) -> None:
        """Push the MonitorFullScreen (F6), pause main-screen-only timers."""
        self._details_active = True
        if self._monitor_timer:
            self._monitor_timer.pause()
        if self._status_timer:
            self._status_timer.pause()

        fs = MonitorFullScreen(name="fullscreen_monitor")
        self._fullscreen_monitor = fs
        self.push_screen(fs)

        def _init() -> None:
            try:
                fs._store = self._store
                fs._dbc = self._dbc
                fs._theme_idx = self._theme_idx
                fs.apply_theme(THEMES[THEME_NAMES[self._theme_idx]])
            except Exception as exc:
                self._log(f"WARNING: FullscreenMonitor init failed: {exc}")

        self.call_after_refresh(_init)

    # -----------------------------------------------------------------------
    # Signal Discovery Screen  (F7)
    # -----------------------------------------------------------------------
    def action_show_discovery(self) -> None:
        """Push the Signal Discovery screen (F7), pause main-screen-only timers."""
        self._details_active = True
        if self._monitor_timer:
            self._monitor_timer.pause()
        if self._status_timer:
            self._status_timer.pause()

        ds = DiscoveryScreen(name="discovery")
        self._discovery_screen = ds
        self.push_screen(ds)

        def _init() -> None:
            try:
                ds._store = self._store
                ds._dbc = self._dbc
                ds._theme_idx = self._theme_idx
                ds.apply_theme(THEMES[THEME_NAMES[self._theme_idx]])
            except Exception as exc:
                self._log(f"WARNING: DiscoveryScreen init failed: {exc}")

        self.call_after_refresh(_init)

    # -----------------------------------------------------------------------
    def _dbc_load(self, path: str) -> None:
        """Load a DBC file and refresh the DBC tab UI."""
        ok, err = self._dbc.load(path)
        if ok:
            count = self._dbc.message_count
            self._log(f"DBC loaded: {os.path.basename(path)}  ({count} messages)")
            self._populate_dbc_msg_table()
            self._update_dbc_status_label()
        else:
            self._log(f"ERROR: DBC load failed: {err}")
            self._update_dbc_status_label(error=err)

    def _dbc_unload(self) -> None:
        """Unload the current DBC database."""
        self._dbc.unload()
        self._log("DBC unloaded")
        self._update_dbc_status_label()
        self._clear_dbc_tables()

    def _populate_dbc_msg_table(self) -> None:
        """Fill the DBC messages DataTable from the loaded database."""
        if self._details_screen is None:
            return
        try:
            panel = self._details_screen.query_one(DBCPanel)
            tbl = panel.query_one("#dbc-msg-table", DataTable)
            tbl.clear()
            for msg in self._dbc.get_messages():
                is_ext = msg.can_id > CAN_STD_MAX
                id_str = (
                    f"0x{msg.can_id:08X}" if is_ext else f"0x{msg.can_id:03X}"
                )
                tbl.add_row(
                    id_str,
                    msg.name,
                    str(msg.dlc),
                    str(len(msg.signals)),
                    key=str(msg.can_id),
                )
        except Exception as exc:
            self._log(f"WARNING: DBC msg table populate failed: {exc}")

    def _update_dbc_status_label(self, error: str = "") -> None:
        """Refresh the DBC status label in the Details screen."""
        if self._details_screen is None:
            return
        try:
            lbl = self._details_screen.query_one("#dbc-status", Static)
            t = THEMES[THEME_NAMES[self._theme_idx]]
            if error:
                lbl.update(f"ERROR: {error}")
                lbl.styles.color = t["err"]
            elif self._dbc.loaded:
                lbl.update(
                    f"Loaded: {os.path.basename(self._dbc.path)}"
                    f"  |  {self._dbc.message_count} messages"
                )
                lbl.styles.color = t["ok"]
            else:
                lbl.update("No DBC loaded")
                lbl.styles.color = t["fg"]
        except Exception:
            pass

    def _clear_dbc_tables(self) -> None:
        """Clear the DBC messages DataTable in the Details screen."""
        if self._details_screen is None:
            return
        try:
            panel = self._details_screen.query_one(DBCPanel)
            panel.query_one("#dbc-msg-table", DataTable).clear()
        except Exception:
            pass

    def _update_dbc_signals(self) -> None:
        """Timer callback: update live-data indicators in the DBC messages table (1 Hz)."""
        if self._details_screen is None or not self._dbc.loaded:
            return
        try:
            panel = self._details_screen.query_one(DBCPanel)
            msg_tbl = panel.query_one("#dbc-msg-table", DataTable)
        except Exception:
            return

        rows = self._store.read()
        t = THEMES[THEME_NAMES[self._theme_idx]]
        msg_col_keys = list(msg_tbl.columns)
        if not msg_col_keys:
            return

        # Highlight the Signals column in ok-colour for IDs that have live data
        for can_id, e in rows.items():
            if not self._dbc.lookup_name(can_id):
                continue
            signals = self._dbc.decode(can_id, bytes(e["frame"].data))
            if signals is None:
                continue
            try:
                msg_tbl.update_cell(
                    str(can_id),
                    msg_col_keys[3],
                    Text(str(len(signals)), style=t["ok"]),
                    update_width=False,
                )
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Statistics update
    # -----------------------------------------------------------------------
    def _update_statistics(self) -> None:
        """Guard against re-entrant calls; skip entirely when Details is not visible."""
        if not self._details_active:
            return
        if self._stats_updating:
            return
        self._stats_updating = True
        try:
            self._do_update_statistics()
        finally:
            self._stats_updating = False

    def _do_update_statistics(self) -> None:
        """Recompute per-ID rates and refresh the Statistics DataTable."""
        rows = self._store.read()
        if not rows:
            return

        t = THEMES[THEME_NAMES[self._theme_idx]]
        max_fps = SPEED_MAX_FPS.get(self._current_speed, _DEFAULT_MAX_FPS)

        stats_rows: List[Tuple] = []
        for can_id, e in rows.items():
            elapsed = e["last_ts"] - e["first_ts"]
            rate = (e["count"] - 1) / elapsed if elapsed > 0 and e["count"] > 1 else 0.0
            pct = (rate / max_fps * 100) if max_fps > 0 else 0.0
            stats_rows.append((
                can_id, e["frame"].is_extended,
                len(e["frame"].data), rate, e["count"], pct,
            ))

        stats_rows.sort(key=lambda x: x[3], reverse=True)
        stats_rows = stats_rows[:STATS_TOP_N]

        total_rate = sum(r[3] for r in stats_rows)
        bus_load = min(total_rate / max_fps * 100, 100.0) if max_fps > 0 else 0.0

        ds = self._details_screen
        if ds is not None:
            try:
                table = ds.query_one("#stats-table", DataTable)
                col_keys = list(table.columns)

                existing_keys: Set[str] = {
                    str(row.key.value) for row in table.rows.values()
                }
                desired_keys: Set[str] = {
                    str(rank) for rank in range(1, len(stats_rows) + 1)
                }

                for key in existing_keys - desired_keys:
                    try:
                        table.remove_row(key)
                    except Exception as exc:
                        self._log(f"WARNING: stats remove_row({key}) failed: {exc}")

                for rank, (can_id, is_ext, dlc, rate, count, pct) in enumerate(
                    stats_rows, 1
                ):
                    id_str = f"0x{can_id:08X}" if is_ext else f"0x{can_id:03X}"
                    type_str = "Ext" if is_ext else "Std"
                    pct_color = (
                        t["load_high"] if pct >= 20
                        else t["load_mid"] if pct >= 10
                        else t["load_low"]
                    )
                    pct_text = Text(f"{pct:5.1f}%", style=pct_color)
                    row_key = str(rank)

                    if row_key in existing_keys:
                        table.update_cell(row_key, col_keys[0], str(rank),
                                          update_width=False)
                        table.update_cell(row_key, col_keys[1], id_str,
                                          update_width=False)
                        table.update_cell(row_key, col_keys[2], type_str,
                                          update_width=False)
                        table.update_cell(row_key, col_keys[3], str(dlc),
                                          update_width=False)
                        table.update_cell(row_key, col_keys[4], f"{rate:7.1f}",
                                          update_width=False)
                        table.update_cell(row_key, col_keys[5], str(count),
                                          update_width=False)
                        table.update_cell(row_key, col_keys[6], pct_text,
                                          update_width=False)
                    else:
                        table.add_row(
                            str(rank), id_str, type_str, str(dlc),
                            f"{rate:7.1f}", str(count), pct_text,
                            key=row_key,
                        )

                summary = (
                    f"Total IDs: {len(rows)}   "
                    f"Total Rate: {total_rate:.1f} Hz   "
                    f"Max @ {self._current_speed.name.replace('SPEED_', '')}bps: "
                    f"{max_fps:.0f} Hz   "
                    f"Est. Bus Load: {bus_load:.1f}%"
                )
                ds.query_one("#stats-summary", Static).update(summary)
            except Exception as exc:
                self._log(f"WARNING: statistics update failed: {exc}")

        try:
            self.query_one("#status-unique-ids", Static).update(
                f"Unique IDs: {len(rows)}"
            )
        except Exception:
            pass  # Widget not reachable while Details is active

    # -----------------------------------------------------------------------
    # Bus-load display
    # -----------------------------------------------------------------------
    def _render_bus_load(self, fps: float) -> None:
        """Update the bus-load percentage label and ASCII bar in the status panel."""
        max_fps = SPEED_MAX_FPS.get(self._current_speed, _DEFAULT_MAX_FPS)
        load = min(fps / max_fps * 100, 100.0) if max_fps > 0 else 0.0
        t = THEMES[THEME_NAMES[self._theme_idx]]
        color = (
            t["load_high"] if load >= 75
            else t["load_mid"] if load >= 40
            else t["load_low"]
        )
        try:
            bl = self.query_one("#status-busload", Static)
            bl.update(f"Bus Load: {load:5.1f}%")
            bl.styles.color = color
        except Exception:
            pass
        bar_width = 20
        filled = int(load / 100 * bar_width)
        bar = "[" + "\u2588" * filled + "\u2591" * (bar_width - filled) + "]"
        try:
            bb = self.query_one("#status-busload-bar", Static)
            bb.update(bar)
            bb.styles.color = color
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Port refresh
    # -----------------------------------------------------------------------
    def _refresh_ports(self) -> None:
        """Re-scan serial ports and repopulate the port selector."""
        ports = detect_serial_ports()
        sel = self.query_one("#port-select", Select)
        sel.set_options([(p, p) for p in ports])
        if ports:
            sel.value = ports[0]
        self._log(f"Ports: {', '.join(ports)}")

    # -----------------------------------------------------------------------
    # Trace - state machine helpers
    # -----------------------------------------------------------------------
    def _trace_record(self) -> None:
        """Start a new trace recording session."""
        self._trace_buf.start()
        self._trace_elapsed_start = time.time()
        self._log("Trace RECORDING started")
        self._update_trace_controls()
        self._update_rec_indicator()

    def _trace_pause(self) -> None:
        """Pause the active trace recording."""
        self._trace_buf.pause()
        self._log("Trace PAUSED")
        self._update_trace_controls()
        self._update_rec_indicator()

    def _trace_resume(self) -> None:
        """Resume a paused trace recording."""
        self._trace_buf.resume()
        self._log("Trace RESUMED")
        self._update_trace_controls()
        self._update_rec_indicator()

    def _trace_stop(self) -> None:
        """Stop the trace recorder; data is retained."""
        self._trace_buf.stop()
        self._log(f"Trace STOPPED  ({self._trace_buf.count} frames)")
        self._update_trace_controls()
        self._update_rec_indicator()

    def _trace_clear(self) -> None:
        """Discard all trace data and reset the trace UI."""
        self._trace_buf.clear()
        self._trace_elapsed_start = None
        if self._details_screen is not None:
            try:
                self._details_screen.query_one("#trace-table", DataTable).clear()
                self._details_screen.query_one("#trace-count", Static).update(
                    "Frames: 0"
                )
                self._details_screen.query_one("#trace-elapsed", Static).update(
                    "Elapsed: --"
                )
                self._details_screen.query_one("#trace-warning", Static).update("")
            except Exception as exc:
                self._log(f"WARNING: trace clear UI update failed: {exc}")
        self._log("Trace cleared")
        self._update_trace_controls()
        self._update_rec_indicator()

    def _trace_export(self) -> None:
        """Export all trace records in the format selected in the Trace tab.

        The target format is read from the ``#trace-fmt-select`` Select widget.
        The output file is written to the current working directory with a
        timestamped filename and the appropriate extension.

        Handles the BLF dependency gracefully: if python-can is not installed
        the method logs an informative error instead of raising.
        """
        records = self._trace_buf.snapshot_records()
        if not records:
            self._log("Trace export: no records to export")
            self._set_trace_warning("Nothing to export")
            return

        # --- Determine selected format ---
        fmt = ExportFormat.CSV  # safe default
        if self._details_screen is not None:
            try:
                sel = self._details_screen.query_one(
                    "#trace-fmt-select", Select
                )
                if sel.value is not Select.BLANK:
                    fmt = sel.value  # type: ignore[assignment]
            except Exception:
                pass

        # --- Check optional dependency for BLF ---
        if not fmt.available:
            msg = (
                "BLF export requires python-can.  "
                "Run: pip install python-can"
            )
            self._log(f"ERROR: {msg}")
            self._set_trace_warning(msg)
            return

        # --- Build output filepath ---
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"can_trace_{ts_str}.{fmt.extension}"
        filepath = os.path.join(os.getcwd(), filename)

        # --- Export ---
        ok, message = export_records(
            records,
            fmt,
            filepath,
            channel=1,
            bitrate=500_000,
        )

        if ok:
            self._log(f"Trace exported [{fmt.value}]: {message}")
        else:
            self._log(f"ERROR: {message}")

        self._set_trace_warning(message)

    def _set_trace_warning(self, text: str) -> None:
        """Update the #trace-warning label in the TracePanel (best-effort)."""
        if self._details_screen is None:
            return
        try:
            self._details_screen.query_one(
                "#trace-warning", Static
            ).update(text)
        except Exception:
            pass

    def _update_trace_controls(self) -> None:
        """Sync the Record/Pause/Stop button states and state label in the TracePanel."""
        if self._details_screen is None:
            return
        try:
            tp = self._details_screen.query_one(TracePanel)
            state = self._trace_buf.state

            btn_rec = tp.query_one("#btn-trace-record", Button)
            btn_pause = tp.query_one("#btn-trace-pause", Button)
            btn_stop = tp.query_one("#btn-trace-stop", Button)
            lbl = tp.query_one("#trace-state", Static)

            if state == TraceState.IDLE:
                btn_rec.disabled = False
                btn_pause.disabled = True
                btn_stop.disabled = True
                btn_rec.label = "Record"
                lbl.update("IDLE")
                lbl.remove_class("trace-state-recording", "trace-state-paused")
                lbl.add_class("trace-state-idle")
            elif state == TraceState.RECORDING:
                btn_rec.disabled = True
                btn_pause.disabled = False
                btn_stop.disabled = False
                lbl.update("REC")
                lbl.remove_class("trace-state-idle", "trace-state-paused")
                lbl.add_class("trace-state-recording")
            elif state == TraceState.PAUSED:
                btn_rec.disabled = False
                btn_rec.label = "Resume"
                btn_pause.disabled = True
                btn_stop.disabled = False
                lbl.update("PAUSED")
                lbl.remove_class("trace-state-idle", "trace-state-recording")
                lbl.add_class("trace-state-paused")
        except Exception as exc:
            self._log(f"WARNING: trace controls update failed: {exc}")

    def _trace_record_tx(self, can_id: int, data: bytes, is_extended: bool) -> None:
        """Inject a single Tx frame into the trace buffer."""
        frame = CANFrame(
            can_id=can_id, data=data,
            is_extended=is_extended, timestamp=time.time(),
        )
        self._trace_buf.record(frame, direction="Tx")

    # -----------------------------------------------------------------------
    # Trace UI flush timer
    # -----------------------------------------------------------------------
    def _flush_trace(self) -> None:
        """Drain pending trace records into the DataTable in batches.

        While PAUSED the display is frozen – pending frames accumulate but are
        not written to the table.  On Resume they are flushed in order.
        While IDLE (stopped) the pending queue has already been cleared by stop().
        When the DetailsScreen is not open, pending frames are silently dropped
        only if we are NOT paused (avoids unbounded memory growth when the
        Details screen is closed during a long recording).
        """
        state = self._trace_buf.state

        # Display frozen – keep pending, update counter/elapsed only
        if state == TraceState.PAUSED:
            if self._details_screen is not None:
                try:
                    tp = self._details_screen.query_one(TracePanel)
                    count = self._trace_buf.count
                    tp.query_one("#trace-count", Static).update(
                        f"Frames: {count:,}  (+{self._trace_buf.pending_count} buffered)"
                    )
                except Exception:
                    pass
            return

        if self._details_screen is None:
            # Details screen closed: drain to prevent unbounded growth, but only
            # when actually recording (not paused, handled above).
            self._trace_buf.drain_pending()
            return
        try:
            tp = self._details_screen.query_one(TracePanel)
        except Exception:
            self._trace_buf.drain_pending()
            return

        pending = self._trace_buf.drain_pending()
        if not pending:
            return

        batch = pending[:TRACE_BATCH_SIZE]
        remainder = pending[TRACE_BATCH_SIZE:]
        if remainder:
            self._trace_buf.prepend_pending(remainder)

        t = THEMES[THEME_NAMES[self._theme_idx]]

        try:
            table = tp.query_one("#trace-table", DataTable)
            for rec in batch:
                id_str = (
                    f"0x{rec.can_id:08X}" if rec.is_extended
                    else f"0x{rec.can_id:03X}"
                )
                data_str = rec.data.hex(" ").upper()
                ts_str = f"{rec.rel_ts:10.4f}"
                row_color = t["highlight"] if rec.direction == "Tx" else t["fg"]
                table.add_row(
                    Text(ts_str, style=row_color),
                    Text(id_str, style=row_color),
                    Text(rec.direction, style=row_color),
                    Text(str(rec.dlc), style=row_color),
                    Text(data_str, style=row_color),
                )

            if tp._auto_scroll and batch:
                table.scroll_end(animate=False)

            count = self._trace_buf.count
            tp.query_one("#trace-count", Static).update(f"Frames: {count:,}")

            if (
                self._trace_elapsed_start is not None
                and self._trace_buf.state == TraceState.RECORDING
            ):
                elapsed = time.time() - self._trace_elapsed_start
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = elapsed % 60
                tp.query_one("#trace-elapsed", Static).update(
                    f"Elapsed: {h:02d}:{m:02d}:{s:05.2f}"
                )

            warn_w = tp.query_one("#trace-warning", Static)
            warn_w.update(">100k frames - memory warning" if self._trace_buf.warning else "")

        except Exception as exc:
            self._log(f"WARNING: flush_trace UI update failed: {exc}")

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------
    def _do_connect(self) -> None:
        """Open the serial port, initialise the CAN dongle and start Auto-Detection."""
        if self.is_connected:
            self._log("Already connected")
            return

        port = self.query_one("#port-select", Select).value
        speed = self.query_one("#speed-select", Select).value
        mode = self.query_one("#mode-select", Select).value
        frame_type_sel = self.query_one("#frame-type-select", Select).value

        if port in (Select.BLANK, "(no ports found)"):
            self._log("ERROR: No valid port selected")
            return

        self._log(f"Connecting to {port} ...")
        self.can = WaveshareCAN(port=str(port))
        if not self.can.open():
            self._log(f"ERROR: Failed to open {port}")
            self.can = None
            return

        # Determine effective frame type (None = Auto-Detect)
        manual_frame_type: Optional[CANFrameType] = frame_type_sel  # type: ignore[assignment]

        # Use Extended as starting point for Auto-Detection
        initial_frame_type = (
            manual_frame_type if manual_frame_type is not None
            else CANFrameType.EXTENDED
        )

        speed_name = next((label for label, v in SPEED_OPTIONS if v == speed), str(speed))
        mode_name = next((label for label, v in MODE_OPTIONS if v == mode), str(mode))

        if not self.can.setup(speed=speed, mode=mode, frame_type=initial_frame_type):
            self._log("ERROR: Setup failed")
            self.can.close()
            self.can = None
            return

        self._current_speed = speed
        self._current_frame_type = initial_frame_type
        self.can.on_message_received = self._on_can_frame
        self.can.start_listening()

        self.is_connected = True
        with self._counters_lock:
            self._rx_count_raw = 0
            self._fps_raw = 0
        self.rx_count = 0

        # Clear stale data from any previous session
        self._store.clear()
        self._table_rows.clear()
        self._change_ts.clear()
        self._last_actual_change_ts.clear()
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass

        ft_name = initial_frame_type.name.capitalize()
        self._log(f"Connected: {port} @ {speed_name}, {mode_name}, {ft_name}")
        self._update_connection_ui(True, str(port), speed_name, mode_name)

        # Launch Auto-Detection if user chose "Auto-Detect"
        if manual_frame_type is None:
            self._log("Auto-Detection: listening for frames (Extended)...")
            self._update_frame_type_label("Detecting...")
            self.can.reset_rx_count()
            self._auto_detect_thread = threading.Thread(
                target=self._auto_detect_frame_type,
                args=(speed, mode, speed_name, mode_name),
                daemon=True,
            )
            self._auto_detect_thread.start()

    def _auto_detect_frame_type(
        self,
        speed: CANSpeed,
        mode: CANMode,
        speed_name: str,
        mode_name: str,
    ) -> None:
        """Background thread: Auto-Detect the correct frame type.

        Waits AUTO_DETECT_TIMEOUT_S for frames in Extended mode.
        If none arrive, switches to Standard mode and waits again.
        Posts the result back to the UI thread via call_from_thread.
        """
        # Phase 1: already in Extended mode – just wait
        deadline = time.monotonic() + AUTO_DETECT_TIMEOUT_S
        while time.monotonic() < deadline:
            if not self.is_connected or self.can is None:
                return  # Disconnect happened during detection
            if self.can.rx_frame_count > 0:
                self.call_from_thread(
                    self._auto_detect_done,
                    CANFrameType.EXTENDED, speed_name, mode_name,
                )
                return
            time.sleep(0.1)

        # Phase 1 failed – no frames in Extended mode
        # Phase 2: switch to Standard and try again
        if not self.is_connected or self.can is None:
            return

        self.call_from_thread(
            self._log, "Auto-Detection: no Extended frames – trying Standard..."
        )
        self.call_from_thread(self._update_frame_type_label, "Detecting (Std)...")

        self.can.reset_rx_count()
        if not self.can.setup(speed=speed, mode=mode,
                              frame_type=CANFrameType.STANDARD):
            self.call_from_thread(
                self._log, "Auto-Detection: Standard setup failed"
            )
            return

        self._current_frame_type = CANFrameType.STANDARD

        deadline = time.monotonic() + AUTO_DETECT_TIMEOUT_S
        while time.monotonic() < deadline:
            if not self.is_connected or self.can is None:
                return
            if self.can.rx_frame_count > 0:
                self.call_from_thread(
                    self._auto_detect_done,
                    CANFrameType.STANDARD, speed_name, mode_name,
                )
                return
            time.sleep(0.1)

        # Both phases failed – bus may be silent or wrong baudrate
        self.call_from_thread(self._auto_detect_no_frames)

    def _auto_detect_done(
        self,
        detected: CANFrameType,
        speed_name: str,
        mode_name: str,
    ) -> None:
        """Called on UI thread when Auto-Detection successfully identifies frame type."""
        self._current_frame_type = detected
        ft_label = "Extended (29-bit)" if detected == CANFrameType.EXTENDED else "Standard (11-bit)"
        self._log(f"Auto-detected: {ft_label}")
        self._update_frame_type_label(ft_label)
        # Update the selector to reflect detected type
        try:
            self.query_one("#frame-type-select", Select).value = detected
        except Exception:
            pass

    def _auto_detect_no_frames(self) -> None:
        """Called on UI thread when Auto-Detection finds no frames in either mode."""
        self._log(
            "Auto-Detection: no frames received in either mode. "
            "Check baudrate or bus activity."
        )
        self._update_frame_type_label("No frames")
        self._update_connection_ui(
            True,
            warning="⚠ No frames – check baudrate/bus",
        )

    def _update_frame_type_label(self, label: str) -> None:
        """Update the status panel to show the current/detected frame type."""
        try:
            self.query_one("#status-frame-type", Static).update(f"Frame: {label}")
        except Exception:
            pass

    def _do_disconnect(self) -> None:
        """Stop listening, join cyclic threads and close the serial port."""
        if not self.is_connected:
            self._log("Not connected")
            return

        # Signal auto-detect thread to abort (it checks is_connected)
        self.is_connected = False

        if self._trace_buf.state == TraceState.RECORDING:
            self._trace_stop()
            self._log("Trace auto-stopped on disconnect")

        for name in list(self._active_cyclics):
            if name in self._local_cyclic_threads:
                ev, thr = self._local_cyclic_threads.pop(name)
                ev.set()
                thr.join(timeout=THREAD_JOIN_TIMEOUT_S)
            if self.can:
                try:
                    self.can.stop_cyclic(name)
                except Exception:
                    pass
        self._active_cyclics.clear()

        if self.can:
            self.can.close()
            self.can = None
        # is_connected already set to False above (for auto-detect abort)
        with self._counters_lock:
            self._rx_count_raw = 0
            self._fps_raw = 0
        self.rx_count = 0
        self.fps = 0.0

        self._store.clear()
        self._table_rows.clear()
        self._change_ts.clear()
        self._last_actual_change_ts.clear()
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass

        self._set_send_status("Disconnected", error=True)
        self._log("Disconnected")
        self._update_connection_ui(False)

    # -----------------------------------------------------------------------
    # CAN frame callback - background thread only
    # -----------------------------------------------------------------------
    def _on_can_frame(self, frame: CANFrame) -> None:
        """Receive a frame from the background listener thread.

        Both _store and _trace_buf use internal locks; the counters lock is
        separate and fine-grained to minimise contention.
        """
        self._store.update(frame)
        self._trace_buf.record(frame, direction="Rx")
        with self._counters_lock:
            self._rx_count_raw += 1
            self._fps_raw += 1

    # -----------------------------------------------------------------------
    # Send helpers
    # -----------------------------------------------------------------------
    def _parse_send_inputs(
        self,
    ) -> Tuple[int, bytes, bool, int, str]:
        """Parse and validate the transmit panel inputs.

        Returns:
            Tuple of (can_id, data, is_extended, period_ms, name).

        Raises:
            ValueError: On any invalid or out-of-range input.
        """
        id_str = self.query_one("#send-id", Input).value.strip()
        data_str = self.query_one("#send-data", Input).value.strip()
        period_str = self.query_one("#send-period", Input).value.strip()
        name_str = self.query_one("#send-name", Input).value.strip()
        extended = self.query_one("#send-extended", Checkbox).value

        if not id_str:
            raise ValueError("ID is empty")

        can_id = int(id_str, 0)
        max_id = CAN_EXT_MAX if extended else CAN_STD_MAX
        if can_id < 0 or can_id > max_id:
            limit = "0x1FFFFFFF (Extended)" if extended else "0x7FF (Standard)"
            raise ValueError(f"ID 0x{can_id:X} out of range - max {limit}")

        data = bytes.fromhex(data_str.replace(" ", "")) if data_str else bytes()
        if len(data) > 8:
            raise ValueError("Data exceeds 8 bytes")

        period_ms = int(period_str) if period_str else 0
        name = name_str if name_str else f"task_{can_id:X}"
        return can_id, data, extended, period_ms, name

    def _do_send_single(self) -> None:
        """Send a single CAN frame from the transmit panel inputs."""
        if not self.is_connected or not self.can:
            self._set_send_status("Not connected", error=True)
            return
        try:
            can_id, data, extended, _, _ = self._parse_send_inputs()
        except ValueError as exc:
            self._set_send_status(f"Input error: {exc}", error=True)
            return

        if self.can.send(can_id, data, is_extended=extended):
            id_str = f"0x{can_id:08X}" if extended else f"0x{can_id:03X}"
            self._trace_record_tx(can_id, data, extended)
            self._set_send_status(f"Sent {id_str}  {data.hex(' ').upper()}")
            self._log(f"TX: {id_str}  {data.hex(' ').upper()}")
        else:
            self._set_send_status("Send failed", error=True)

    def _do_start_cyclic(self) -> None:
        """Start a periodic CAN frame transmission task."""
        if not self.is_connected or not self.can:
            self._set_send_status("Not connected", error=True)
            return
        try:
            can_id, data, extended, period_ms, name = self._parse_send_inputs()
        except ValueError as exc:
            self._set_send_status(f"Input error: {exc}", error=True)
            return

        if period_ms <= 0:
            self._set_send_status("Period must be > 0 ms", error=True)
            return
        if name in self._active_cyclics:
            self._set_send_status(f"'{name}' already running", error=True)
            return

        can_ref = self.can
        trace_buf = self._trace_buf

        def _cyclic_trace_sender(stop_event: threading.Event) -> None:
            while not stop_event.is_set():
                if can_ref.send(can_id, data, is_extended=extended):
                    frame = CANFrame(
                        can_id=can_id, data=data,
                        is_extended=extended, timestamp=time.time(),
                    )
                    trace_buf.record(frame, direction="Tx")
                stop_event.wait(period_ms / 1000.0)

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_cyclic_trace_sender, args=(stop_event,), daemon=True
        )
        thread.start()

        self._active_cyclics.add(name)
        self._local_cyclic_threads[name] = (stop_event, thread)

        self.query_one("#btn-cyclic-stop", Button).disabled = False
        id_str = f"0x{can_id:08X}" if extended else f"0x{can_id:03X}"
        self._set_send_status(f"Cyclic '{name}' {id_str} every {period_ms}ms")
        self._log(f"Cyclic start: '{name}' {id_str}  {period_ms}ms")

    def _do_stop_cyclic(self) -> None:
        """Stop a named cyclic task, or all tasks if no name is entered."""
        if not self.is_connected or not self.can:
            self._set_send_status("Not connected", error=True)
            return

        try:
            _, _, _, _, name = self._parse_send_inputs()
        except ValueError:
            name = None

        def _stop_one(n: str) -> None:
            if n in self._local_cyclic_threads:
                ev, thr = self._local_cyclic_threads.pop(n)
                ev.set()
                thr.join(timeout=THREAD_JOIN_TIMEOUT_S)
            try:
                if self.can:
                    self.can.stop_cyclic(n)
            except Exception:
                pass
            self._active_cyclics.discard(n)
            self._log(f"Cyclic stop: '{n}'")

        if name and name in self._active_cyclics:
            _stop_one(name)
            self._set_send_status(f"Cyclic '{name}' stopped")
        elif self._active_cyclics:
            for n in list(self._active_cyclics):
                _stop_one(n)
            self._set_send_status("All cyclic tasks stopped")
            self._log("All cyclic tasks stopped")
        else:
            self._set_send_status("No active cyclic tasks")

        if not self._active_cyclics:
            self.query_one("#btn-cyclic-stop", Button).disabled = True

    def _set_send_status(self, msg: str, error: bool = False) -> None:
        """Update the send status label with an ok or error colour."""
        try:
            t = THEMES[THEME_NAMES[self._theme_idx]]
            w = self.query_one("#send-status", Static)
            w.update(msg)
            w.styles.color = t["err"] if error else t["ok"]
        except Exception:
            pass  # Widget not yet mounted - intentional

    # -----------------------------------------------------------------------
    # Filter
    # -----------------------------------------------------------------------
    def _id_passes_filter(self, can_id: int) -> bool:
        """Return True if can_id should be shown given the current filter settings."""
        if not self._filter_ids:
            return True
        return (
            (can_id in self._filter_ids) if self._filter_mode == "whitelist"
            else (can_id not in self._filter_ids)
        )

    def _apply_filter_from_input(self) -> None:
        """Parse the filter input field and rebuild the monitor table."""
        self._filter_ids = parse_id_list(
            self.query_one("#filter-input", Input).value
        )
        count = len(self._filter_ids)
        if count == 0:
            self._log("Filter cleared")
        else:
            ids_str = ", ".join(f"0x{i:X}" for i in sorted(self._filter_ids))
            self._log(f"Filter ({self._filter_mode}): {ids_str}")
        self._rebuild_table()

    def _toggle_filter_mode(self) -> None:
        """Switch between whitelist and blacklist filter modes."""
        self._filter_mode = (
            "blacklist" if self._filter_mode == "whitelist" else "whitelist"
        )
        self.query_one("#btn-filter-mode", Button).label = (
            self._filter_mode.capitalize()
        )
        self._log(f"Filter mode: {self._filter_mode}")
        self._rebuild_table()

    def _clear_filter(self) -> None:
        """Remove all filter IDs and clear the filter input field."""
        self._filter_ids = set()
        self.query_one("#filter-input", Input).value = ""
        self._log("Filter cleared")
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        """Clear the monitor DataTable so it is fully repopulated on the next tick."""
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass
        self._table_rows.clear()
        self._store._dirty = True

    # -----------------------------------------------------------------------
    # Sort
    # -----------------------------------------------------------------------
    def _cycle_sort(self) -> None:
        """Advance to the next sort mode and rebuild the monitor table."""
        self._sort_idx = (self._sort_idx + 1) % len(SORT_MODES)
        label, _, _ = SORT_MODES[self._sort_idx]
        self.query_one("#btn-sort", Button).label = f"Sort: {label}"
        self._log(f"Sort: {label}")
        self._rebuild_table()

    def _sorted_ids(self, rows: dict) -> List[int]:
        """Return a list of CAN-IDs sorted according to the current sort mode."""
        _, sort_key, reverse = SORT_MODES[self._sort_idx]
        if sort_key == "id":
            return sorted(rows.keys(), reverse=reverse)
        if sort_key == "rate":
            def rate_of(cid: int) -> float:
                e = rows[cid]
                el = e["last_ts"] - e["first_ts"]
                return (e["count"] - 1) / el if el > 0 and e["count"] > 1 else 0.0
            return sorted(rows.keys(), key=rate_of, reverse=reverse)
        return sorted(rows.keys(), key=lambda i: rows[i]["count"], reverse=reverse)

    # -----------------------------------------------------------------------
    # Monitor table update
    # -----------------------------------------------------------------------
    def _build_data_cell(
        self, data: bytes, mask: List[bool], highlight: str, fg: str
    ) -> Text:
        """Build a Rich Text object for the Data column, highlighting changed bytes."""
        text = Text()
        for i, bv in enumerate(data):
            color = highlight if (mask[i] if i < len(mask) else False) else fg
            text.append(f"{bv:02X}", style=color)
            if i < len(data) - 1:
                text.append(" ", style=fg)
        return text

    def _update_monitor_table(self) -> None:
        """Refresh the live monitor DataTable from the frame store (called by timer)."""
        if self.paused:
            return
        dirty, rows = self._store.snapshot()
        if not dirty:
            return
        try:
            mp = self.query_one(MonitorPanel)
            table = self.query_one("#monitor-table", DataTable)
            ck = mp.col_keys
        except Exception:
            return

        t = THEMES[THEME_NAMES[self._theme_idx]]
        fg = t["fg"]
        highlight = t["highlight"]
        stale_color = t["err"]   # re-use the theme error colour for stale frames
        now = time.time()

        for can_id in self._sorted_ids(rows):
            if not self._id_passes_filter(can_id):
                if can_id in self._table_rows:
                    try:
                        table.remove_row(str(can_id))
                    except Exception as exc:
                        self._log(
                            f"WARNING: monitor remove_row({can_id:#x}) failed: {exc}"
                        )
                    del self._table_rows[can_id]
                continue

            e = rows[can_id]
            frame = e["frame"]
            count = e["count"]
            mask = e["changed_mask"]
            elapsed = e["last_ts"] - e["first_ts"]
            rate = (count - 1) / elapsed if elapsed > 0 and count > 1 else 0.0

            id_str = f"0x{can_id:08X}" if frame.is_extended else f"0x{can_id:03X}"
            if self._dbc.loaded:
                msg_name = self._dbc.lookup_name(can_id)
                if msg_name:
                    id_str = f"{id_str} {msg_name}"
            type_str = "Ext" if frame.is_extended else "Std"
            dlc_str = str(len(frame.data))
            rate_str = f"{rate:6.1f}"
            cnt_str = str(count)

            # 1-second highlight for changed bytes
            active_mask = (
                mask if (
                    can_id in self._change_ts
                    and (now - self._change_ts[can_id]) < HIGHLIGHT_FADE_S
                )
                else [False] * len(frame.data)
            )

            # Update stale tracking: only on actual data change
            if any(mask):
                self._change_ts[can_id] = now
                self._last_actual_change_ts[can_id] = now
            elif can_id not in self._last_actual_change_ts:
                # First time we see this ID: start the stale clock
                self._last_actual_change_ts[can_id] = now

            # Determine stale state (10 s without data change → red)
            is_stale = (
                can_id in self._last_actual_change_ts
                and (now - self._last_actual_change_ts[can_id]) > STALE_TIMEOUT_S
            )
            data_fg = stale_color if is_stale else fg

            data_cell = self._build_data_cell(
                frame.data, active_mask,
                highlight, data_fg,
            )
            dlc_text = Text(dlc_str, style=stale_color if is_stale else fg)
            row_key = str(can_id)

            if can_id not in self._table_rows:
                table.add_row(
                    id_str, type_str, dlc_text, rate_str, cnt_str,
                    data_cell, key=row_key,
                )
                self._table_rows[can_id] = True
            else:
                table.update_cell(row_key, ck["ID"], id_str, update_width=False)
                table.update_cell(row_key, ck["Type"], type_str, update_width=False)
                table.update_cell(row_key, ck["DLC"], dlc_text, update_width=False)
                table.update_cell(row_key, ck["Rate Hz"], rate_str, update_width=False)
                table.update_cell(row_key, ck["Count"], cnt_str, update_width=False)
                table.update_cell(row_key, ck["Data"], data_cell, update_width=False)

    # -----------------------------------------------------------------------
    # Status display
    # -----------------------------------------------------------------------
    def _update_status_display(self) -> None:
        """Refresh the status panel widgets (guarded against Details screen active)."""
        try:
            qs = self.can.rx_queue.qsize() if self.can else 0
            self.query_one("#status-rx-count", Static).update(
                f"RX Frames: {self.rx_count}"
            )
            self.query_one("#status-fps", Static).update(f"Frames/s: {self.fps:.1f}")
            self.query_one("#status-queue", Static).update(f"Queue: {qs}")
            self._render_bus_load(self.fps)
        except Exception:
            pass  # Main screen widgets not reachable while Details is active

    def _calculate_fps(self) -> None:
        """Atomically snapshot and reset the FPS counter; update REC blink indicator."""
        with self._counters_lock:
            self.fps = float(self._fps_raw)
            self.rx_count = self._rx_count_raw
            self._fps_raw = 0

        self._update_rec_indicator()

    def _update_rec_indicator(self) -> None:
        """Update the Header sub_title with a blinking REC indicator when recording.

        Called once per second by _fps_timer so no extra timer is needed.
        When not recording the sub_title reverts to the plain version string.
        """
        state = self._trace_buf.state
        if state == TraceState.IDLE:
            self.sub_title = "v1.3.0"
            self._blink_state = False
            return

        self._blink_state = not self._blink_state
        dot = "\u25cf" if self._blink_state else " "  # ● or space

        count = self._trace_buf.count
        count_str = f"{count:,}"

        if state == TraceState.RECORDING and self._trace_elapsed_start is not None:
            elapsed = time.time() - self._trace_elapsed_start
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.sub_title = f"{dot} REC  {h:02d}:{m:02d}:{s:02d}  |  {count_str} frames"
        elif state == TraceState.PAUSED:
            # Steady dot when paused - no blink
            self.sub_title = f"\u25cf PAUSED  |  {count_str} frames"

    # -----------------------------------------------------------------------
    # Connection UI helpers
    # -----------------------------------------------------------------------
    def _update_connection_ui(
        self,
        connected: bool,
        port: str = "-",
        speed: str = "-",
        mode: str = "-",
        warning: Optional[str] = None,
    ) -> None:
        """Update connection panel widgets to reflect the new connection state.

        Args:
            connected: True if connected, False if disconnected.
            port:      Port string for the status panel.
            speed:     Speed label for the status panel.
            mode:      Mode label for the status panel.
            warning:   Optional warning text shown in the connection status widget.
        """
        t = THEMES[THEME_NAMES[self._theme_idx]]
        try:
            sc = self.query_one("#status-connection", Static)
            bc = self.query_one("#btn-connect", Button)
            bdc = self.query_one("#btn-disconnect", Button)
            if connected:
                if warning:
                    sc.update(warning)
                    sc.styles.color = t["paused"]   # orange – connected but degraded
                else:
                    sc.update("Connected")
                    sc.remove_class("status-disconnected")
                    sc.add_class("status-connected")
                    sc.styles.color = t["ok"]
                bc.disabled = True
                bdc.disabled = False
            else:
                sc.update("Disconnected")
                sc.remove_class("status-connected")
                sc.add_class("status-disconnected")
                sc.styles.color = t["err"]
                bc.disabled = False
                bdc.disabled = True
                # Reset frame-type display on disconnect
                try:
                    self.query_one("#status-frame-type", Static).update("Frame: -")
                except Exception:
                    pass
            self.query_one("#status-port", Static).update(f"Port: {port}")
            self.query_one("#status-speed", Static).update(f"Speed: {speed}")
            self.query_one("#status-mode", Static).update(f"Mode: {mode}")
        except Exception:
            pass  # Main screen widgets not reachable while Details is active

    # -----------------------------------------------------------------------
    # Monitor title (paused indicator)
    # -----------------------------------------------------------------------
    def _update_monitor_title(self) -> None:
        """Refresh the monitor panel title to show or hide the PAUSED indicator."""
        t = THEMES[THEME_NAMES[self._theme_idx]]
        title = self.query_one("#monitor-title", Static)
        if self.paused:
            title.update("LIVE MONITOR  [ PAUSED ]")
            title.styles.background = t["paused"]
            title.styles.color = "#000000"
        else:
            title.update("LIVE MONITOR")
            title.styles.background = t["title_bg"]
            title.styles.color = t["title_fg"]

    def watch_paused(self, paused: bool) -> None:
        """Reactive watcher: update the monitor title whenever paused changes."""
        self._update_monitor_title()

    # -----------------------------------------------------------------------
    # Theme
    # -----------------------------------------------------------------------
    def action_cycle_theme(self) -> None:
        """Advance to the next theme and apply it to all screens."""
        self._theme_idx = (self._theme_idx + 1) % len(THEME_NAMES)
        name = THEME_NAMES[self._theme_idx]
        self._apply_theme(THEMES[name])
        self._log(f"Theme: {name}")
        if self._details_screen is not None:
            self._details_screen.apply_theme(THEMES[name])
        if self._fullscreen_monitor is not None:
            self._fullscreen_monitor._theme_idx = self._theme_idx
            self._fullscreen_monitor.apply_theme(THEMES[name])
        if self._discovery_screen is not None:
            self._discovery_screen._theme_idx = self._theme_idx
            self._discovery_screen.apply_theme(THEMES[name])

    def _apply_theme(self, t: Dict[str, str]) -> None:
        """Apply a theme colour dict to all widgets on the main screen."""
        self.screen.styles.background = t["bg"]
        self.screen.styles.color = t["fg"]
        for cls, bk, fk in [
            (Header, "header_bg", "header_fg"),
            (Footer, "footer_bg", "footer_fg"),
        ]:
            try:
                w = self.query_one(cls)
                w.styles.background = t[bk]
                w.styles.color = t[fk]
            except Exception:
                pass
        try:
            self.query_one("#left-col").styles.background = t["bg"]
            self.query_one("#right-col").styles.background = t["bg"]
        except Exception:
            pass
        for panel in (ConnectionPanel, StatusPanel, MonitorPanel, FilterPanel, SendPanel):
            try:
                p = self.query_one(panel)
                p.styles.background = t["bg"]
                p.styles.color = t["fg"]
                p.styles.border = ("solid", t["border"])
            except Exception:
                pass
        for el in self.query(".panel-title"):
            el.styles.background = t["title_bg"]
            el.styles.color = t["title_fg"]
        for el in self.query(".form-label, .send-label, .send-label-sm"):
            el.styles.color = t["accent"]
        for btn in self.query(Button):
            btn.styles.background = t["btn_bg"]
            btn.styles.color = t["btn_fg"]
        sc = self.query_one("#status-connection", Static)
        sc.styles.color = t["ok"] if self.is_connected else t["err"]
        for wid, bk in [
            ("#monitor-table", "table_bg"),
            ("#filter-input", "input_bg"),
            ("#send-id", "input_bg"),
            ("#send-data", "input_bg"),
            ("#send-period", "input_bg"),
            ("#send-name", "input_bg"),
        ]:
            try:
                w = self.query_one(wid)
                w.styles.background = t[bk]
                w.styles.color = t["fg"]
            except Exception:
                pass
        self._update_monitor_title()

    # -----------------------------------------------------------------------
    # Button / key handlers
    # -----------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button press events to the appropriate handler method."""
        bid = event.button.id
        if bid == "btn-connect":
            self._do_connect()
        elif bid == "btn-disconnect":
            self._do_disconnect()
        elif bid == "btn-refresh":
            self._refresh_ports()
        elif bid == "btn-filter-mode":
            self._toggle_filter_mode()
        elif bid == "btn-filter-clear":
            self._clear_filter()
        elif bid == "btn-sort":
            self._cycle_sort()
        elif bid == "btn-send":
            self._do_send_single()
        elif bid == "btn-cyclic-start":
            self._do_start_cyclic()
        elif bid == "btn-cyclic-stop":
            self._do_stop_cyclic()
        # Trace controls
        elif bid == "btn-trace-record":
            if self._trace_buf.state == TraceState.PAUSED:
                self._trace_resume()
            else:
                self._trace_record()
        elif bid == "btn-trace-pause":
            self._trace_pause()
        elif bid == "btn-trace-stop":
            self._trace_stop()
        elif bid == "btn-trace-clear":
            self._trace_clear()
        elif bid == "btn-trace-export":
            self._trace_export()
        # DBC controls
        elif bid == "btn-dbc-load":
            self._handle_dbc_load_button()
        elif bid == "btn-dbc-unload":
            self._dbc_unload()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields."""
        if event.input.id == "filter-input":
            self._apply_filter_from_input()
        elif event.input.id in ("send-id", "send-data", "send-period", "send-name"):
            self._do_send_single()
        elif event.input.id == "dbc-path-input":
            self._handle_dbc_load_button()

    def _handle_dbc_load_button(self) -> None:
        """Read the DBC path input and trigger a load."""
        if self._details_screen is None:
            return
        try:
            path = self._details_screen.query_one(
                "#dbc-path-input", Input
            ).value.strip()
        except Exception:
            return
        if not path:
            self._update_dbc_status_label(error="Please enter a file path")
            return
        if not os.path.isfile(path):
            self._update_dbc_status_label(error=f"File not found: {path}")
            return
        self._dbc_load(path)

    # -----------------------------------------------------------------------
    # Action methods (bound to keys via BINDINGS)
    # -----------------------------------------------------------------------
    def action_connect(self) -> None:
        """Action: F2 - connect to the selected CAN port."""
        self._do_connect()

    def action_disconnect(self) -> None:
        """Action: F3 - disconnect from the CAN port."""
        self._do_disconnect()

    def action_refresh_ports(self) -> None:
        """Action: F5 - refresh the list of available serial ports."""
        self._refresh_ports()

    def action_cycle_sort(self) -> None:
        """Action: s - advance to the next sort mode."""
        self._cycle_sort()

    def action_focus_filter(self) -> None:
        """Action: f - move keyboard focus to the filter input field."""
        try:
            self.query_one("#filter-input", Input).focus()
        except Exception:
            pass

    def action_toggle_pause(self) -> None:
        """Action: Space - toggle monitor pause/resume."""
        self.paused = not self.paused
        self._log("Monitor PAUSED" if self.paused else "Monitor RESUMED")

    def action_clear_monitor(self) -> None:
        """Action: Del - clear the monitor table and frame store."""
        self._store.clear()
        self._table_rows.clear()
        self._change_ts.clear()
        self._last_actual_change_ts.clear()
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass
        self._log("Monitor cleared")
        # Inform discovery screen so it can warn the user
        if self._discovery_screen is not None:
            self._discovery_screen.notify_store_cleared()

    def action_quit(self) -> None:
        """Action: q - disconnect (if connected) and exit the application."""
        if self.is_connected and self.can:
            self.can.close()
        self.exit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = CANBusTUI(startup_args=parse_cli_args())
    app.run()
