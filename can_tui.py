#!/usr/bin/env python3
# 2026-02-19 13:00 v0.7.2 - Fix SPEED_OPTIONS/SPEED_MAX_FPS for corrected CANSpeed enum,
#                           Fix DuplicateKey crash on reconnect (clear table on disconnect),
#                           Fix NoMatches crash when pressing 'v' (Details screen) on Linux
"""
CAN Bus TUI - Textual-based Terminal UI for Waveshare USB-CAN-A
Requires waveshare_can.py v1.0.2 (corrected CANSpeed enum + setup command)
"""

import csv
import os
import platform
import glob
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rich.text import Text

from textual.app import App, ComposeResult
from textual.screen import Screen, ModalScreen
from textual.containers import Horizontal, Container, Vertical
from textual.widgets import (
    Header, Footer, Static, Button, Select, Label,
    Log, DataTable, Input, Checkbox, TabbedContent, TabPane
)
from textual.reactive import reactive
from textual.timer import Timer

from waveshare_can import WaveshareCAN, CANFrame, CANSpeed, CANMode


# ---------------------------------------------------------------------------
# CAN ID range constants
# ---------------------------------------------------------------------------
CAN_STD_MAX = 0x7FF           # 11-bit
CAN_EXT_MAX = 0x1FFFFFFF      # 29-bit


# ---------------------------------------------------------------------------
# Port Detection
# ---------------------------------------------------------------------------
def detect_serial_ports() -> List[str]:
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
# Speed / Mode mappings
# FIX v0.7.1 - Removed SPEED_40K, SPEED_80K, SPEED_666K (not in corrected enum)
# ---------------------------------------------------------------------------
SPEED_OPTIONS = [
    ("5 kbps",   CANSpeed.SPEED_5K),   ("10 kbps",  CANSpeed.SPEED_10K),
    ("20 kbps",  CANSpeed.SPEED_20K),  ("50 kbps",  CANSpeed.SPEED_50K),
    ("100 kbps", CANSpeed.SPEED_100K), ("125 kbps", CANSpeed.SPEED_125K),
    ("200 kbps", CANSpeed.SPEED_200K), ("250 kbps", CANSpeed.SPEED_250K),
    ("400 kbps", CANSpeed.SPEED_400K), ("500 kbps", CANSpeed.SPEED_500K),
    ("800 kbps", CANSpeed.SPEED_800K), ("1 Mbps",   CANSpeed.SPEED_1M),
]

# FIX v0.7.1 - Mode order matches corrected CANMode enum (Silent=0x01, Loopback=0x02)
MODE_OPTIONS = [
    ("Normal",          CANMode.NORMAL),
    ("Silent",          CANMode.SILENT),
    ("Loopback",        CANMode.LOOPBACK),
    ("Loopback-Silent", CANMode.LOOPBACK_SILENT),
]

SORT_MODES = [
    ("ID \u2191",    "id",    False),
    ("Rate \u2193",  "rate",  True),
    ("Count \u2193", "count", True),
]

# CAN bus theoretical max frames/s - ~108 bits per frame (std, 8 data bytes, stuffing)
# FIX v0.7.1 - Removed entries for speeds no longer in SPEED_OPTIONS
SPEED_MAX_FPS: Dict[CANSpeed, float] = {
    CANSpeed.SPEED_5K:   5_000     / 108,
    CANSpeed.SPEED_10K:  10_000    / 108,
    CANSpeed.SPEED_20K:  20_000    / 108,
    CANSpeed.SPEED_50K:  50_000    / 108,
    CANSpeed.SPEED_100K: 100_000   / 108,
    CANSpeed.SPEED_125K: 125_000   / 108,
    CANSpeed.SPEED_200K: 200_000   / 108,
    CANSpeed.SPEED_250K: 250_000   / 108,
    CANSpeed.SPEED_400K: 400_000   / 108,
    CANSpeed.SPEED_500K: 500_000   / 108,
    CANSpeed.SPEED_800K: 800_000   / 108,
    CANSpeed.SPEED_1M:   1_000_000 / 108,
}


# ---------------------------------------------------------------------------
# Thread-safe CAN Frame Store
# ---------------------------------------------------------------------------
class CANFrameStore:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._rows: Dict[int, dict] = {}
        self._dirty = False

    def update(self, frame: CANFrame) -> None:
        with self._lock:
            now    = frame.timestamp
            can_id = frame.can_id
            if can_id not in self._rows:
                self._rows[can_id] = {
                    "frame": frame, "count": 1,
                    "prev_data": frame.data,
                    "changed_mask": [False] * len(frame.data),
                    "first_ts": now, "last_ts": now,
                }
            else:
                e    = self._rows[can_id]
                prev = e["prev_data"]
                new  = frame.data
                ml   = max(len(prev), len(new))
                mask = [
                    (prev[i] if i < len(prev) else -1) != (new[i] if i < len(new) else -1)
                    for i in range(ml)
                ]
                e["frame"]        = frame
                e["count"]       += 1
                e["prev_data"]    = frame.data
                e["changed_mask"] = mask
                e["last_ts"]      = now
            self._dirty = True

    def snapshot(self) -> Tuple[bool, dict]:
        """Consume dirty flag + return copy - used by monitor table."""
        with self._lock:
            dirty        = self._dirty
            self._dirty  = False
            return dirty, dict(self._rows)

    def read(self) -> dict:
        """Return copy WITHOUT touching dirty flag - used by statistics."""
        with self._lock:
            return dict(self._rows)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
            self._dirty = True


# ---------------------------------------------------------------------------
# Trace buffer
# ---------------------------------------------------------------------------
class TraceState(Enum):
    IDLE      = "IDLE"
    RECORDING = "REC"
    PAUSED    = "PAUSED"


@dataclass
class TraceRecord:
    rel_ts:      float    # seconds since recording start
    can_id:      int
    is_extended: bool
    direction:   str      # "Rx" | "Tx"
    dlc:         int
    data:        bytes


class TraceBuffer:
    """Unbounded, thread-safe trace buffer with a pending queue for the UI thread."""

    WARN_THRESHOLD = 100_000

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._records:  List[TraceRecord] = []
        self._pending:  List[TraceRecord] = []
        self._state     = TraceState.IDLE
        self._start_ts: Optional[float] = None

    @property
    def state(self) -> TraceState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: TraceState) -> None:
        with self._lock:
            self._state = value

    def record(self, frame: CANFrame, direction: str = "Rx") -> None:
        with self._lock:
            if self._state != TraceState.RECORDING:
                return
            if self._start_ts is None:
                self._start_ts = frame.timestamp
            rec = TraceRecord(
                rel_ts      = frame.timestamp - self._start_ts,
                can_id      = frame.can_id,
                is_extended = frame.is_extended,
                direction   = direction,
                dlc         = len(frame.data),
                data        = frame.data,
            )
            self._records.append(rec)
            self._pending.append(rec)

    def drain_pending(self) -> List[TraceRecord]:
        with self._lock:
            out = self._pending[:]
            self._pending.clear()
        return out

    def prepend_pending(self, records: List[TraceRecord]) -> None:
        with self._lock:
            self._pending[:0] = records

    def snapshot_records(self) -> List[TraceRecord]:
        with self._lock:
            return list(self._records)

    def start(self, ts: Optional[float] = None) -> None:
        with self._lock:
            if self._state == TraceState.IDLE:
                self._start_ts = ts
            self._state = TraceState.RECORDING

    def pause(self) -> None:
        with self._lock:
            if self._state == TraceState.RECORDING:
                self._state = TraceState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state == TraceState.PAUSED:
                self._state = TraceState.RECORDING

    def stop(self) -> None:
        with self._lock:
            self._state = TraceState.IDLE

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._pending.clear()
            self._start_ts = None
            self._state    = TraceState.IDLE

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def warning(self) -> bool:
        with self._lock:
            return len(self._records) >= self.WARN_THRESHOLD


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------
def parse_id_list(text: str) -> Set[int]:
    ids: Set[int] = set()
    for token in text.replace(",", " ").split():
        try:
            ids.add(int(token, 0))
        except ValueError:
            pass
    return ids


# ---------------------------------------------------------------------------
# Shared theme store
# ---------------------------------------------------------------------------
THEMES = {
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
THEME_NAMES = ["midnight", "norton", "amber", "green", "modern"]


# ---------------------------------------------------------------------------
# Keyboard Shortcuts Modal
# ---------------------------------------------------------------------------
SHORTCUTS = [
    ("q",     "Quit"),
    ("c",     "Connect"),
    ("d",     "Disconnect"),
    ("r",     "Refresh Ports"),
    ("t",     "Cycle Theme"),
    ("x",     "Clear Monitor"),
    ("Space", "Pause / Resume Monitor"),
    ("s",     "Cycle Sort Mode"),
    ("f",     "Focus Filter Input"),
    ("v",     "Open Details Screen"),
    ("?",     "This Help Overlay"),
    ("",      ""),
    ("Details Screen", ""),
    ("v / q", "Back to Main"),
    ("Trace buttons", "Record / Pause / Stop / Clear"),
    ("Export CSV",    "Save trace to can_trace_<ts>.csv"),
]

SHORTCUTS_CSS = """
ShortcutsScreen {
    align: center middle;
}
#shortcuts-dialog {
    width: 56;
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
    """Modal overlay showing all keyboard shortcuts."""

    BINDINGS = [("escape", "dismiss", "Close"), ("?", "dismiss", "Close")]
    CSS = SHORTCUTS_CSS

    def compose(self) -> ComposeResult:
        with Vertical(id="shortcuts-dialog"):
            yield Static("KEYBOARD SHORTCUTS", id="shortcuts-title")
            tbl = DataTable(id="shortcuts-table", show_cursor=False, show_header=False)
            yield tbl
            yield Button("Close  [Esc]", id="shortcuts-close", variant="default")

    def on_mount(self) -> None:
        tbl = self.query_one("#shortcuts-table", DataTable)
        tbl.add_columns("Key", "Action")
        for key, action in SHORTCUTS:
            if key == "" and action == "":
                tbl.add_row("", "")
            elif action == "":
                tbl.add_row(
                    Text(key, style="bold #ffff00"),
                    Text("", style=""),
                )
            else:
                tbl.add_row(
                    Text(key,    style="bold #ffff00"),
                    Text(action, style="#00ffff"),
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "shortcuts-close":
            self.dismiss()

    def apply_theme(self, t: dict) -> None:
        try:
            dlg = self.query_one("#shortcuts-dialog")
            dlg.styles.background = t["modal_bg"]
            dlg.styles.border     = ("double", t["modal_border"])
        except Exception:
            pass
        try:
            self.query_one("#shortcuts-title").styles.background = t["title_bg"]
            self.query_one("#shortcuts-title").styles.color      = t["title_fg"]
        except Exception:
            pass
        try:
            tbl = self.query_one("#shortcuts-table", DataTable)
            tbl.styles.background = t["modal_bg"]
            tbl.styles.color      = t["fg"]
        except Exception:
            pass
        try:
            btn = self.query_one("#shortcuts-close", Button)
            btn.styles.background = t["btn_bg"]
            btn.styles.color      = t["btn_fg"]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reusable panel widgets
# ---------------------------------------------------------------------------
class ConnectionPanel(Container):
    def compose(self) -> ComposeResult:
        yield Static("CONNECTION", classes="panel-title")
        with Horizontal(classes="form-row"):
            yield Label("Port:", classes="form-label")
            ports = detect_serial_ports()
            yield Select([(p, p) for p in ports], id="port-select",
                         value=ports[0] if ports else Select.BLANK, allow_blank=False)
            yield Button("Refresh", id="btn-refresh", variant="default")
        with Horizontal(classes="form-row"):
            yield Label("Speed:", classes="form-label")
            yield Select(SPEED_OPTIONS, id="speed-select",
                         value=CANSpeed.SPEED_500K, allow_blank=False)
        with Horizontal(classes="form-row"):
            yield Label("Mode:", classes="form-label")
            yield Select(MODE_OPTIONS, id="mode-select",
                         value=CANMode.NORMAL, allow_blank=False)
        with Horizontal(classes="button-row"):
            yield Button("Connect",    id="btn-connect",    variant="success")
            yield Button("Disconnect", id="btn-disconnect", variant="error", disabled=True)


class StatusPanel(Container):
    def compose(self) -> ComposeResult:
        yield Static("STATUS", classes="panel-title")
        yield Static("Disconnected", id="status-connection", classes="status-disconnected")
        with Horizontal(classes="status-row"):
            yield Static("Port: -",       id="status-port")
            yield Static("Speed: -",      id="status-speed")
            yield Static("Mode: -",       id="status-mode")
        with Horizontal(classes="status-row"):
            yield Static("RX Frames: 0",  id="status-rx-count")
            yield Static("Frames/s: 0.0", id="status-fps")
            yield Static("Queue: 0",      id="status-queue")
        with Horizontal(classes="status-row"):
            yield Static("Bus Load: --%", id="status-busload")
            yield Static("Unique IDs: 0", id="status-unique-ids")
            yield Static("",              id="status-busload-bar")


class MonitorPanel(Container):
    COL_LABELS = ("ID", "Type", "DLC", "Rate Hz", "Count", "Data")

    def compose(self) -> ComposeResult:
        yield Static("LIVE MONITOR", id="monitor-title", classes="panel-title")
        yield DataTable(id="monitor-table", show_cursor=False)

    def on_mount(self) -> None:
        table = self.query_one("#monitor-table", DataTable)
        col_keys = table.add_columns(*self.COL_LABELS)
        self.col_keys: dict = dict(zip(self.COL_LABELS, col_keys))


class FilterPanel(Container):
    def compose(self) -> ComposeResult:
        yield Static("FILTER & SORT", classes="panel-title")
        with Horizontal(classes="filter-row"):
            yield Label("IDs:", classes="form-label")
            yield Input(placeholder="0x123 0x7FFE  (space/comma, Enter to apply)",
                        id="filter-input")
            yield Button("Whitelist",  id="btn-filter-mode",  variant="default")
            yield Button("Clear",      id="btn-filter-clear", variant="default")
            yield Button("Sort: ID \u2191", id="btn-sort",    variant="default")


class SendPanel(Container):
    """Single-shot and cyclic CAN frame transmit panel."""

    def compose(self) -> ComposeResult:
        yield Static("TRANSMIT", classes="panel-title")
        with Horizontal(classes="send-row"):
            yield Label("ID:",   classes="send-label")
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
            yield Input(placeholder="task1", id="send-name", classes="send-input-name")
        with Horizontal(classes="button-row"):
            yield Button("Send",         id="btn-send",         variant="success")
            yield Button("Start Cyclic", id="btn-cyclic-start", variant="default")
            yield Button("Stop Cyclic",  id="btn-cyclic-stop",  variant="error",
                         disabled=True)
        yield Static("", id="send-status", classes="send-status")


# ---------------------------------------------------------------------------
# Statistics Tab widget
# ---------------------------------------------------------------------------
class StatisticsPanel(Container):
    """Statistics tab: top IDs by rate/count + bus-load summary."""

    COL_LABELS = ("Rank", "ID", "Type", "DLC", "Rate Hz", "Count", "% of Bus")

    def compose(self) -> ComposeResult:
        yield Static("TOP IDs BY RATE", classes="panel-title")
        yield DataTable(id="stats-table", show_cursor=False)
        yield Static("", id="stats-summary", classes="stats-summary")

    def on_mount(self) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.add_columns(*self.COL_LABELS)


# ---------------------------------------------------------------------------
# Trace Tab widget
# ---------------------------------------------------------------------------
class TracePanel(Container):
    """Tab 4 - chronological frame trace with Record/Pause/Stop/Export controls."""

    COL_LABELS = ("Time (s)", "CAN-ID", "Dir", "DLC", "Data")
    BATCH_SIZE = 50   # max rows added per UI tick

    def compose(self) -> ComposeResult:
        yield Static("TRACE", classes="panel-title")
        with Horizontal(classes="trace-ctrl-row"):
            yield Button("\u23fa Record",  id="btn-trace-record", variant="success")
            yield Button("\u23f8 Pause",   id="btn-trace-pause",  variant="default", disabled=True)
            yield Button("\u23f9 Stop",    id="btn-trace-stop",   variant="error",   disabled=True)
            yield Button("\U0001f5d1 Clear", id="btn-trace-clear", variant="default")
            yield Button("\U0001f4be CSV",   id="btn-trace-export", variant="default")
            yield Static("\u25cf IDLE", id="trace-state", classes="trace-state-idle")
        with Horizontal(classes="trace-info-row"):
            yield Static("Frames: 0",         id="trace-count")
            yield Static("Elapsed: --",        id="trace-elapsed")
            yield Static("",                   id="trace-warning",    classes="trace-warning")
            yield Static("\u21a7 AUTO-SCROLL", id="trace-scroll-ind", classes="trace-scroll-on")
        yield DataTable(id="trace-table", show_cursor=True)

    def on_mount(self) -> None:
        table = self.query_one("#trace-table", DataTable)
        table.add_columns(*self.COL_LABELS)
        self._auto_scroll = True

    def on_data_table_scroll(self, event) -> None:  # type: ignore[override]
        table  = self.query_one("#trace-table", DataTable)
        at_end = (table.scroll_y >= table.max_scroll_y - 1)
        if self._auto_scroll and not at_end:
            self._auto_scroll = False
            self._update_scroll_indicator()
        elif not self._auto_scroll and at_end:
            self._auto_scroll = True
            self._update_scroll_indicator()

    def _update_scroll_indicator(self) -> None:
        ind = self.query_one("#trace-scroll-ind", Static)
        if self._auto_scroll:
            ind.update("\u21a7 AUTO-SCROLL")
            ind.remove_class("trace-scroll-off")
            ind.add_class("trace-scroll-on")
        else:
            ind.update("\u23f8 SCROLL LOCKED")
            ind.remove_class("trace-scroll-on")
            ind.add_class("trace-scroll-off")


# ---------------------------------------------------------------------------
# Details Screen CSS
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
    """Secondary screen: TabbedContent with EventLog, Statistics, DBC placeholder, Trace."""

    BINDINGS = [("v", "pop_screen", "Back to Main"), ("q", "pop_screen", "Back")]
    CSS = DETAILS_CSS

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="details-tabs"):
            with TabPane("\U0001f4cb Event Log", id="tab-eventlog"):
                yield Static("EVENT LOG", classes="panel-title")
                yield Log(id="event-log", max_lines=500)
            with TabPane("\U0001f4ca Statistics", id="tab-stats"):
                yield StatisticsPanel()
            with TabPane("\U0001f527 DBC Decoder", id="tab-dbc"):
                yield Static(
                    "DBC Decoder\n\n"
                    "Not yet implemented.\n\n"
                    "Planned features:\n"
                    "  \u2022 Load .dbc file\n"
                    "  \u2022 Decode signals from live frames\n"
                    "  \u2022 Show physical values with units\n"
                    "  \u2022 Filter by message name",
                    id="dbc-placeholder",
                    classes="dbc-placeholder",
                )
            with TabPane("\U0001f4e1 Trace", id="tab-trace"):
                yield TracePanel()
        yield Footer()

    def apply_theme(self, t: dict) -> None:
        self.screen.styles.background = t["bg"]
        try:
            self.query_one(Header).styles.background = t["header_bg"]
            self.query_one(Header).styles.color      = t["header_fg"]
            self.query_one(Footer).styles.background = t["footer_bg"]
            self.query_one(Footer).styles.color      = t["footer_fg"]
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
            el.styles.color      = t["title_fg"]
        try:
            log = self.query_one("#event-log", Log)
            log.styles.background = t["log_bg"]
            log.styles.color      = t["fg"]
        except Exception:
            pass
        try:
            st = self.query_one("#stats-table", DataTable)
            st.styles.background = t["table_bg"]
            st.styles.color      = t["fg"]
        except Exception:
            pass
        for wid_id in ("#stats-summary", "#dbc-placeholder"):
            try:
                w = self.query_one(wid_id)
                w.styles.color      = t["fg"]
                w.styles.background = t["bg"]
            except Exception:
                pass
        try:
            self.query_one(TracePanel).styles.background = t["bg"]
        except Exception:
            pass
        try:
            tt = self.query_one("#trace-table", DataTable)
            tt.styles.background = t["table_bg"]
            tt.styles.color      = t["fg"]
        except Exception:
            pass
        for wid_id in ("#trace-count", "#trace-elapsed"):
            try:
                self.query_one(wid_id).styles.color = t["fg"]
            except Exception:
                pass

    def action_pop_screen(self) -> None:
        self.app._details_screen = None  # type: ignore[attr-defined]
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Main Screen CSS
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

.button-row          { height: auto; margin-top: 1; align-horizontal: center; }
.button-row Button   { margin: 0 1; background: #004488; color: #ffffff; }

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

#btn-refresh       { width: auto; min-width: 10; background: #004488; color: #ffffff; }
#filter-input      { width: 1fr; background: #000060; color: #00ffff; }
#btn-filter-mode   { width: auto; min-width: 12; background: #004488; color: #ffffff; }
#btn-filter-clear  { width: auto; min-width: 8;  background: #004488; color: #ffffff; }
#btn-sort          { width: auto; min-width: 12; background: #004488; color: #ffffff; }

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
# Main Application
# ---------------------------------------------------------------------------
class CANBusTUI(App):
    """CAN Bus TUI v0.7.1 - Main + Details screen with TabbedContent."""

    TITLE     = "Waveshare CAN Bus Tool"
    SUB_TITLE = "v0.7.2"
    CSS       = MAIN_CSS

    BINDINGS = [
        ("q",     "quit",           "Quit"),
        ("c",     "connect",        "Connect"),
        ("d",     "disconnect",     "Disconnect"),
        ("r",     "refresh_ports",  "Refresh Ports"),
        ("t",     "cycle_theme",    "Theme"),
        ("x",     "clear_monitor",  "Clear"),
        ("space", "toggle_pause",   "Pause"),
        ("s",     "cycle_sort",     "Sort"),
        ("f",     "focus_filter",   "Filter"),
        ("v",     "show_details",   "Details"),
        ("?",     "show_shortcuts", "Help"),
    ]

    is_connected = reactive(False)
    rx_count     = reactive(0)
    fps          = reactive(0.0)
    paused       = reactive(False)

    HIGHLIGHT_FADE_S = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.can: Optional[WaveshareCAN] = None
        self._fps_timer:     Optional[Timer] = None
        self._status_timer:  Optional[Timer] = None
        self._monitor_timer: Optional[Timer] = None
        self._stats_timer:   Optional[Timer] = None
        self._theme_idx = 0

        self._store       = CANFrameStore()
        self._table_rows: Dict[int, bool] = {}
        self._change_ts:  Dict[int, float] = {}

        # Counters written by RX background thread - protected by lock
        self._counters_lock = threading.Lock()
        self._rx_count_raw  = 0
        self._fps_raw       = 0

        self._sort_idx    = 0
        self._filter_ids: Set[int] = set()
        self._filter_mode = "whitelist"

        self._active_cyclics: Set[str] = set()
        self._local_cyclic_threads: Dict[str, Tuple[threading.Event, threading.Thread]] = {}
        self._log_lines: List[str] = []

        # Direct reference to DetailsScreen while pushed (None otherwise)
        self._details_screen: Optional[DetailsScreen] = None
        self._stats_updating = False

        # Trace
        self._trace_buf = TraceBuffer()
        self._trace_timer: Optional[Timer] = None
        self._trace_elapsed_start: Optional[float] = None

        self._current_speed: CANSpeed = CANSpeed.SPEED_500K

    # -----------------------------------------------------------------------
    # Composition
    # -----------------------------------------------------------------------
    def compose(self) -> ComposeResult:
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
        self._log("CAN Bus TUI v0.7.2 started")
        self._log(f"Platform: {platform.system()} {platform.release()}")
        self._log("v=Details  Space=Pause  s=Sort  f=Filter  t=Theme  ?=Help")

        self._status_timer  = self.set_interval(0.5, self._update_status_display)
        self._fps_timer     = self.set_interval(1.0, self._calculate_fps)
        self._monitor_timer = self.set_interval(0.2, self._update_monitor_table)
        self._stats_timer   = self.set_interval(2.0, self._update_statistics)
        self._trace_timer   = self.set_interval(0.2, self._flush_trace)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def _log(self, message: str) -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
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
    def action_show_shortcuts(self) -> None:
        screen = ShortcutsScreen()
        t = THEMES[THEME_NAMES[self._theme_idx]]

        def _style():
            screen.apply_theme(t)

        self.push_screen(screen)
        self.call_after_refresh(_style)

    # -----------------------------------------------------------------------
    # Details screen
    # -----------------------------------------------------------------------
    def action_show_details(self) -> None:
        ds = DetailsScreen(name="details")
        self._details_screen = ds
        self.push_screen(ds)

        def _populate():
            try:
                log_w = ds.query_one("#event-log", Log)
                for line in self._log_lines[-200:]:
                    log_w.write_line(line)
                ds.apply_theme(THEMES[THEME_NAMES[self._theme_idx]])
                self._update_statistics()
                self._update_trace_controls()
            except Exception as exc:
                self._log(f"WARNING: DetailsScreen populate failed: {exc}")

        self.call_after_refresh(_populate)

    # -----------------------------------------------------------------------
    # Statistics update (every 2 s + on Details open)
    # -----------------------------------------------------------------------
    def _update_statistics(self) -> None:
        if self._stats_updating:
            return
        self._stats_updating = True
        try:
            self._do_update_statistics()
        finally:
            self._stats_updating = False

    def _do_update_statistics(self) -> None:
        rows    = self._store.read()
        if not rows:
            return

        t       = THEMES[THEME_NAMES[self._theme_idx]]
        max_fps = SPEED_MAX_FPS.get(self._current_speed, 4629.6)

        stats_rows = []
        for can_id, e in rows.items():
            elapsed = e["last_ts"] - e["first_ts"]
            rate    = (e["count"] - 1) / elapsed if elapsed > 0 and e["count"] > 1 else 0.0
            pct     = (rate / max_fps * 100) if max_fps > 0 else 0.0
            stats_rows.append((can_id, e["frame"].is_extended, len(e["frame"].data),
                                rate, e["count"], pct))

        stats_rows.sort(key=lambda x: x[3], reverse=True)
        stats_rows = stats_rows[:30]

        total_rate = sum(r[3] for r in stats_rows)
        bus_load   = min(total_rate / max_fps * 100, 100.0) if max_fps > 0 else 0.0

        ds = self._details_screen
        if ds is not None:
            try:
                table = ds.query_one("#stats-table", DataTable)
                col_keys = list(table.columns)

                existing_keys: Set[str] = {
                    str(row.key.value) for row in table.rows.values()
                }
                desired_keys: Set[str] = {str(rank) for rank in range(1, len(stats_rows) + 1)}

                for key in existing_keys - desired_keys:
                    try:
                        table.remove_row(key)
                    except Exception as exc:
                        self._log(f"WARNING: stats remove_row({key}) failed: {exc}")

                for rank, (can_id, is_ext, dlc, rate, count, pct) in enumerate(stats_rows, 1):
                    id_str    = f"0x{can_id:08X}" if is_ext else f"0x{can_id:03X}"
                    type_str  = "Ext" if is_ext else "Std"
                    pct_color = (t["load_high"] if pct >= 20
                                 else t["load_mid"] if pct >= 10
                                 else t["load_low"])
                    pct_text  = Text(f"{pct:5.1f}%", style=pct_color)
                    row_key   = str(rank)

                    if row_key in existing_keys:
                        table.update_cell(row_key, col_keys[0], str(rank),      update_width=False)
                        table.update_cell(row_key, col_keys[1], id_str,         update_width=False)
                        table.update_cell(row_key, col_keys[2], type_str,       update_width=False)
                        table.update_cell(row_key, col_keys[3], str(dlc),       update_width=False)
                        table.update_cell(row_key, col_keys[4], f"{rate:7.1f}", update_width=False)
                        table.update_cell(row_key, col_keys[5], str(count),     update_width=False)
                        table.update_cell(row_key, col_keys[6], pct_text,       update_width=False)
                    else:
                        table.add_row(str(rank), id_str, type_str, str(dlc),
                                      f"{rate:7.1f}", str(count), pct_text,
                                      key=row_key)

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
            self.query_one("#status-unique-ids", Static).update(f"Unique IDs: {len(rows)}")
        except Exception:
            pass  # Widget not yet mounted - intentional

    # -----------------------------------------------------------------------
    # Bus-load display
    # -----------------------------------------------------------------------
    def _render_bus_load(self, fps: float) -> None:
        max_fps = SPEED_MAX_FPS.get(self._current_speed, 4629.6)
        load    = min(fps / max_fps * 100, 100.0) if max_fps > 0 else 0.0
        t       = THEMES[THEME_NAMES[self._theme_idx]]
        color   = (t["load_high"] if load >= 75
                   else t["load_mid"] if load >= 40
                   else t["load_low"])
        try:
            bl = self.query_one("#status-busload", Static)
            bl.update(f"Bus Load: {load:5.1f}%")
            bl.styles.color = color
        except Exception:
            pass
        bar_width = 20
        filled    = int(load / 100 * bar_width)
        bar       = "[" + "\u2588" * filled + "\u2591" * (bar_width - filled) + "]"
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
        ports = detect_serial_ports()
        sel   = self.query_one("#port-select", Select)
        sel.set_options([(p, p) for p in ports])
        if ports:
            sel.value = ports[0]
        self._log(f"Ports: {', '.join(ports)}")

    # -----------------------------------------------------------------------
    # Trace - state machine helpers
    # -----------------------------------------------------------------------
    def _trace_record(self) -> None:
        self._trace_buf.start()
        self._trace_elapsed_start = time.time()
        self._log("Trace RECORDING started")
        self._update_trace_controls()

    def _trace_pause(self) -> None:
        self._trace_buf.pause()
        self._log("Trace PAUSED")
        self._update_trace_controls()

    def _trace_resume(self) -> None:
        self._trace_buf.resume()
        self._log("Trace RESUMED")
        self._update_trace_controls()

    def _trace_stop(self) -> None:
        self._trace_buf.stop()
        self._log(f"Trace STOPPED  ({self._trace_buf.count} frames)")
        self._update_trace_controls()

    def _trace_clear(self) -> None:
        self._trace_buf.clear()
        self._trace_elapsed_start = None
        if self._details_screen is not None:
            try:
                self._details_screen.query_one("#trace-table", DataTable).clear()
                self._details_screen.query_one("#trace-count",   Static).update("Frames: 0")
                self._details_screen.query_one("#trace-elapsed", Static).update("Elapsed: --")
                self._details_screen.query_one("#trace-warning", Static).update("")
            except Exception as exc:
                self._log(f"WARNING: trace clear UI update failed: {exc}")
        self._log("Trace cleared")
        self._update_trace_controls()

    def _trace_export_csv(self) -> None:
        """Export all trace records to a CSV file in the current working directory."""
        records = self._trace_buf.snapshot_records()
        if not records:
            self._log("Trace export: no records to export")
            if self._details_screen is not None:
                try:
                    self._details_screen.query_one("#trace-warning", Static).update(
                        "\u26a0 Nothing to export"
                    )
                except Exception:
                    pass
            return

        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"can_trace_{ts_str}.csv"
        filepath = os.path.join(os.getcwd(), filename)

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["Time_s", "CAN_ID_hex", "Type", "Dir", "DLC", "Data_hex"])
                for rec in records:
                    id_str   = (f"0x{rec.can_id:08X}" if rec.is_extended
                                else f"0x{rec.can_id:03X}")
                    type_str = "Ext" if rec.is_extended else "Std"
                    writer.writerow([
                        f"{rec.rel_ts:.6f}",
                        id_str,
                        type_str,
                        rec.direction,
                        rec.dlc,
                        rec.data.hex(" ").upper(),
                    ])
            self._log(f"Trace exported: {filename}  ({len(records)} frames)")
            if self._details_screen is not None:
                try:
                    self._details_screen.query_one("#trace-warning", Static).update(
                        f"\u2713 Saved: {filename}"
                    )
                except Exception:
                    pass
        except OSError as exc:
            self._log(f"ERROR: Trace export failed: {exc}")
            if self._details_screen is not None:
                try:
                    self._details_screen.query_one("#trace-warning", Static).update(
                        f"\u26a0 Export failed: {exc}"
                    )
                except Exception:
                    pass

    def _update_trace_controls(self) -> None:
        if self._details_screen is None:
            return
        try:
            tp    = self._details_screen.query_one(TracePanel)
            state = self._trace_buf.state

            btn_rec   = tp.query_one("#btn-trace-record", Button)
            btn_pause = tp.query_one("#btn-trace-pause",  Button)
            btn_stop  = tp.query_one("#btn-trace-stop",   Button)
            lbl       = tp.query_one("#trace-state",      Static)

            if state == TraceState.IDLE:
                btn_rec.disabled   = False
                btn_pause.disabled = True
                btn_stop.disabled  = True
                btn_rec.label      = "\u23fa Record"
                lbl.update("\u25cf IDLE")
                lbl.remove_class("trace-state-recording", "trace-state-paused")
                lbl.add_class("trace-state-idle")
            elif state == TraceState.RECORDING:
                btn_rec.disabled   = True
                btn_pause.disabled = False
                btn_stop.disabled  = False
                lbl.update("\u25cf REC")
                lbl.remove_class("trace-state-idle", "trace-state-paused")
                lbl.add_class("trace-state-recording")
            elif state == TraceState.PAUSED:
                btn_rec.disabled   = False
                btn_rec.label      = "\u25b6 Resume"
                btn_pause.disabled = True
                btn_stop.disabled  = False
                lbl.update("\u23f8 PAUSED")
                lbl.remove_class("trace-state-idle", "trace-state-recording")
                lbl.add_class("trace-state-paused")
        except Exception as exc:
            self._log(f"WARNING: trace controls update failed: {exc}")

    def _trace_record_tx(self, can_id: int, data: bytes, is_extended: bool) -> None:
        """Log a single-shot Tx frame into the trace buffer."""
        frame = CANFrame(
            can_id=can_id, data=data,
            is_extended=is_extended, timestamp=time.time()
        )
        self._trace_buf.record(frame, direction="Tx")

    # -----------------------------------------------------------------------
    # Trace - UI flush timer (every 0.2 s, UI thread)
    # -----------------------------------------------------------------------
    def _flush_trace(self) -> None:
        """Drain pending trace records into the DataTable in batches.

        Frames exceeding BATCH_SIZE per tick are put back via prepend_pending()
        so no frames are ever lost.
        """
        if self._details_screen is None:
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

        batch     = pending[:TracePanel.BATCH_SIZE]
        remainder = pending[TracePanel.BATCH_SIZE:]
        if remainder:
            self._trace_buf.prepend_pending(remainder)

        t = THEMES[THEME_NAMES[self._theme_idx]]

        try:
            table = tp.query_one("#trace-table", DataTable)

            for rec in batch:
                id_str   = (f"0x{rec.can_id:08X}" if rec.is_extended
                            else f"0x{rec.can_id:03X}")
                data_str = rec.data.hex(" ").upper()
                ts_str   = f"{rec.rel_ts:10.4f}"
                row_color = t["highlight"] if rec.direction == "Tx" else t["fg"]
                table.add_row(
                    Text(ts_str,        style=row_color),
                    Text(id_str,        style=row_color),
                    Text(rec.direction, style=row_color),
                    Text(str(rec.dlc),  style=row_color),
                    Text(data_str,      style=row_color),
                )

            if tp._auto_scroll and batch:
                table.scroll_end(animate=False)

            count = self._trace_buf.count
            tp.query_one("#trace-count", Static).update(f"Frames: {count:,}")

            if self._trace_elapsed_start and self._trace_buf.state == TraceState.RECORDING:
                elapsed = time.time() - self._trace_elapsed_start
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = elapsed % 60
                tp.query_one("#trace-elapsed", Static).update(
                    f"Elapsed: {h:02d}:{m:02d}:{s:05.2f}"
                )

            warn_w = tp.query_one("#trace-warning", Static)
            warn_w.update("\u26a0  >100k frames" if self._trace_buf.warning else "")

        except Exception as exc:
            self._log(f"WARNING: flush_trace UI update failed: {exc}")

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------
    def _do_connect(self) -> None:
        if self.is_connected:
            self._log("Already connected")
            return

        port  = self.query_one("#port-select",  Select).value
        speed = self.query_one("#speed-select", Select).value
        mode  = self.query_one("#mode-select",  Select).value

        if port in (Select.BLANK, "(no ports found)"):
            self._log("ERROR: No valid port selected")
            return

        self._log(f"Connecting to {port} ...")
        self.can = WaveshareCAN(port=str(port))
        if not self.can.open():
            self._log(f"ERROR: Failed to open {port}")
            self.can = None
            return

        speed_name = next((l for l, v in SPEED_OPTIONS if v == speed), str(speed))
        mode_name  = next((l for l, v in MODE_OPTIONS  if v == mode),  str(mode))

        if not self.can.setup(speed=speed, mode=mode):
            self._log("ERROR: Setup failed")
            self.can.close()
            self.can = None
            return

        self._current_speed = speed
        self.can.on_message_received = self._on_can_frame
        self.can.start_listening()

        self.is_connected = True
        with self._counters_lock:
            self._rx_count_raw = 0
            self._fps_raw      = 0
        self.rx_count = 0

        # Belt-and-suspenders clear (primary clear happens in _do_disconnect)
        self._store.clear()
        self._table_rows.clear()
        self._change_ts.clear()
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass

        self._log(f"Connected: {port} @ {speed_name}, {mode_name}")
        self._update_connection_ui(True, str(port), speed_name, mode_name)

    def _do_disconnect(self) -> None:
        if not self.is_connected:
            self._log("Not connected")
            return

        # Stop trace on disconnect to avoid timestamp gap
        if self._trace_buf.state == TraceState.RECORDING:
            self._trace_stop()
            self._log("Trace auto-stopped on disconnect")

        # Stop all cyclic tasks
        for name in list(self._active_cyclics):
            if name in self._local_cyclic_threads:
                ev, thr = self._local_cyclic_threads.pop(name)
                ev.set()
                thr.join(timeout=1.0)
            if self.can:
                try:
                    self.can.stop_cyclic(name)
                except Exception:
                    pass
        self._active_cyclics.clear()

        if self.can:
            self.can.close()
            self.can = None
        self.is_connected = False
        with self._counters_lock:
            self._rx_count_raw = 0
            self._fps_raw      = 0
        self.rx_count = 0
        self.fps      = 0.0

        # FIX v0.7.1 - Clear monitor table on disconnect so the next Connect
        # starts with a clean DataTable, preventing DuplicateKey errors.
        self._store.clear()
        self._table_rows.clear()
        self._change_ts.clear()
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass

        self._set_send_status("Disconnected", error=True)
        self._log("Disconnected")
        self._update_connection_ui(False)

    # -----------------------------------------------------------------------
    # CAN frame callback - background thread only
    # Both _store.update() and _trace_buf.record() use their own locks
    # internally; the counters lock is separate and fine-grained.
    # -----------------------------------------------------------------------
    def _on_can_frame(self, frame: CANFrame) -> None:
        self._store.update(frame)
        self._trace_buf.record(frame, direction="Rx")
        with self._counters_lock:
            self._rx_count_raw += 1
            self._fps_raw      += 1

    # -----------------------------------------------------------------------
    # Send helpers
    # -----------------------------------------------------------------------
    def _parse_send_inputs(self):
        """Parse and validate send panel inputs.

        Returns: (can_id, data, extended, period_ms, name)
        Raises: ValueError on any invalid input.
        """
        id_str     = self.query_one("#send-id",      Input).value.strip()
        data_str   = self.query_one("#send-data",    Input).value.strip()
        period_str = self.query_one("#send-period",  Input).value.strip()
        name_str   = self.query_one("#send-name",    Input).value.strip()
        extended   = self.query_one("#send-extended", Checkbox).value

        if not id_str:
            raise ValueError("ID is empty")

        can_id = int(id_str, 0)

        max_id = CAN_EXT_MAX if extended else CAN_STD_MAX
        if can_id < 0 or can_id > max_id:
            raise ValueError(
                f"ID 0x{can_id:X} out of range for "
                f"{'Extended (max 0x1FFFFFFF)' if extended else 'Standard (max 0x7FF)'}"
            )

        data = bytes.fromhex(data_str.replace(" ", "")) if data_str else bytes()
        if len(data) > 8:
            raise ValueError("Data exceeds 8 bytes")

        period_ms = int(period_str) if period_str else 0
        name      = name_str if name_str else f"task_{can_id:X}"
        return can_id, data, extended, period_ms, name

    def _do_send_single(self) -> None:
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

        can_ref   = self.can
        trace_buf = self._trace_buf

        def _cyclic_trace_sender(stop_event: threading.Event) -> None:
            """Thin wrapper: send + trace each cyclic frame."""
            while not stop_event.is_set():
                if can_ref.send(can_id, data, is_extended=extended):
                    frame = CANFrame(
                        can_id=can_id, data=data,
                        is_extended=extended, timestamp=time.time()
                    )
                    trace_buf.record(frame, direction="Tx")
                stop_event.wait(period_ms / 1000.0)

        stop_event = threading.Event()
        thread     = threading.Thread(target=_cyclic_trace_sender,
                                      args=(stop_event,), daemon=True)
        thread.start()

        self._active_cyclics.add(name)
        self._local_cyclic_threads[name] = (stop_event, thread)

        self.query_one("#btn-cyclic-stop", Button).disabled = False
        id_str = f"0x{can_id:08X}" if extended else f"0x{can_id:03X}"
        self._set_send_status(f"Cyclic '{name}' {id_str} every {period_ms}ms")
        self._log(f"Cyclic start: '{name}' {id_str}  {period_ms}ms  (trace enabled)")

    def _do_stop_cyclic(self) -> None:
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
                thr.join(timeout=1.0)
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
        if not self._filter_ids:
            return True
        return (can_id in self._filter_ids) if self._filter_mode == "whitelist" \
               else (can_id not in self._filter_ids)

    def _apply_filter_from_input(self) -> None:
        self._filter_ids = parse_id_list(self.query_one("#filter-input", Input).value)
        count = len(self._filter_ids)
        if count == 0:
            self._log("Filter cleared")
        else:
            ids_str = ", ".join(f"0x{i:X}" for i in sorted(self._filter_ids))
            self._log(f"Filter ({self._filter_mode}): {ids_str}")
        self._rebuild_table()

    def _toggle_filter_mode(self) -> None:
        self._filter_mode = "blacklist" if self._filter_mode == "whitelist" else "whitelist"
        self.query_one("#btn-filter-mode", Button).label = self._filter_mode.capitalize()
        self._log(f"Filter mode: {self._filter_mode}")
        self._rebuild_table()

    def _clear_filter(self) -> None:
        self._filter_ids = set()
        self.query_one("#filter-input", Input).value = ""
        self._log("Filter cleared")
        self._rebuild_table()

    def _rebuild_table(self) -> None:
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
        self._sort_idx = (self._sort_idx + 1) % len(SORT_MODES)
        label, _, _    = SORT_MODES[self._sort_idx]
        self.query_one("#btn-sort", Button).label = f"Sort: {label}"
        self._log(f"Sort: {label}")
        self._rebuild_table()

    def _sorted_ids(self, rows: dict) -> List[int]:
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
    def _build_data_cell(self, data: bytes, mask: List[bool],
                          highlight: str, fg: str) -> Text:
        text = Text()
        for i, bv in enumerate(data):
            color = highlight if (mask[i] if i < len(mask) else False) else fg
            text.append(f"{bv:02X}", style=color)
            if i < len(data) - 1:
                text.append(" ", style=fg)
        return text

    def _update_monitor_table(self) -> None:
        if self.paused:
            return
        dirty, rows = self._store.snapshot()
        if not dirty:
            return
        try:
            mp    = self.query_one(MonitorPanel)
            table = self.query_one("#monitor-table", DataTable)
            ck    = mp.col_keys
        except Exception:
            return

        t         = THEMES[THEME_NAMES[self._theme_idx]]
        fg        = t["fg"]
        highlight = t["highlight"]
        now       = time.time()

        for can_id in self._sorted_ids(rows):
            if not self._id_passes_filter(can_id):
                if can_id in self._table_rows:
                    try:
                        table.remove_row(str(can_id))
                    except Exception as exc:
                        self._log(f"WARNING: monitor remove_row({can_id:#x}) failed: {exc}")
                    del self._table_rows[can_id]
                continue

            e        = rows[can_id]
            frame    = e["frame"]
            count    = e["count"]
            mask     = e["changed_mask"]
            elapsed  = e["last_ts"] - e["first_ts"]
            rate     = (count - 1) / elapsed if elapsed > 0 and count > 1 else 0.0

            id_str   = f"0x{can_id:08X}" if frame.is_extended else f"0x{can_id:03X}"
            type_str = "Ext" if frame.is_extended else "Std"
            dlc_str  = str(len(frame.data))
            rate_str = f"{rate:6.1f}"
            cnt_str  = str(count)

            active_mask = (
                mask if (can_id in self._change_ts and
                         (now - self._change_ts[can_id]) < self.HIGHLIGHT_FADE_S)
                else [False] * len(frame.data)
            )
            if any(mask):
                self._change_ts[can_id] = now

            data_cell = self._build_data_cell(frame.data, active_mask, highlight, fg)
            row_key   = str(can_id)

            if can_id not in self._table_rows:
                table.add_row(id_str, type_str, dlc_str, rate_str, cnt_str,
                              data_cell, key=row_key)
                self._table_rows[can_id] = True
            else:
                table.update_cell(row_key, ck["ID"],      id_str,    update_width=False)
                table.update_cell(row_key, ck["Type"],    type_str,  update_width=False)
                table.update_cell(row_key, ck["DLC"],     dlc_str,   update_width=False)
                table.update_cell(row_key, ck["Rate Hz"], rate_str,  update_width=False)
                table.update_cell(row_key, ck["Count"],   cnt_str,   update_width=False)
                table.update_cell(row_key, ck["Data"],    data_cell, update_width=False)

    # -----------------------------------------------------------------------
    # Status display
    # -----------------------------------------------------------------------
    def _update_status_display(self) -> None:
        # FIX v0.7.1 - Wrapped in try/except: these widgets live on the main
        # screen only. When DetailsScreen is pushed (e.g. pressing 'v'), the
        # timer still fires but query_one raises NoMatches on the wrong screen,
        # crashing the app. Guard against that here.
        try:
            qs = self.can.rx_queue.qsize() if self.can else 0
            self.query_one("#status-rx-count", Static).update(f"RX Frames: {self.rx_count}")
            self.query_one("#status-fps",      Static).update(f"Frames/s: {self.fps:.1f}")
            self.query_one("#status-queue",    Static).update(f"Queue: {qs}")
            self._render_bus_load(self.fps)
        except Exception:
            pass  # Main screen widgets not reachable while Details is active

    def _calculate_fps(self) -> None:
        """UI thread - atomically snapshot and reset the fps counter."""
        with self._counters_lock:
            self.fps          = float(self._fps_raw)
            self.rx_count     = self._rx_count_raw
            self._fps_raw     = 0

    # -----------------------------------------------------------------------
    # Connection UI
    # -----------------------------------------------------------------------
    def _update_connection_ui(self, connected: bool,
                               port: str = "-", speed: str = "-",
                               mode: str = "-") -> None:
        # FIX v0.7.1 - Wrapped in try/except: same NoMatches risk as
        # _update_status_display when DetailsScreen is the active screen.
        t = THEMES[THEME_NAMES[self._theme_idx]]
        try:
            sc  = self.query_one("#status-connection", Static)
            bc  = self.query_one("#btn-connect",       Button)
            bdc = self.query_one("#btn-disconnect",    Button)
            if connected:
                sc.update("Connected")
                sc.remove_class("status-disconnected")
                sc.add_class("status-connected")
                sc.styles.color = t["ok"]
                bc.disabled     = True
                bdc.disabled    = False
            else:
                sc.update("Disconnected")
                sc.remove_class("status-connected")
                sc.add_class("status-disconnected")
                sc.styles.color = t["err"]
                bc.disabled     = False
                bdc.disabled    = True
            self.query_one("#status-port",  Static).update(f"Port: {port}")
            self.query_one("#status-speed", Static).update(f"Speed: {speed}")
            self.query_one("#status-mode",  Static).update(f"Mode: {mode}")
        except Exception:
            pass  # Main screen widgets not reachable while Details is active

    # -----------------------------------------------------------------------
    # Monitor title (paused indicator)
    # -----------------------------------------------------------------------
    def _update_monitor_title(self) -> None:
        t     = THEMES[THEME_NAMES[self._theme_idx]]
        title = self.query_one("#monitor-title", Static)
        if self.paused:
            title.update("LIVE MONITOR  [ PAUSED ]")
            title.styles.background = t["paused"]
            title.styles.color      = "#000000"
        else:
            title.update("LIVE MONITOR")
            title.styles.background = t["title_bg"]
            title.styles.color      = t["title_fg"]

    def watch_paused(self, paused: bool) -> None:
        self._update_monitor_title()

    # -----------------------------------------------------------------------
    # Theme
    # -----------------------------------------------------------------------
    def action_cycle_theme(self) -> None:
        self._theme_idx = (self._theme_idx + 1) % len(THEME_NAMES)
        name = THEME_NAMES[self._theme_idx]
        self._apply_theme(THEMES[name])
        self._log(f"Theme: {name}")
        if self._details_screen is not None:
            self._details_screen.apply_theme(THEMES[name])

    def _apply_theme(self, t: dict) -> None:
        self.screen.styles.background = t["bg"]
        self.screen.styles.color      = t["fg"]
        for cls, bk, fk in [(Header, "header_bg", "header_fg"),
                             (Footer, "footer_bg", "footer_fg")]:
            try:
                w = self.query_one(cls)
                w.styles.background = t[bk]
                w.styles.color      = t[fk]
            except Exception:
                pass
        try:
            self.query_one("#left-col").styles.background  = t["bg"]
            self.query_one("#right-col").styles.background = t["bg"]
        except Exception:
            pass
        for panel in (ConnectionPanel, StatusPanel, MonitorPanel, FilterPanel, SendPanel):
            try:
                p = self.query_one(panel)
                p.styles.background = t["bg"]
                p.styles.color      = t["fg"]
                p.styles.border     = ("solid", t["border"])
            except Exception:
                pass
        for el in self.query(".panel-title"):
            el.styles.background = t["title_bg"]
            el.styles.color      = t["title_fg"]
        for el in self.query(".form-label, .send-label, .send-label-sm"):
            el.styles.color = t["accent"]
        for btn in self.query(Button):
            btn.styles.background = t["btn_bg"]
            btn.styles.color      = t["btn_fg"]
        sc = self.query_one("#status-connection", Static)
        sc.styles.color = t["ok"] if self.is_connected else t["err"]
        for wid, bk in [("#monitor-table", "table_bg"),
                        ("#filter-input",  "input_bg"),
                        ("#send-id",       "input_bg"),
                        ("#send-data",     "input_bg"),
                        ("#send-period",   "input_bg"),
                        ("#send-name",     "input_bg")]:
            try:
                w = self.query_one(wid)
                w.styles.background = t[bk]
                w.styles.color      = t["fg"]
            except Exception:
                pass
        self._update_monitor_title()

    # -----------------------------------------------------------------------
    # Button / key handlers
    # -----------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if   bid == "btn-connect":       self._do_connect()
        elif bid == "btn-disconnect":    self._do_disconnect()
        elif bid == "btn-refresh":       self._refresh_ports()
        elif bid == "btn-filter-mode":   self._toggle_filter_mode()
        elif bid == "btn-filter-clear":  self._clear_filter()
        elif bid == "btn-sort":          self._cycle_sort()
        elif bid == "btn-send":          self._do_send_single()
        elif bid == "btn-cyclic-start":  self._do_start_cyclic()
        elif bid == "btn-cyclic-stop":   self._do_stop_cyclic()
        # Trace controls
        elif bid == "btn-trace-record":
            if self._trace_buf.state == TraceState.PAUSED:
                self._trace_resume()
            else:
                self._trace_record()
        elif bid == "btn-trace-pause":   self._trace_pause()
        elif bid == "btn-trace-stop":    self._trace_stop()
        elif bid == "btn-trace-clear":   self._trace_clear()
        elif bid == "btn-trace-export":  self._trace_export_csv()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter-input":
            self._apply_filter_from_input()
        elif event.input.id in ("send-id", "send-data", "send-period", "send-name"):
            self._do_send_single()

    def action_connect(self)       -> None: self._do_connect()
    def action_disconnect(self)    -> None: self._do_disconnect()
    def action_refresh_ports(self) -> None: self._refresh_ports()
    def action_cycle_sort(self)    -> None: self._cycle_sort()

    def action_focus_filter(self) -> None:
        try:
            self.query_one("#filter-input", Input).focus()
        except Exception:
            pass

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self._log("Monitor PAUSED" if self.paused else "Monitor RESUMED")

    def action_clear_monitor(self) -> None:
        self._store.clear()
        self._table_rows.clear()
        self._change_ts.clear()
        try:
            self.query_one("#monitor-table", DataTable).clear()
        except Exception:
            pass
        self._log("Monitor cleared")

    def action_quit(self) -> None:
        if self.is_connected and self.can:
            self.can.close()
        self.exit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = CANBusTUI()
    app.run()
