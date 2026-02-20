# 2026-02-20 12:00 v0.9.0 - Initial README

# CAN Bus TUI

A professional-grade, terminal-based CAN bus monitoring and diagnostic tool for the
**Waveshare USB-CAN-A** adapter.  Built with Python and the
[Textual](https://github.com/Textualize/textual) TUI framework.

---

## Features

| Category | Details |
|---|---|
| **Live Monitor** | Real-time DataTable of all active CAN IDs – rate (Hz), frame count, byte-level change highlighting |
| **DBC Decoder** | Load any Vector-standard `.dbc` file; message names shown in the main monitor and fullscreen view; live signal-count indicators in the DBC tab |
| **Fullscreen Monitor (F6)** | 2-column split: left = ID list with selection toggle, right = decoded signals with physical values and units (raw hex fallback when no DBC loaded) |
| **Trace Recorder** | Chronological frame log with Record / Pause / Stop / Export CSV; Pause buffers frames silently – all are flushed on Resume (PCAN-View behaviour) |
| **Transmit** | Single-shot and cyclic frame transmission with configurable period and task name |
| **Filter & Sort** | Whitelist / blacklist filter on CAN IDs; sort by ID, rate, or count |
| **Statistics** | Top-N IDs by frame rate, bus-load estimation |
| **5 Themes** | Midnight Commander, Norton Commander, Amber, Green, Modern |
| **Cross-platform** | Linux (`/dev/ttyUSB*`, `/dev/ttyACM*`) and Windows (`COMx`) |
| **CLI Arguments** | Pre-set port, bitrate, DBC file and auto-connect at startup |

---

## Requirements

### Hardware
- Waveshare USB-CAN-A (or compatible) adapter

### Software

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | Runtime |
| [textual](https://pypi.org/project/textual/) | ≥ 0.40 | TUI framework |
| [pyserial](https://pypi.org/project/pyserial/) | ≥ 3.5 | Serial communication |
| [cantools](https://pypi.org/project/cantools/) | ≥ 39.0 | DBC file parsing |
| [rich](https://pypi.org/project/rich/) | ≥ 13.0 | Terminal colour rendering |

Install all dependencies:

```bash
pip install textual pyserial cantools rich
```

### Project files

Both files must be in the same directory:

```
can_tui.py          # Main application  (this file)
waveshare_can.py    # Hardware driver   (v1.0.2 required)
```

---

## Usage

### Basic start

```bash
python3 can_tui.py
```

### Command-line arguments

```
usage: can_tui.py [-h] [-p PORT] [-s SPEED] [-d FILE] [-c]

options:
  -p, --port   PORT   Serial port  (e.g. /dev/ttyUSB0  or  COM3)
  -s, --speed  SPEED  CAN bitrate  (see valid values below)
  -d, --dbc    FILE   Path to a .dbc file to load on startup
  -c, --connect       Auto-connect on startup (requires --port)
```

**Valid speed values:** `5K  10K  20K  50K  100K  125K  200K  250K  400K  500K  800K  1M`

### Examples

```bash
# Pre-load a DBC file only
python3 can_tui.py --dbc /path/to/my.dbc

# Pre-select port and bitrate, open UI manually
python3 can_tui.py --port /dev/ttyUSB0 --speed 250K

# Fully automatic start
python3 can_tui.py -p /dev/ttyUSB0 -s 500K -d /path/to/my.dbc --connect

# Windows example
python3 can_tui.py -p COM3 -s 500K --connect
```

---

## Keyboard Shortcuts

### Main Screen

| Key | Action |
|---|---|
| `F1` | Help overlay |
| `F2` | Connect |
| `F3` | Disconnect |
| `F4` | Open Details screen (Log / Statistics / DBC / Trace) |
| `F5` | Refresh port list |
| `F6` | Fullscreen Monitor |
| `Space` | Pause / Resume live monitor |
| `Del` | Clear monitor table |
| `s` | Cycle sort mode (ID ↑ / Rate ↓ / Count ↓) |
| `f` | Focus filter input |
| `t` | Cycle theme |
| `q` | Quit |

### Details Screen

| Key | Action |
|---|---|
| `Esc` / `q` | Back to main |
| `t` | Cycle theme |
| `F1` | Help overlay |

### Fullscreen Monitor (F6)

| Key | Action |
|---|---|
| `Space` / `Enter` | Toggle ID selection |
| `Esc` / `q` | Back to main |
| `t` | Cycle theme |

---

## Screen Overview

### Main Screen

```
┌─ CONNECTION ─────────────────┐  ┌─ STATUS ───────────────────────┐
│ Port  [/dev/ttyUSB0     ]    │  │ Connected  Port: ...  Speed:... │
│ Speed [500 kbps         ]    │  │ RX: 12345  Frames/s: 450.2     │
│ Mode  [Normal           ]    │  │ Bus Load:  42.1%  [████░░░░░]   │
│ [Connect F2] [Disconnect F3] │  ├─ TRANSMIT ─────────────────────┤
├─ LIVE MONITOR ───────────────┤  │ ID: [0x123]  Ext [x]           │
│ ID          Type DLC Rate Hz │  │ Data: [DE AD BE EF]             │
│ 0x123 MSG_A Std  8   100.0   │  │ Period: [100ms]  Name: [task1]  │
│ 0x456 MSG_B Ext  4    50.0   │  ├─ FILTER & SORT ────────────────┤
│ ...                          │  │ IDs: [              ] Whitelist │
└──────────────────────────────┘  └────────────────────────────────┘
```

### Details Screen – DBC Decoder Tab

The DBC tab shows all messages defined in the loaded `.dbc` file.
The **Signals** column turns green and shows the decoded signal count for any
message whose CAN ID is currently being received on the bus.

```
┌─ DBC FILE LOADER ─────────────────────────────────────────────┐
│ Path: [/home/user/vehicle.dbc              ] [Load] [Unload]  │
│ Loaded: vehicle.dbc  |  47 messages                           │
├─ MESSAGES IN DATABASE ────────────────────────────────────────┤
│ CAN-ID      Message Name           DLC   Signals              │
│ 0x123       EngineStatus           8     5  ← green = live    │
│ 0x1A4       TransmissionData       8     3                    │
│ 0x98920004  BatteryPack1Status     8     6                    │
│ ...                                                           │
└───────────────────────────────────────────────────────────────┘
```

### Fullscreen Monitor (F6)

```
┌─ ID LIST  (Space = toggle) ──┐  ┌─ LIVE DECODED SIGNALS ───────────────┐
│ Sel  CAN-ID      Name  Rate  │  │ CAN-ID    Name   Signal      Value  U │
│ [X]  0x123  EngineStatus 100 │  │ 0x123  EngStat  RPM         3450   /m │
│ [ ]  0x1A4  TransData     50 │  │ 0x123  EngStat  Throttle      42   %  │
│ [X]  0x456  BrakePressure 20 │  │ 0x456  Brake    FrontLeft   12.5  bar │
│ ...                          │  │ 0x456  Brake    RearRight   11.8  bar │
└──────────────────────────────┘  └──────────────────────────────────────┘
```

### Trace Recorder

| State | Behaviour |
|---|---|
| **IDLE** | No recording; data retained from last session until Clear |
| **REC** | All Rx and Tx frames recorded with relative timestamp |
| **PAUSED** | Display frozen; frames continue to buffer in memory; counter shows `(+N buffered)`; Resume flushes all buffered frames in order |

Export produces a CSV file named `can_trace_YYYYMMDD_HHMMSS.csv` in the working directory.

---

## DBC File Support

The tool uses [cantools](https://github.com/eerimoq/cantools) to parse
Vector-standard `.dbc` files.

- Standard (11-bit) and Extended (29-bit / J1939) CAN IDs are supported
- Physical values are computed from scale/offset defined in the DBC
- Signal units are displayed in the Fullscreen Monitor
- If a received CAN ID is not present in the loaded DBC, raw hex is shown as fallback

---

## Architecture Notes

| Component | Description |
|---|---|
| `waveshare_can.py` | Hardware driver – serial framing, `CANFrame` dataclass, `CANSpeed` / `CANMode` enums |
| `CANFrameStore` | Thread-safe store for latest frame per CAN ID; dirty-flag for efficient UI updates |
| `TraceBuffer` | Unbounded thread-safe recorder with IDLE / RECORDING / PAUSED state machine and pending queue |
| `DBCDatabase` | Thread-safe cantools wrapper; indexed by `frame_id` for O(1) lookup |
| `CANBusTUI` | Textual `App`; owns all timers, all screen references, DBC instance |
| `DetailsScreen` | Secondary screen: Event Log, Statistics, DBC Decoder, Trace |
| `MonitorFullScreen` | F6 fullscreen split-view with ID selection and live decoded signals |

Background RX runs in a dedicated thread owned by `waveshare_can.py`.
All UI updates happen exclusively on the Textual event-loop thread via timer callbacks.
Cross-thread communication uses only locks and queues – no shared mutable state without protection.

---

## Tested Hardware & Environments

| Hardware | OS | Status |
|---|---|---|
| Waveshare USB-CAN-A | Ubuntu 24.04 | ✓ |
| Waveshare USB-CAN-A | Windows 11 | ✓ |
| PEAK USB dongle | Ubuntu 24.04 | ✓ (via loopback) |
| Raspberry Pi CAN HAT | Raspberry Pi OS | ✓ |

---

## Version History

| Version | Date | Summary |
|---|---|---|
| v0.9.0 | 2026-02-20 | DBC support, Fullscreen Monitor (F6), CLI args, context-sensitive shortcuts, Trace Pause behaviour matching professional tools |
| v0.8.1 | 2026-02-19 | F4 for Details screen, PEP 8 compliance, type hints, named constants, `on_unmount()` cleanup |
| v0.8.0 | — | Trace recorder with Record/Pause/Stop/Export CSV, 5 themes |
| v0.7.x | — | Cyclic transmit, filter whitelist/blacklist, statistics tab |
| v0.6.x | — | Multi-screen architecture (Details screen), event log |
