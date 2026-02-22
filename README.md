# Waveshare USB-CAN-A Python Library

A CAN bus monitoring and diagnostic suite for the
**Waveshare USB-CAN-A** dongle, built in Python with a full Terminal User
Interface (TUI).  Designed to run entirely in a terminal,
cross-platform, no GUI required.

---

## Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Arguments](#cli-arguments)
- [User Interface](#user-interface)
  - [Main Screen](#main-screen)
  - [Live Monitor](#live-monitor)
  - [Trace Tab](#trace-tab)
  - [Statistics Tab](#statistics-tab)
  - [DBC Tab](#dbc-tab)
  - [F4 Details Screen](#f4-details-screen)
  - [F6 Fullscreen Monitor](#f6-fullscreen-monitor)
  - [F7 Signal Discovery](#f7-signal-discovery)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Frame Type & Auto-Detection](#frame-type--auto-detection)
- [DBC File Support](#dbc-file-support)
- [Log Export Formats](#log-export-formats)
- [Known Hardware Limitations](#known-hardware-limitations)
- [File Overview](#file-overview)
- [Version History](#version-history)
- [ToDo / Roadmap](#todo--roadmap)

---

## Features

| Category | Capability |
|---|---|
| **Live Monitor** | Real-time per-ID table with byte-change highlighting, rate, count, DLC |
| **Stale Detection** | Frames with no data change for > 10 s turn red automatically |
| **Trace** | Chronological frame log with Record / Pause / Stop controls |
| **Export** | CSV, ASC, TRC (PEAK PCAN-View), BLF |
| **Statistics** | Top-N IDs by rate / count, bus load percentage |
| **Transmit** | Single-shot and cyclic frame transmission |
| **Filter** | Whitelist / blacklist by CAN-ID, space- or comma-separated |
| **DBC Support** | Load `.dbc` files for live signal decoding and message name display |
| **Signal Discovery** | Snapshot-based change detection with noise filtering (F7) |
| **Themes** | Multiple colour themes, cycle with `t` |
| **Frame Type** | Auto-detect Standard (11-bit) vs Extended (29-bit) on connect |
| **Cross-platform** | Linux and Windows (serial port detection automatic) |

---

## Hardware Requirements

- **Waveshare USB-CAN-A** dongle
  (USB to CAN bus converter, variable-length serial protocol)
- CAN bus to connect to (vehicle, test bench, Raspberry Pi with MCP2515, etc.)

> **Hardware note:** The Waveshare USB-CAN-A dongle can receive **only one
> frame type at a time** — either Standard (11-bit) or Extended (29-bit).
> This is a hardware/protocol limitation documented in the Waveshare spec
> (config byte 4: `0x01` = Standard only, `0x02` = Extended only).
> The tool handles this transparently via Auto-Detection on connect.
> See [Frame Type & Auto-Detection](#frame-type--auto-detection).

---

## Software Requirements

| Package | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | f-strings, `match`, dataclasses |
| `pyserial` | any | Serial communication with dongle |
| `textual` | ≥ 0.50 | TUI framework |
| `rich` | any | Coloured text in tables (installed with textual) |
| `cantools` | ≥ 39.0 | DBC file parsing and signal decoding |
| `python-can` | any | **Optional** — required for BLF export only |

---

## Installation

```bash
# 1. Clone or copy the project files
git clone https://github.com/yourname/waveshare-can-python
cd waveshare-can-python

# 2. Install dependencies
pip install pyserial textual cantools

# Optional: BLF export support
pip install python-can

# 3. On Linux: add your user to the dialout group (one-time)
sudo usermod -aG dialout $USER
# then log out and back in

# 4. Run
python3 can_tui.py
```

---

## Quick Start

```bash
# Basic start — select port and speed in the UI
python3 can_tui.py

# Auto-connect on startup (port may differ for your setup)
python3 can_tui.py --port /dev/ttyUSB0 --speed 500K --connect

# With DBC file loaded on startup (port may differ for your setup)
python3 can_tui.py --port /dev/ttyUSB0 --speed 500K --dbc my_car.dbc --connect

# Windows (port may differ for your setup)
python can_tui.py --port COM3 --speed 500K --connect
```

---

## CLI Arguments

| Argument | Short | Description |
|---|---|---|
| `--port PORT` | `-p` | Serial port (e.g. `/dev/ttyUSB0` or `COM3`) |
| `--speed SPEED` | `-s` | CAN bitrate: `5K 10K 20K 50K 100K 125K 200K 250K 400K 500K 800K 1M` |
| `--dbc FILE` | `-d` | Path to a `.dbc` file to load on startup |
| `--connect` | `-c` | Auto-connect on startup (requires `--port`) |

---

## User Interface

### Main Screen

The main screen is split into two columns:

**Left column**
- Connection panel (port, speed, mode, frame type selectors)
- Live Monitor table

**Right column**
- Status panel (connection state, RX counters, bus load, detected frame type)
- Transmit panel (single-shot and cyclic transmission)
- Filter & Sort panel

A tabbed area on the right holds: **Monitor · Trace · Statistics · DBC**

---

### Live Monitor

The central table updates every 200 ms and shows one row per unique CAN-ID:

| Column | Description |
|---|---|
| ID | Hex CAN-ID, suffixed with DBC message name if loaded |
| Type | `Std` (11-bit) or `Ext` (29-bit) |
| DLC | Data length in bytes |
| Rate Hz | Frames per second (rolling average) |
| Count | Total frames received |
| Data | Hex bytes — changed bytes highlighted for 1 second |

**Stale highlighting:** If a frame's data has not changed for more than
10 seconds, the DLC and Data cells turn red. This makes inactive signals
immediately visible on a busy bus.

---

### Trace Tab

Chronological log of every frame received (and transmitted):

- **Record** — start recording (blinking `● REC` indicator in header)
- **Pause** — freeze the display without losing incoming frames
- **Stop** — end recording, keep buffer for export
- **Clear** — discard the trace buffer
- **Export** — save to file in the selected format

Trace columns: `Time (s)` · `CAN-ID` · `Dir (Rx/Tx)` · `DLC` · `Data`

A warning is shown when the buffer exceeds 100 000 frames.

---

### Statistics Tab

Top-30 CAN-IDs ranked by receive rate, with a bus load summary line.
Refreshes every 2 seconds.

---

### DBC Tab

Load a `.dbc` file to enable:
- Message names shown next to CAN-IDs in the monitor
- Live signal decoding in a dedicated signals table
- DBC status in Signal Discovery results

---

### F4 Details Screen

Push `F4` to open a full-screen overlay with:
- Extended log view (last 200 lines)
- DBC file path input and load/unload controls
- Live decoded signals table (refreshes every second)

---

### F6 Fullscreen Monitor

Push `F6` for a fullscreen version of the live monitor table.
Main-screen timers are paused while F6 is active.
Click any row to pin/unpin that CAN-ID for focused monitoring.

---

### F7 Signal Discovery

The Signal Discovery screen (`F7`) helps identify which CAN-IDs change
in response to a specific physical action (door, button, switch, etc.).
It works by comparing two bus snapshots and showing the byte-level diff.

#### Workflow — with Noise Filter (recommended on active buses)

```
[o] Observe Start
      │
      │  Watch the bus for a few seconds while it is idle.
      │  Every CAN-ID that changes is added to the Noise set.
      │  The header shows live: "Noise erkannt: 12 IDs"
      │
[c] Capture Start  ←── Observe ends, Snapshot 1 taken
      │
      │  Perform the physical action (open door, press button, …)
      │
[x] Capture Stop   ←── Snapshot 2 taken
      │
      ▼
  RESULTS table — Noise IDs filtered out automatically
```

#### Workflow — without Noise Filter (quick mode)

Press `[c]` directly without `[o]` first.  
No noise filtering is applied — all changed IDs are shown.

#### Results Table

| Column | Description |
|---|---|
| CAN-ID | Hex identifier |
| Status | `✓ MessageName` (DBC known) · `⚠ Unbekannt` (no DBC match) |
| Δ Bytes | Indices of bytes that changed, e.g. `[0] [3] [5]` |
| VORHER | Byte values before action — changed bytes in **red** |
| NACHHER | Byte values after action — changed bytes in **green** |

#### Discovery Controls

| Key | Action |
|---|---|
| `o` | Observe Start — build noise baseline |
| `c` | Capture Start — from IDLE, OBSERVING, or RESULTS |
| `x` | Capture Stop — compute and show results |
| `s` | Cycle sort: ID ↑ → Δ-Bytes ↓ → Status |
| `u` | Toggle filter: All results ↔ Unknown (no DBC match) only |
| `Esc` / `q` | Return to main screen |

#### Edge Cases

| Situation | Behaviour |
|---|---|
| `[c]` without prior `[o]` | No noise filter — all changes shown |
| All changes were noise | Header: "All changes were noise — shorten Observe or use `[c]` directly" |
| `Del` during OBSERVING | Noise baseline reset; Observe continues with fresh data |
| `Del` during CAPTURING | Warning shown in results: data may be incomplete |
| Empty store at `[c]` | Warning: connect first and receive frames |

---

## Keyboard Shortcuts

### Main Screen

| Key | Action |
|---|---|
| `F1` | Keyboard shortcuts help |
| `F2` | Connect |
| `F3` | Disconnect |
| `F4` | Details screen (log + DBC) |
| `F5` | Refresh port list |
| `F6` | Fullscreen monitor |
| `F7` | Signal Discovery |
| `Space` | Pause / resume live monitor |
| `Del` | Clear monitor and frame store |
| `s` | Cycle sort mode |
| `f` | Focus filter input |
| `t` | Cycle colour theme |
| `q` | Quit |

---

## Frame Type & Auto-Detection

The Waveshare USB-CAN-A dongle accepts only **one frame type at a time**:

| Setting | Byte 4 | Receives |
|---|---|---|
| Standard | `0x01` | 11-bit IDs only |
| Extended | `0x02` | 29-bit IDs only |

The **Frame Type** selector in the connection panel offers three options:

- **Auto-Detect** *(default)* — on connect, listens 3 seconds for Extended
  frames. If none arrive, automatically switches to Standard and listens for
  another 3 seconds. The detected type is shown in the status panel and the
  selector is updated. If no frames arrive in either mode, a warning is shown
  suggesting a baudrate or bus-activity check.
- **Extended (29-bit)** — fixed Extended mode, no detection.
- **Standard (11-bit)** — fixed Standard mode, no detection.

> Most modern vehicle buses (CAN FD, ISO-TP diagnostics, OBD-II on newer
> vehicles) use Extended 29-bit IDs. Classic/legacy buses and many body
> control modules use Standard 11-bit IDs. Mixed buses (both types
> simultaneously) are rare in practice.

---

## DBC File Support

Load any `.dbc` file to enable message and signal decoding:

```bash
# Via CLI
python3 can_tui.py --dbc path/to/your.dbc

# Via UI
# F4 → DBC path input → Enter or "Load" button
```

With a DBC loaded:
- CAN-IDs in the monitor show the message name: `0x3B4 DoorControl`
- The DBC signals table decodes live signal values
- Signal Discovery shows `✓ MessageName` for known messages

> **Partial DBC coverage is common.** A message may be listed in the DBC
> with only some of its signals defined. Signal Discovery will correctly
> identify the message name, but bytes not covered by any signal definition
> are not further decoded. This is a known limitation and a planned
> improvement (see ToDo).

---

## Log Export Formats

| Format | Extension | Tool compatibility |
|---|---|---|
| CSV | `.csv` | Excel, Python, any text tool |
| ASC | `.asc` | Vector CANalyzer, CANoe |
| TRC | `.trc` | PEAK PCAN-View v2.1 |
| BLF | `.blf` | Vector tools (requires `python-can`) |

Export is available from the Trace tab after stopping a recording.

---

## Known Hardware Limitations

| Limitation | Details |
|---|---|
| One frame type at a time | Waveshare protocol byte 4: `0x01` Std or `0x02` Ext, no combined mode |
| No CAN FD support | The USB-CAN-A is a classic CAN (ISO 11898) adapter, max 1 Mbit/s |
| Serial baud fixed at 2 Mbit/s | USB ↔ dongle communication speed is fixed in hardware |
| Same CAN-ID in Std and Ext | If a Standard and Extended frame share the same numeric ID, they occupy the same row in the monitor (latent issue, rare in practice) |

---

## File Overview

| File | Description |
|---|---|
| `can_tui.py` | Main TUI application — all screens, timers, UI logic |
| `waveshare_can.py` | Low-level Waveshare dongle driver — serial protocol, RX thread |
| `can_log_exporter.py` | Multi-format trace export module (CSV / ASC / TRC / BLF) |
| `can_diagnostic_tool.py` | Standalone CLI diagnostic tool (independent of TUI) |
| `example_CAN.dbc` | Example DBC file for testing signal decoding |

---

## Version History

| Version | Date | Highlights |
|---|---|---|
| v0.9.1 | 2026-02-19 | Timer optimisations, blinking REC indicator |
| v1.0.0 | 2026-02-22 | Multi-format trace export (CSV / ASC / TRC / BLF), format selector |
| v1.1.0 | 2026-02-22 | Signal Discovery screen (F7), Stale Value Highlighting (10 s) |
| v1.2.0 | 2026-02-22 | `CANFrameType` enum, Frame Type selector, Auto-Detection on connect |
| v1.3.0 | 2026-02-22 | Discovery Observe phase: noise baseline, `[o]` key, noise filtering |

---

## ToDo / Roadmap

The following features are planned but not yet implemented, roughly in
priority order:

**Signal Discovery**
- **Soft noise marking (Option A)** — instead of hard-filtering noise IDs,
  mark them with `~` status and allow toggling their visibility with a key.
  Currently noise IDs are always hidden (strict mode only).
- **Byte-level DBC coverage check** — for known messages, flag bytes that are
  not covered by any signal definition as `~ Undefined byte`. Currently the
  status is binary (message known / unknown), but a message can have undocumented
  signals occupying bytes that the DBC does not define.
- **Bit-diff display** — for changed bytes, show which individual bits changed
  (e.g. `0x1F → 0x2F  bit 4↓ bit 5↑`). High value for manual reverse
  engineering without a DBC.
- **Signal value decoding in results** — for known messages with a DBC loaded,
  decode and display signal values (VORHER/NACHHER) alongside byte values.
- **CSV / JSON export** of Discovery results.

**Monitor**
- **Configurable stale timeout** — currently fixed at 10 seconds (`STALE_TIMEOUT_S`).
- **CAN-ID collision handling** — Standard and Extended frames with the same
  numeric ID currently share one monitor row. Use `(can_id, is_extended)` tuple
  as the store key.

**Connection**
- **Configurable Auto-Detection timeout** — currently fixed at 3 seconds
  (`AUTO_DETECT_TIMEOUT_S`) per frame type.

**Export / Analysis**
- **Bit-level visualisation** — heatmap of which bits change most frequently
  across a recording session.
- **Replay** — send back a recorded trace to the bus for stimulus/response
  testing.
