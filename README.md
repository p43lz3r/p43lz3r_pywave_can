# Waveshare USB-CAN-A Python Library

Professional CAN bus workflows on a $25 dongle. Live monitoring, DBC signal decoding,
trace recording, and signal discovery — all from the terminal, on Linux and Windows.

---

![20260220_181255](https://github.com/user-attachments/assets/0bad2f4e-777d-4a2a-9fd4-12481b0eb92f)

---

> **All GUI elements are in English.**  
> Tested on **Ubuntu Linux** and **Windows 11**.

> **Terminal display:** The tool requires a **maximized terminal window** to render
> correctly. On some displays, zooming out one step (`Ctrl` + `-`) before
> launching can improve layout density.

> **Active development:** The Textual-based UI is under ongoing development.
> Layout and usability improvements are being worked on.

---

## Requirements

**Python 3.9+** is required on both platforms.

```bash
pip install textual cantools pyserial
pip install python-can          # required for PEAK PCAN-USB and BLF export
```

### Linux — Waveshare USB-CAN-A

Add a udev rule so the device is accessible without `sudo`:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/99-waveshare-can.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Linux — PEAK PCAN-USB

The kernel `peak_usb` module is loaded automatically on plug-in on most distributions.
Verify with:

```bash
ip link show type can      # should list can0, can1, …
```

No extra driver install needed. `python-can` handles everything via socketcan.

### Windows — PEAK PCAN-USB

Download and install the PEAK System driver package from
https://www.peak-system.com/Downloads. The installer places `PCANBasic.dll`
which `python-can` uses automatically.

---

## Quick Start

```bash
python can_tui.py
```

Optional CLI arguments let you pre-configure the session:

| Argument | Short | Description | Example |
|---|---|---|---|
| `--port PORT` | `-p` | Serial port (Waveshare) | `-p /dev/ttyUSB0` or `-p COM3` |
| `--speed SPEED` | `-s` | CAN bitrate | `-s 500K` |
| `--dbc FILE` | `-d` | Load a DBC file on startup | `-d vehicle.dbc` |
| `--connect` | `-c` | Auto-connect on startup | requires `--port` |

Valid speed values: `5K 10K 20K 50K 100K 125K 200K 250K 400K 500K 800K 1M`

**Example — auto-connect at 500 kbps with DBC decoding:**
```bash
python can_tui.py -p /dev/ttyUSB0 -s 500K -d my_vehicle.dbc -c
```

---

## Using the TUI

### Connection Panel

When the tool starts, the **Connection Panel** is shown on the left.

1. Select your **Hardware** (Waveshare USB-CAN-A or PEAK PCAN-USB).
2. Select the **Port / Channel** (auto-detected; click **Refresh** if your device
   is not listed).
3. Set **Speed** to match your CAN bus (500 kbps is the most common for automotive).
4. Set **Mode**: Normal for monitoring, Silent to listen without transmitting.
5. Set **Frame Type**: *Auto* lets the tool detect Standard vs. Extended frames;
   choose manually if you know the bus type.
6. Press **F2** to connect.

### F-Key Reference

| Key | Action |
|---|---|
| **F2** | Connect to hardware |
| **F3** | Disconnect |
| **F4** | Message Details — expand a selected CAN-ID |
| **F5** | Refresh port list |
| **F6** | Fullscreen Monitor with Byte Inspector |
| **F7** | Signal Discovery screen |
| **F8** | Toggle Connection Panel |
| **Space** | Pause / resume live display |
| **Delete** | Clear monitor table |
| **S** | Cycle sort mode (by ID / Rate / Count) |
| **F** | Focus filter input |
| **Q** | Quit |

### Main Monitor Table

Once connected, the monitor shows a live table — one row per CAN-ID:

| Column | Description |
|---|---|
| CAN-ID | Message identifier (hex) |
| DLC | Data Length Code (0–8 bytes) |
| Data | Payload bytes in hex — **yellow** = recently changed, **red** = stale (>10 s) |
| Rate | Frames per second for this ID |
| Count | Total frames received |
| Signal | Decoded signal name from DBC (if loaded) |

### Signal Discovery (F7)

Signal Discovery helps you find which CAN-IDs change when you interact with
a vehicle component (press a button, turn a knob, etc.).

1. **[O] Observe** — captures baseline bus noise. Operate the bus normally
   without touching the target component. Press **[O]** again to stop observing.
2. **[C] Capture** — takes a snapshot of the current bus state.
   Trigger your target action, then press **[C]** again for a second snapshot.
3. The results table shows byte-level differences between snapshots,
   filtered against the noise baseline. IDs already covered by your DBC are
   marked accordingly.

### Trace Recording

The **Trace** tab records every frame with a relative timestamp.

- **[R]** Start recording / **[R]** Stop
- **[P]** Pause / resume without losing frames
- **[S]** Scroll-to-live toggle
- **Export** button opens a save dialog — choose ASC, TRC, BLF, or CSV

### DBC Signal Decoding

Load a `.dbc` file from the **DBC** tab or via `--dbc` on the command line.
Once loaded, decoded signal values appear in the main monitor and in the
Fullscreen Monitor signal list (F6). The DBC tab shows all messages and
their decode coverage.

### Transmit (TX)

The **TX** tab allows sending frames manually or cyclically.

- Enter a CAN-ID (hex), DLC, data bytes, and an optional period (ms).
- Leave period empty for single-shot transmission.
- Named cyclic tasks appear in the task list and can be stopped individually.

### Fullscreen Monitor + Byte Inspector (F6)

Fullscreen monitor shows the live signal table on the left.
On the right, the **Byte Inspector** renders the selected message as an
8-byte hex grid with per-byte change highlighting. Navigate with arrow keys,
select bytes with **Space**, and see multi-byte integer / float decodes
(Big-Endian and Little-Endian) in the panel below.

---

## Features

- Live monitor — one row per CAN-ID, byte-change highlighting, stale detection
- Signal Discovery with noise-baseline filtering (F7)
- DBC file support via `cantools` — signal decode in monitor and fullscreen view
- Byte Inspector — hex grid + UInt/Int/Float BE/LE decode with scaling (x0.1/0.01/0.001)
- Frame filtering — whitelist/blacklist by CAN-ID
- Trace recording with PCAN-View-style controls
- Log export: **ASC** (Vector CANalyzer), **TRC** (PCAN-View v1.1), **BLF** (Vector binary), **CSV**
- Cyclic and single-shot frame transmission
- Bus-load bar and per-ID statistics
- Auto frame-type detection (Standard vs. Extended)
- Cross-platform: Ubuntu Linux + Windows 11

---

## Project Architecture

```
can_tui.py              — Textual TUI application (main entry point)
     │
     ├── waveshare_can.py   — Waveshare USB-CAN-A backend (serial protocol)
     ├── peak_can.py        — PEAK PCAN-USB backend (python-can / socketcan)
     │        └── Both backends expose an identical public API so the
     │            frontend never needs to know which hardware is active.
     │
     ├── can_log_exporter.py — Export engine (ASC / TRC / BLF / CSV)
     └── theme_midnight.py   — UI colour theme constants
```

### Hardware Backend Contract

Both `WaveshareCAN` and `PeakCAN` expose the same interface:

```python
can.open()                          # → bool
can.setup(speed, mode, frame_type)  # → bool
can.start_receiver(callback)        # callback(CANFrame)
can.send(can_id, data, is_extended) # → bool
can.send_cyclic(can_id, data, is_extended, period_ms, task_name)
can.stop_cyclic(task_name)
can.close()
```

Shared data types (`CANFrame`, `CANSpeed`, `CANMode`, `CANFrameType`) are
defined once in `waveshare_can.py` and imported by `peak_can.py` and
`can_tui.py`. To add a new hardware backend, implement the contract above
and add it to the `HardwareType` enum and `HARDWARE_OPTIONS` list in
`can_tui.py`.

---

## Developer: Using waveshare_can.py Standalone

`waveshare_can.py` has no dependency on the TUI and can be used independently
in any Python project.

```python
from waveshare_can import WaveshareCAN, CANFrame, CANSpeed, CANMode, CANFrameType

# --- Open and configure ---
can = WaveshareCAN(port="/dev/ttyUSB0")   # Windows: port="COM3"
can.open()
can.setup(
    speed=CANSpeed.SPEED_500K,
    mode=CANMode.NORMAL,
    frame_type=CANFrameType.EXTENDED,
)

# --- Receive frames via callback ---
def on_frame(frame: CANFrame):
    print(f"ID=0x{frame.can_id:X}  data={frame.data.hex()}  ext={frame.is_extended}")

can.start_receiver(on_frame)

# --- Or poll the queue directly ---
import queue, time
time.sleep(0.5)
while not can.rx_queue.empty():
    frame: CANFrame = can.rx_queue.get_nowait()
    print(frame)   # uses CANFrame.__str__()

# --- Send a frame ---
can.send(can_id=0x123, data=bytes([0x01, 0x02, 0x03]), is_extended=False)

# --- Send cyclically (every 100 ms) ---
can.send_cyclic(
    can_id=0x200,
    data=bytes([0xDE, 0xAD]),
    is_extended=False,
    period_ms=100,
    task_name="heartbeat",
)
time.sleep(1)
can.stop_cyclic("heartbeat")

# --- Clean up ---
can.close()
```

### CANFrame Attributes

| Attribute | Type | Description |
|---|---|---|
| `can_id` | `int` | CAN identifier (raw, without extended flag) |
| `data` | `bytes` | Payload (0–8 bytes) |
| `is_extended` | `bool` | True = 29-bit extended frame |
| `timestamp` | `float` | `time.time()` at receive |

### CANSpeed Values

`SPEED_5K` `SPEED_10K` `SPEED_20K` `SPEED_50K` `SPEED_100K` `SPEED_125K`
`SPEED_200K` `SPEED_250K` `SPEED_400K` `SPEED_500K` `SPEED_800K` `SPEED_1M`

### CANMode Values

| Value | Description |
|---|---|
| `NORMAL` | Standard bidirectional operation |
| `SILENT` | Listen-only, no ACK transmitted |
| `LOOPBACK` | Internal loopback for testing |
| `LOOPBACK_SILENT` | Loopback without ACK |

---

## Developer: Using peak_can.py Standalone

`peak_can.py` mirrors the `WaveshareCAN` API exactly. Replace `WaveshareCAN`
with `PeakCAN` and use the appropriate channel name for your platform:

```python
from peak_can import PeakCAN, detect_peak_channels
from waveshare_can import CANSpeed, CANMode, CANFrameType

channels = detect_peak_channels()   # ['can0'] on Linux, ['PCAN_USBBUS1'] on Windows
print("Available channels:", channels)

can = PeakCAN(channel=channels[0])
can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL,
          frame_type=CANFrameType.EXTENDED)
can.start_receiver(lambda f: print(f))
```

---

## Known Limitations

### Waveshare USB-CAN-A

- **Standard OR Extended frames only** — the dongle hardware accepts one frame
  type per session. Auto-Detection picks the correct type at connect time.
  Buses that mix Standard and Extended frames on the same session are not
  fully supported on this device.
- **No hardware timestamps** — timestamps are assigned by the host CPU on
  receive. Precision timing analysis (inter-frame gaps < ~1 ms) is not reliable.
- **No CAN FD support** — the dongle is limited to classical CAN (max 1 Mbps,
  8-byte payload).
- **Throughput limit** — the USB serial bridge runs at 2 Mbps which limits
  practical sustained throughput on heavily loaded buses.

### PEAK PCAN-USB

- CAN FD is not yet supported in this tool (hardware capability exists).
- Silent mode on Linux socketcan uses local loopback disable as an approximation;
  behaviour may differ slightly from the Windows PCAN driver.

---

## Roadmap

- UDS / ISO-TP protocol layer (ISO 14229 request/response)
- Bit-flip visualizer and entropy heatmap for signal hypothesis
- OBD-II correlation for known PID mapping
- ESP32-S3 + MCP2515 backend (WiFi/Bluetooth access)
- CAN FD support (PEAK hardware)
- Fuzzing / frame injection workflow

---

## Contributing

Pull requests are welcome. Please follow **PEP 8** for all Python code.
Open an issue first for larger feature additions so we can align on the
approach before you invest the time.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
