# Waveshare USB-CAN-A Python Library and Diagnostic Tool

Version 1.4.0 | 2026-02-12

## Overview

Professional CAN bus interface library for Waveshare USB-CAN-A dongles with comprehensive diagnostic tools. Supports Standard (11-bit) and Extended (29-bit) CAN identifiers at speeds from 5kbps to 1Mbps.

## Files

- waveshare_can.py - Core library with thread-safe CAN interface
- can_diagnostic_tool.py - Full-featured CLI diagnostic tool
- Additional scripts for testing and examples

## Installation

### Requirements

```bash
pip install pyserial
```

### Optional (for DBC support)

```bash
pip install cantools
```

## Core Library: waveshare_can.py

### Quick Start

```python
from waveshare_can import WaveshareCAN, CANSpeed, CANMode

# Initialize
can = WaveshareCAN(port="/dev/ttyUSB0")

# Open and configure
can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)

# Start listening
can.start_listening()

# Send a message
can.send(0x123, bytes([0x11, 0x22, 0x33, 0x44]), is_extended=False)

# Receive a message
frame = can.get_frame(timeout=1.0)
if frame:
    print(f"ID: {hex(frame.can_id)}, Data: {frame.data.hex()}")

# Close
can.close()
```

### Class: WaveshareCAN

#### Initialization

```python
WaveshareCAN(port="/dev/ttyUSB0", timeout=0.1)
```

Parameters:
- port: Serial port device path (default: /dev/ttyUSB0)
- timeout: Serial read timeout in seconds (default: 0.1)

#### Methods

**open() -> bool**

Opens serial connection to the dongle.

Returns: True if successful, False otherwise

```python
if can.open():
    print("Connected")
```

**setup(speed=CANSpeed, mode=CANMode, extended=True) -> bool**

Configures CAN bus parameters.

Parameters:
- speed: CAN bus speed (see CANSpeed enum)
- mode: Operating mode (see CANMode enum)
- extended: Enable extended frame support (default: True)

Returns: True if successful

```python
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)
```

**start_listening()**

Starts background thread for receiving CAN frames. Frames are queued automatically.

```python
can.start_listening()
```

**stop_listening()**

Stops the background receiver thread.

```python
can.stop_listening()
```

**send(can_id, data, is_extended=True, verbose=False) -> bool**

Sends a single CAN frame.

Parameters:
- can_id: CAN identifier (0-0x7FF for Standard, 0-0x1FFFFFFF for Extended)
- data: Data bytes (0-8 bytes)
- is_extended: True for Extended (29-bit) ID, False for Standard (11-bit)
- verbose: Print debug information (default: False)

Returns: True if sent successfully

```python
# Standard frame
can.send(0x123, bytes([0xAA, 0xBB, 0xCC]), is_extended=False)

# Extended frame
can.send(0x12345678, bytes([0xDE, 0xAD, 0xBE, 0xEF]), is_extended=True)
```

**send_cyclic(name, can_id, data, period_ms, is_extended=True)**

Starts cyclic transmission at fixed interval.

Parameters:
- name: Unique identifier for this cyclic task
- can_id: CAN identifier
- data: Data bytes
- period_ms: Transmission period in milliseconds
- is_extended: True for Extended ID

```python
# Send heartbeat every 100ms
can.send_cyclic("heartbeat", 0x100, bytes([0x01]), period_ms=100, is_extended=False)
```

**stop_cyclic(name)**

Stops a cyclic transmission task.

```python
can.stop_cyclic("heartbeat")
```

**get_frame(timeout=None) -> CANFrame | None**

Retrieves a frame from the receive queue.

Parameters:
- timeout: Wait timeout in seconds (None = blocking, 0 = non-blocking)

Returns: CANFrame object or None if timeout

```python
# Blocking
frame = can.get_frame()

# Non-blocking
frame = can.get_frame(timeout=0)

# Wait up to 1 second
frame = can.get_frame(timeout=1.0)
```

**close()**

Closes connection and cleans up resources. Stops all cyclic tasks and receiver thread.

```python
can.close()
```

#### Callbacks

**on_message_received**

Set callback function for real-time message processing. Called from background thread when frame received.

```python
def my_handler(frame):
    print(f"Received: {frame}")

can.on_message_received = my_handler
```

### Enums

#### CANSpeed

```python
CANSpeed.SPEED_5K      # 5 kbps
CANSpeed.SPEED_10K     # 10 kbps
CANSpeed.SPEED_20K     # 20 kbps
CANSpeed.SPEED_25K     # 25 kbps
CANSpeed.SPEED_40K     # 40 kbps
CANSpeed.SPEED_50K     # 50 kbps
CANSpeed.SPEED_80K     # 80 kbps
CANSpeed.SPEED_100K    # 100 kbps
CANSpeed.SPEED_125K    # 125 kbps
CANSpeed.SPEED_200K    # 200 kbps
CANSpeed.SPEED_250K    # 250 kbps
CANSpeed.SPEED_400K    # 400 kbps
CANSpeed.SPEED_500K    # 500 kbps (most common)
CANSpeed.SPEED_666K    # 666 kbps
CANSpeed.SPEED_800K    # 800 kbps
CANSpeed.SPEED_1M      # 1 Mbps
```

#### CANMode

```python
CANMode.NORMAL           # Standard operation - receives both Standard and Extended
CANMode.LOOPBACK         # Transmit echoes back (self-test)
CANMode.SILENT           # Listen-only (no ACK)
CANMode.LOOPBACK_SILENT  # Combined
```

#### CANFrame

```python
frame.can_id        # CAN identifier (int)
frame.data          # Data bytes (bytes)
frame.is_extended   # True if Extended ID (bool)
frame.timestamp     # Unix timestamp (float)
```

### Usage Examples

#### Example 1: Basic Send/Receive

```python
from waveshare_can import WaveshareCAN, CANSpeed, CANMode

can = WaveshareCAN()
can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)
can.start_listening()

# Send
can.send(0x123, bytes([0x11, 0x22, 0x33, 0x44]), is_extended=False)

# Receive
frame = can.get_frame(timeout=1.0)
if frame:
    print(frame)

can.close()
```

#### Example 2: Cyclic Transmission

```python
can = WaveshareCAN()
can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)

# Start cyclic messages
can.send_cyclic("status", 0x100, bytes([0x01, 0x02]), period_ms=100)
can.send_cyclic("heartbeat", 0x200, bytes([0xFF]), period_ms=1000)

time.sleep(10)  # Run for 10 seconds

can.stop_cyclic("status")
can.stop_cyclic("heartbeat")
can.close()
```

#### Example 3: Callback Processing

```python
def message_handler(frame):
    if frame.can_id == 0x123:
        print(f"Important message: {frame.data.hex()}")

can = WaveshareCAN()
can.on_message_received = message_handler

can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)
can.start_listening()

# Messages trigger callback automatically
time.sleep(60)

can.close()
```

#### Example 4: Loopback Self-Test

```python
can = WaveshareCAN()
can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.LOOPBACK)
can.start_listening()

# Send message
can.send(0x123, bytes([0x11, 0x22, 0x33]))

# Should receive it back
frame = can.get_frame(timeout=0.5)
if frame and frame.can_id == 0x123:
    print("Loopback test PASSED")

can.close()
```

#### Example 5: Non-blocking Reception

```python
can = WaveshareCAN()
can.open()
can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)
can.start_listening()

while True:
    # Non-blocking check
    frame = can.get_frame(timeout=0)
    
    if frame:
        print(f"Frame received: {frame}")
    else:
        # Do other work
        time.sleep(0.01)
```

## Diagnostic Tool: can_diagnostic_tool.py

Command-line interface for CAN bus diagnostics.

### Usage

```bash
python3 can_diagnostic_tool.py [options]
```

### Options

```
--port, -p      Serial port (default: /dev/ttyUSB0)
--speed, -s     CAN bus speed in kbps (default: 500)
                Choices: 5, 10, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 666, 800, 1000
--mode, -m      CAN operating mode (default: normal)
                Choices: normal, loopback, silent, loopback-silent
--dbc           DBC file for signal decoding
--log           Log file for recording CAN traffic (CSV format)
--monitor       Use fixed-view monitor mode (default: logger mode)
--send ID DATA  Send single message
```

### Monitor Mode

Fixed-view display that updates in place. Shows real-time statistics, message rates, and decoded signals.

```bash
python3 can_diagnostic_tool.py --monitor --speed 500
```

Output:
```
================================================================================
CAN Bus Monitor - 2026-02-12 14:30:45
Total Frames: 1523 | Unique IDs: 8
================================================================================

          ID   Type  DLC Rate (Hz)                     Data Signals/Decoded
--------------------------------------------------------------------------------
    0x000100    Std    4       10.0              01 02 03 04 
    0x000200    Std    8        5.0  AA BB CC DD EE FF 00 11 
 0x12345678    Ext    4        1.0              DE AD BE EF 
--------------------------------------------------------------------------------

Press Ctrl+C to stop
```

### Logger Mode

Scrolling output with optional CSV logging.

```bash
python3 can_diagnostic_tool.py --log traffic.csv --speed 500
```

Output:
```
--- CAN Bus Logger Mode ---
[Std] ID: 0x100 | DLC: 4 | Data: 01 02 03 04
[Ext] ID: 0x12345678 | DLC: 4 | Data: DE AD BE EF
```

CSV format:
```
timestamp,can_id,data,is_extended
2026-02-12T14:30:45.123456,00000100,01020304,False
2026-02-12T14:30:45.234567,12345678,DEADBEEF,True
```

### Send Mode

Send single message and exit.

```bash
# Standard frame
python3 can_diagnostic_tool.py --send 0x123 "11 22 33 44"

# Extended frame (auto-detected if ID > 0x7FF)
python3 can_diagnostic_tool.py --send 0x12345678 "DE AD BE EF"
```

### DBC File Support

Decode signals using DBC database.

```bash
python3 can_diagnostic_tool.py --monitor --dbc vehicle.dbc
```

Requires: pip install cantools

### Examples

Monitor 250kbps bus:
```bash
python3 can_diagnostic_tool.py --monitor --speed 250
```

Log 1Mbps traffic to file:
```bash
python3 can_diagnostic_tool.py --log high_speed.csv --speed 1000
```

Send test message:
```bash
python3 can_diagnostic_tool.py --send 0x456 "AA BB CC DD EE FF"
```

Loopback self-test with monitoring:
```bash
python3 can_diagnostic_tool.py --monitor --mode loopback
```

## Protocol Details

### Waveshare USB-CAN-A Protocol

Serial: 2,000,000 bps, 8N1

#### Setup Command

Format: [0xAA, 0x55, 0x12, Speed, Mode, Filter(14 bytes), Checksum]

Working configuration (500kbps, receives both ID types):
```
[0xAA, 0x55, 0x12, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16]
```

- Byte 3: Speed (0x03 = 500kbps)
- Byte 4: Mode (0x01 = receives both Standard and Extended)
- Byte 19: Checksum (sum of bytes[2:19] & 0xFF)

#### Frame Format

Standard frame (11-bit ID):
```
[0xAA][Control][ID_Low][ID_High][Data...][0x55]
```

Extended frame (29-bit ID):
```
[0xAA][Control][ID0][ID1][ID2][ID3][Data...][0x55]
```

Control byte:
- Bits 7-6: Always 11 (data frame marker)
- Bit 5: Extended flag (1 = Extended, 0 = Standard)
- Bits 0-3: DLC (0-8)

Examples:
- Standard 4 bytes: 0xC4 = 11000100
- Extended 4 bytes: 0xE4 = 11100100
- Extended 8 bytes: 0xE8 = 11101000

ID encoding: Little Endian
- Standard: ID = byte0 | (byte1 << 8)
- Extended: ID = byte0 | (byte1<<8) | (byte2<<16) | (byte3<<24)

## Thread Safety

All library operations are thread-safe:
- Transmission uses mutex locking
- Reception uses dedicated thread with queue
- Cyclic tasks run in separate threads
- Queue capacity: 1000 frames

## Performance

- Maximum throughput: ~10,000 frames/second (USB bandwidth limited)
- Latency: <1ms from hardware to queue
- Queue overflow: Warning printed, frame dropped

## Troubleshooting

### Permission Denied

```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

### No Frames Received

- Check CAN bus termination (120Ω at both ends)
- Verify bus speed matches (both devices same speed)
- Check wiring: CAN_H, CAN_L, GND
- Test with loopback mode first

### Queue Full Warning

Process frames faster or increase queue size in __init__:
```python
self.rx_queue = queue.Queue(maxsize=2000)  # Increase from 1000
```

## Testing

Run comprehensive reliability tests:
```bash
python3 reliability_test.py
```

Quick loopback test:
```bash
python3 loopback_test.py
```

## License

Free to use and modify. No warranty provided.

## Support

For issues with Waveshare hardware, consult manufacturer documentation.
For library issues, verify with loopback test first.
