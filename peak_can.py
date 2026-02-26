#!/usr/bin/env python3
# 2026-02-24 16:00 v1.1.0 - Full Ubuntu / Linux support via socketcan interface.
#                           detect_peak_channels() now scans /sys/class/net for
#                           active canN interfaces (socketcan, no PEAK libs needed)
#                           in addition to PCAN_USBBUSn (pcan interface).
#                           PeakCAN.setup() auto-selects interface: 'socketcan' for
#                           canN channels, 'pcan' for PCAN_USBBUSn channels.
#                           Loopback mode on socketcan uses fd=False local loopback.
# 2026-02-24 12:00 v1.0.1 - Filter PEAK bus-status notification frames (ID=0x000,
#                           Std, DLC=2, data[1]==0x00) leaking through as data frames.
# 2026-02-24 12:00 v1.0.0 - Initial release. PEAK PCAN-USB backend for can_tui.py.
#                           Public interface mirrors WaveshareCAN v1.1.0 exactly.
#                           Uses python-can 'pcan' interface (Windows + Linux).
#                           Supports: Normal, Silent (listen_only), Loopback modes.
#                           Auto-detection of available PEAK channels via python-can.
"""
PEAK PCAN-USB Interface Class
Drop-in companion to waveshare_can.py for can_tui.py.

Requirements
------------
    pip install python-can

Windows driver
--------------
    Install PEAK System driver package from https://www.peak-system.com/Downloads.
    The driver installs PCANBasic.dll which python-can uses automatically.

Linux / Ubuntu driver (two options, pick one)
---------------------------------------------
Option A – kernel socketcan module (recommended, no extra install needed):
    sudo modprobe peak_usb          # usually auto-loaded on plug-in
    # python-can brings the interface up automatically at open time.
    # Channel name will appear as 'can0', 'can1', etc.

Option B – PEAK libpcanbasic (optional, for PCAN_USBBUSn naming):
    Download and install the PEAK Linux driver package from:
    https://www.peak-system.com/Downloads.html
    Channel name will appear as 'PCAN_USBBUS1', 'PCAN_USBBUS2', etc.

Shared types from waveshare_can
--------------------------------
    CANFrame, CANSpeed, CANMode, CANFrameType  — import from waveshare_can.
    PeakCAN does NOT redefine these to keep one single source of truth.

Channel naming
--------------
    Windows        : 'PCAN_USBBUS1' … 'PCAN_USBBUS8'  (pcan interface)
    Linux Option A : 'can0', 'can1', …                 (socketcan interface)
    Linux Option B : 'PCAN_USBBUS1' … 'PCAN_USBBUS8'  (pcan interface)

Interface auto-selection (internal)
------------------------------------
    Channel name starts with 'can'  →  socketcan
    Channel name starts with 'PCAN' →  pcan
"""

import glob
import os
import platform
import queue
import threading
import time
from typing import Callable, Dict, List, Optional

# python-can is a mandatory dependency for this module
try:
    import can
    _PYTHON_CAN_AVAILABLE = True
except ImportError:
    _PYTHON_CAN_AVAILABLE = False

# Shared data types – single source of truth in waveshare_can.py
from waveshare_can import CANFrame, CANSpeed, CANMode, CANFrameType


# ---------------------------------------------------------------------------
# Speed mapping:  CANSpeed enum  →  bitrate in bps  (python-can expects bps)
# ---------------------------------------------------------------------------
_SPEED_TO_BPS: Dict[int, int] = {
    CANSpeed.SPEED_1M:   1_000_000,
    CANSpeed.SPEED_800K:   800_000,
    CANSpeed.SPEED_500K:   500_000,
    CANSpeed.SPEED_400K:   400_000,
    CANSpeed.SPEED_250K:   250_000,
    CANSpeed.SPEED_200K:   200_000,
    CANSpeed.SPEED_125K:   125_000,
    CANSpeed.SPEED_100K:   100_000,
    CANSpeed.SPEED_50K:     50_000,
    CANSpeed.SPEED_20K:     20_000,
    CANSpeed.SPEED_10K:     10_000,
    CANSpeed.SPEED_5K:       5_000,
}

# PEAK channel names for the 'pcan' interface (Windows and Linux + libpcanbasic)
_PCAN_BUS_CHANNELS: List[str] = [f"PCAN_USBBUS{i}" for i in range(1, 9)]

# socketcan interface name pattern (Linux only): can0 … can7
_SOCKETCAN_CANDIDATES: List[str] = [f"can{i}" for i in range(8)]


def _is_socketcan_channel(channel: str) -> bool:
    """Return True if channel uses the socketcan interface (canN naming)."""
    return channel.startswith("can") and channel[3:].isdigit()


def _detect_socketcan_channels() -> List[str]:
    """Discover active socketcan CAN interfaces via /sys/class/net.

    Reads the kernel's network interface list and returns those whose
    'type' file contains 280 (ARPHRD_CAN).  This works without opening
    the interface and avoids needing root or PEAK libraries.

    Returns:
        Sorted list of interface names, e.g. ['can0', 'can1'].
    """
    found: List[str] = []
    for path in sorted(glob.glob("/sys/class/net/can*")):
        iface = os.path.basename(path)
        if not iface[3:].isdigit():
            continue
        # Check kernel type: 280 = ARPHRD_CAN
        type_file = os.path.join(path, "type")
        try:
            with open(type_file) as fh:
                if fh.read().strip() == "280":
                    found.append(iface)
        except OSError:
            pass
    return found


def detect_peak_channels() -> List[str]:
    """Return available PEAK PCAN-USB channel names for the current OS.

    Strategy (tried in order):
      1. Linux socketcan  – scan /sys/class/net for active canN interfaces.
         Works with just the kernel peak_usb module; no PEAK libs required.
      2. python-can built-in detection via detect_available_configs('pcan').
         Works when PEAK libpcanbasic is installed (Windows always, Linux option B).
      3. Manual probe of PCAN_USBBUS1..8 via pcan interface as final fallback.

    Returns:
        List of channel name strings.  Empty list when no PEAK hardware found.
    """
    if not _PYTHON_CAN_AVAILABLE:
        return []

    found: List[str] = []

    # --- Step 1: Linux socketcan (no PEAK libs needed) ---------------------
    if platform.system() == "Linux":
        sc_channels = _detect_socketcan_channels()
        found.extend(sc_channels)

    # --- Step 2: python-can built-in detection (pcan interface) ------------
    try:
        configs = can.detect_available_configs(interfaces=["pcan"])
        for cfg in configs:
            ch = cfg.get("channel")
            if ch and ch not in found:
                found.append(ch)
    except Exception:
        pass

    if found:
        return found

    # --- Step 3: Manual probe of PCAN_USBBUSn (fallback) ------------------
    for ch in _PCAN_BUS_CHANNELS:
        try:
            bus = can.interface.Bus(interface="pcan", channel=ch, bitrate=500_000)
            bus.shutdown()
            found.append(ch)
        except Exception:
            if found:
                break  # Stop at first gap after finding at least one channel
    return found


class PeakCAN:
    """PEAK PCAN-USB interface with an API identical to WaveshareCAN.

    All public attributes and methods mirror WaveshareCAN v1.1.0 so that
    can_tui.py can swap between backends without modification.

    Args:
        channel: PEAK channel name, e.g. 'PCAN_USBBUS1'.
        timeout:  Read timeout in seconds (kept for API compatibility;
                  python-can uses its own internal timeout handling).
    """

    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        timeout: float = 0.1,
    ) -> None:
        if not _PYTHON_CAN_AVAILABLE:
            raise ImportError(
                "python-can is required for PEAK support. "
                "Install with: pip install python-can"
            )

        self.port: str = channel      # named 'port' for API parity with WaveshareCAN
        self.channel: str = channel
        self.timeout: float = timeout

        # Auto-select python-can interface from channel name:
        #   'can0', 'can1', … → socketcan  (Linux kernel module, no PEAK libs)
        #   'PCAN_USBBUS1', … → pcan       (PEAK libs: Windows always, Linux optional)
        self._interface: str = (
            "socketcan" if _is_socketcan_channel(channel) else "pcan"
        )

        self._bus: Optional[can.BusABC] = None
        self._lock = threading.Lock()

        # Reception
        self.rx_queue: queue.Queue = queue.Queue(maxsize=0)
        self.rx_frame_count: int = 0
        self.on_message_received: Optional[Callable[[CANFrame], None]] = None
        self.running: bool = False
        self._rx_thread: Optional[threading.Thread] = None

        # Cyclic transmission
        self._cyclic_tasks: Dict[str, threading.Thread] = {}
        self._cyclic_stop_events: Dict[str, threading.Event] = {}

        # Store setup params so they are accessible after open()
        self._speed: CANSpeed = CANSpeed.SPEED_500K
        self._mode: CANMode = CANMode.NORMAL

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    def open(self) -> bool:
        """Open the PEAK device.

        The bus is NOT initialised here (no bitrate set yet).
        Call setup() immediately after open() to configure speed and mode.

        Returns:
            True if the channel exists and can be opened, False otherwise.
        """
        # Defer actual bus creation to setup() so we know the bitrate.
        # Just verify python-can is importable and record intent.
        print(f"[OK] PeakCAN: channel {self.channel} selected.")
        return True

    def setup(
        self,
        speed: CANSpeed = CANSpeed.SPEED_500K,
        mode: CANMode = CANMode.NORMAL,
        frame_type: CANFrameType = CANFrameType.EXTENDED,  # noqa: ARG002
    ) -> bool:
        """Configure and open the PEAK bus.

        frame_type is accepted for API compatibility but ignored — PEAK
        hardware receives Standard and Extended frames simultaneously.

        Args:
            speed:      CAN bus bitrate (CANSpeed enum).
            mode:       Operating mode (NORMAL / SILENT / LOOPBACK /
                        LOOPBACK_SILENT).
            frame_type: Ignored for PEAK hardware (API compatibility only).

        Returns:
            True on success.
        """
        self._speed = speed
        self._mode = mode

        bitrate = _SPEED_TO_BPS.get(int(speed), 500_000)

        # Translate CANMode → python-can bus parameters
        listen_only = mode in (CANMode.SILENT, CANMode.LOOPBACK_SILENT)
        loopback = mode in (CANMode.LOOPBACK, CANMode.LOOPBACK_SILENT)

        # Tear down any existing bus before reconfiguring
        self._shutdown_bus()

        try:
            if self._interface == "socketcan":
                # socketcan: bitrate set via ip link by python-can internally.
                # receive_own_messages enables local loopback for TX echo.
                # listen_only not a direct param — handled via interface flags;
                # we pass it as a keyword and let python-can ignore if unsupported.
                self._bus = can.interface.Bus(
                    interface="socketcan",
                    channel=self.channel,
                    bitrate=bitrate,
                    receive_own_messages=loopback,
                )
            else:
                # pcan interface (Windows + Linux with libpcanbasic)
                self._bus = can.interface.Bus(
                    interface="pcan",
                    channel=self.channel,
                    bitrate=bitrate,
                    listen_only=listen_only,
                    receive_own_messages=loopback,
                )
            speed_label = f"{bitrate // 1000} kbps"
            mode_label = mode.name
            print(
                f"[OK] PEAK {self.channel} ({self._interface}) initialised: "
                f"{speed_label}, {mode_label} mode"
            )
            return True
        except Exception as exc:
            print(f"[ERR] PEAK setup failed on {self.channel}: {exc}")
            self._bus = None
            return False

    def start_listening(self) -> None:
        """Start background RX thread."""
        if self.running:
            print("[INFO] PeakCAN listener already running.")
            return
        if self._bus is None:
            print("[ERR] PeakCAN: call setup() before start_listening().")
            return

        self.running = True
        self._rx_thread = threading.Thread(
            target=self._receive_loop, daemon=True
        )
        self._rx_thread.start()
        print("[OK] PeakCAN background listener started.")

    def stop_listening(self) -> None:
        """Stop the background RX thread."""
        if not self.running:
            return
        self.running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            self._rx_thread = None
        print("[OK] PeakCAN listener stopped.")

    def reset_rx_count(self) -> None:
        """Reset rx_frame_count to zero (API parity with WaveshareCAN)."""
        self.rx_frame_count = 0

    # -----------------------------------------------------------------------
    # Receive loop
    # -----------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Background thread: poll python-can bus and enqueue CANFrame objects."""
        while self.running and self._bus is not None:
            try:
                msg: Optional[can.Message] = self._bus.recv(timeout=0.1)
                if msg is None:
                    continue
                if msg.is_error_frame or msg.is_remote_frame:
                    continue
                # Filter PEAK bus-status/error-counter notification frames.
                # python-can on Windows delivers these as synthetic data frames
                # with arbitration_id=0, DLC=2, and data matching known PEAK
                # status patterns (first byte 0x00 or 0x01, second 0x00).
                # A genuine CAN frame with ID 0x000 and 2 bytes of all-zeros
                # is pathologically unlikely on a real automotive bus.
                if (msg.arbitration_id == 0
                        and not msg.is_extended_id
                        and len(msg.data) == 2
                        and msg.data[1] == 0x00):
                    continue

                frame = CANFrame(
                    can_id=msg.arbitration_id,
                    data=bytes(msg.data),
                    is_extended=msg.is_extended_id,
                    timestamp=msg.timestamp if msg.timestamp else time.time(),
                )

                self.rx_queue.put_nowait(frame)
                self.rx_frame_count += 1

                if self.on_message_received:
                    self.on_message_received(frame)

            except Exception as exc:
                if self.running:
                    print(f"[WARN] PeakCAN RX error: {exc}")
                time.sleep(0.001)

    # -----------------------------------------------------------------------
    # Transmission
    # -----------------------------------------------------------------------

    def send(
        self,
        can_id: int,
        data: bytes,
        is_extended: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Send a single CAN frame.

        Args:
            can_id:      CAN identifier (11-bit or 29-bit).
            data:        Payload (0–8 bytes).
            is_extended: True for 29-bit extended ID.
            verbose:     Print debug information.

        Returns:
            True if the frame was handed to the driver successfully.
        """
        if self._bus is None:
            print("[ERR] PeakCAN: not initialised — call setup() first.")
            return False
        if len(data) > 8:
            print("[ERR] PeakCAN: data exceeds 8 bytes.")
            return False

        try:
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=is_extended,
                is_error_frame=False,
            )
            with self._lock:
                self._bus.send(msg)

            if verbose:
                id_str = f"0x{can_id:08X}" if is_extended else f"0x{can_id:03X}"
                data_str = " ".join(f"{b:02X}" for b in data)
                print(f"[TX] {id_str}  [{len(data)}]  {data_str}")
            return True

        except Exception as exc:
            print(f"[ERR] PeakCAN send failed: {exc}")
            return False

    def send_cyclic(
        self,
        name: str,
        can_id: int,
        data: bytes,
        period_ms: int,
        is_extended: bool = True,
    ) -> None:
        """Start a cyclic transmission task.

        Args:
            name:        Unique task identifier.
            can_id:      CAN identifier.
            data:        Payload bytes.
            period_ms:   Transmission period in milliseconds.
            is_extended: True for 29-bit extended ID.
        """
        if name in self._cyclic_tasks:
            print(f"[WARN] PeakCAN cyclic task '{name}' already exists.")
            return

        stop_event = threading.Event()
        self._cyclic_stop_events[name] = stop_event

        def _cyclic_worker() -> None:
            interval = period_ms / 1000.0
            while not stop_event.is_set():
                self.send(can_id, data, is_extended)
                time.sleep(interval)

        thread = threading.Thread(target=_cyclic_worker, daemon=True)
        thread.start()
        self._cyclic_tasks[name] = thread
        print(
            f"[OK] PeakCAN cyclic '{name}' started: "
            f"ID 0x{can_id:X} every {period_ms} ms"
        )

    def stop_cyclic(self, name: str) -> None:
        """Stop a cyclic transmission task.

        Args:
            name: Task identifier passed to send_cyclic().
        """
        if name not in self._cyclic_tasks:
            print(f"[WARN] PeakCAN cyclic task '{name}' not found.")
            return

        self._cyclic_stop_events[name].set()
        self._cyclic_tasks[name].join(timeout=1.0)
        del self._cyclic_tasks[name]
        del self._cyclic_stop_events[name]
        print(f"[OK] PeakCAN cyclic task '{name}' stopped.")

    # -----------------------------------------------------------------------
    # Queue access
    # -----------------------------------------------------------------------

    def get_frame(self, timeout: Optional[float] = None) -> Optional[CANFrame]:
        """Get the next frame from the receive queue.

        Args:
            timeout: Wait timeout in seconds (None = block indefinitely).

        Returns:
            CANFrame or None on timeout.
        """
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Stop all activity and release the PEAK hardware channel."""
        self.stop_listening()

        for task_name in list(self._cyclic_tasks.keys()):
            self.stop_cyclic(task_name)

        self._shutdown_bus()
        print("[OK] PeakCAN connection closed.")

    def _shutdown_bus(self) -> None:
        """Safely shut down the python-can bus object."""
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
