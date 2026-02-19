#!/usr/bin/env python3
# 2026-02-19 00:00 v1.0.2 - Correct CANSpeed/CANMode enums and full 20-byte setup command per Waveshare PDF spec
"""
Waveshare USB-CAN-A Dongle Interface Class
Professional CAN bus diagnostic tool with thread-safe operation
"""

import serial
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List
from enum import IntEnum


class CANSpeed(IntEnum):
    """CAN Bus Speed — protocol byte for Waveshare config command (PDF p.6, byte 3)."""
    SPEED_1M   = 0x01
    SPEED_800K = 0x02
    SPEED_500K = 0x03
    SPEED_400K = 0x04
    SPEED_250K = 0x05
    SPEED_200K = 0x06
    SPEED_125K = 0x07
    SPEED_100K = 0x08
    SPEED_50K  = 0x09
    SPEED_20K  = 0x0A
    SPEED_10K  = 0x0B
    SPEED_5K   = 0x0C


class CANMode(IntEnum):
    """CAN Bus Mode — protocol byte for config command (PDF p.6, byte 13)."""
    NORMAL          = 0x00
    SILENT          = 0x01
    LOOPBACK        = 0x02
    LOOPBACK_SILENT = 0x03


@dataclass
class CANFrame:
    """CAN Frame Data Structure"""
    can_id: int
    data: bytes
    is_extended: bool
    timestamp: float
    
    def __str__(self):
        id_str = f"0x{self.can_id:08X}" if self.is_extended else f"0x{self.can_id:03X}"
        data_str = ' '.join(f"{b:02X}" for b in self.data)
        ext_flag = "Ext" if self.is_extended else "Std"
        return f"[{ext_flag}] ID: {id_str} | DLC: {len(self.data)} | Data: {data_str}"


class WaveshareCAN:
    """
    Professional Waveshare USB-CAN-A Interface
    
    Features:
    - Thread-safe reception with background listener
    - Message queue for zero data loss
    - Cyclic transmission support
    - Both Standard (11-bit) and Extended (29-bit) ID support
    """
    
    # Protocol Constants
    HEADER_SYNC = 0xAA
    TAIL_BYTE = 0x55
    SETUP_HEADER = [0xAA, 0x55]
    BAUD_SERIAL = 2000000
    
    def __init__(self, port: str = "/dev/ttyUSB0", timeout: float = 0.1):
        """
        Initialize Waveshare CAN interface
        
        Args:
            port: Serial port device path
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.rx_queue = queue.Queue(maxsize=0)  # 0 = unlimited
        self.running = False
        self.rx_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Cyclic transmission support
        self._cyclic_tasks: Dict[str, threading.Thread] = {}
        self._cyclic_stop_events: Dict[str, threading.Event] = {}
        
        # Callback for received messages
        self.on_message_received: Optional[Callable[[CANFrame], None]] = None
        
    def open(self) -> bool:
        """
        Open serial connection to Waveshare dongle
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.ser = serial.Serial(
                self.port, 
                self.BAUD_SERIAL, 
                timeout=self.timeout
            )
            print(f"[OK] Connected to {self.port} at {self.BAUD_SERIAL} bps")
            return True
        except Exception as e:
            print(f"[ERR] Failed to open {self.port}: {e}")
            return False
    
    def setup(self, 
              speed: CANSpeed = CANSpeed.SPEED_500K,
              mode: CANMode = CANMode.NORMAL,
              extended: bool = True) -> bool:
        """
        Initialize Waveshare dongle with CAN bus parameters
        
        Args:
            speed: CAN bus speed (see CANSpeed enum)
            mode: CAN operating mode (see CANMode enum)
            extended: Enable extended frame support
            
        Returns:
            True if successful
        """
        if not self.ser or not self.ser.is_open:
            print("[ERR] Serial port not open")
            return False
        
        # Build 20-byte CAN configuration command per Waveshare protocol spec (PDF p.6):
        # Byte  0-1 : 0xAA 0x55  header
        # Byte  2   : 0x12       type = variable-length protocol
        # Byte  3   : speed      CANSpeed enum value
        # Byte  4   : frame_type 0x01=Standard only, 0x02=Extended (accept both)
        # Bytes 5-8 : filter_id  0x00000000 = pass all
        # Bytes 9-12: block_id   0x00000000 = block nothing
        # Byte  13  : can_mode   CANMode enum value
        # Byte  14  : 0x00       auto-retransmit enabled
        # Bytes 15-18: 0x00      backup
        # Byte  19  : checksum   sum(bytes[2..18]) & 0xFF
        setup_cmd = [
            0xAA, 0x55,        # header
            0x12,              # variable-length protocol
            int(speed),        # byte 3: CAN baud rate
            0x02,              # byte 4: Extended frame (accepts std + ext)
            0x00, 0x00, 0x00, 0x00,  # bytes 5-8:  filter ID (pass all)
            0x00, 0x00, 0x00, 0x00,  # bytes 9-12: block  ID (block nothing)
            int(mode),         # byte 13: CAN mode
            0x00,              # byte 14: auto-retransmit enabled
            0x00, 0x00, 0x00, 0x00,  # bytes 15-18: backup
        ]
        checksum = sum(setup_cmd[2:]) & 0xFF  # sum bytes 2..18
        setup_cmd.append(checksum)

        try:
            self.ser.write(bytes(setup_cmd))
            time.sleep(0.5)  # Give device time to configure
            
            speed_str = f"{speed.name.replace('SPEED_', '')}bps"
            mode_str = mode.name
            ext_str = "Extended" if extended else "Standard"
            print(f"[OK] Waveshare initialized: {speed_str}, {mode_str} mode, {ext_str}")
            return True
        except Exception as e:
            print(f"[ERR] Setup failed: {e}")
            return False
    
    def start_listening(self):
        """Start background thread for receiving CAN frames"""
        if self.running:
            print("[INFO] Listener already running")
            return
        
        self.running = True
        self.rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.rx_thread.start()
        print("[OK] Background listener started")
    
    def stop_listening(self):
        """Stop background receiver thread"""
        if not self.running:
            return
        
        self.running = False
        if self.rx_thread:
            self.rx_thread.join(timeout=2.0)
        print("[OK] Listener stopped")
    
    def _receive_loop(self):
        """
        Background thread: Parse incoming CAN frames
        
        Protocol:
        [0xAA] [Control] [ID bytes] [Data bytes] [0x55]
        
        Control byte format:
        - Bit 5 (0x20): Extended ID flag
        - Bits 0-3: Data Length Code (DLC)
        """
        while self.running and self.ser and self.ser.is_open:
            try:
                # Step 1: Sync on header byte 0xAA
                header = self.ser.read(1)
                if not header or header[0] != self.HEADER_SYNC:
                    continue
                
                # Step 2: Read control byte
                ctrl = self.ser.read(1)
                if not ctrl:
                    continue
                
                ctrl_byte = ctrl[0]
                is_extended = bool(ctrl_byte & 0x20)
                dlc = ctrl_byte & 0x0F
                
                # Step 3: Calculate remaining bytes
                # Extended: 4 bytes ID, Standard: 2 bytes ID
                id_len = 4 if is_extended else 2
                remaining = id_len + dlc + 1  # ID + Data + Tail
                
                payload = self.ser.read(remaining)
                if len(payload) < remaining:
                    continue
                
                # Step 4: Parse CAN ID (Little Endian)
                if is_extended:
                    can_id = (payload[0] | 
                             (payload[1] << 8) | 
                             (payload[2] << 16) | 
                             (payload[3] << 24))
                    data_start = 4
                else:
                    can_id = payload[0] | (payload[1] << 8)
                    data_start = 2
                
                # Step 5: Extract data
                can_data = payload[data_start:data_start + dlc]
                
                # Step 6: Create frame object
                frame = CANFrame(
                    can_id=can_id,
                    data=can_data,
                    is_extended=is_extended,
                    timestamp=time.time()
                )
                
                # Step 7: Queue frame and trigger callback
                try:
                    self.rx_queue.put_nowait(frame)
                except queue.Full:
                    print("[WARN] RX Queue full, dropping frame")
                
                if self.on_message_received:
                    self.on_message_received(frame)
                    
            except Exception as e:
                if self.running:
                    print(f"[WARN] RX Error: {e}")
                time.sleep(0.001)
    
    def send(self, can_id: int, data: bytes, is_extended: bool = True, verbose: bool = False) -> bool:
        """
        Send a CAN frame
        
        Args:
            can_id: CAN identifier (11-bit or 29-bit)
            data: Data bytes (0-8 bytes)
            is_extended: True for extended (29-bit) ID
            verbose: Print debug information
            
        Returns:
            True if sent successfully
        """
        if not self.ser or not self.ser.is_open:
            print("[ERR] Serial port not open")
            return False
        
        if len(data) > 8:
            print("[ERR] Data length exceeds 8 bytes")
            return False
        
        try:
            with self._lock:
                # Construct frame
                dlc = len(data)
                
                # Control byte: Bit 5 = Extended flag, Bits 0-3 = DLC
                # Note: Working code uses 0xC8 base, which is 0xC0 | 0x20 | dlc
                ctrl_byte = 0xC0 | dlc  # Base 0xC0 seems required
                if is_extended:
                    ctrl_byte |= 0x20  # Add extended flag
                
                # Build packet: [Header][Control][ID bytes][Data][Tail]
                packet = [self.HEADER_SYNC, ctrl_byte]
                
                # Add ID bytes (Little Endian)
                if is_extended:
                    packet.extend([
                        can_id & 0xFF,
                        (can_id >> 8) & 0xFF,
                        (can_id >> 16) & 0xFF,
                        (can_id >> 24) & 0xFF
                    ])
                else:
                    packet.extend([
                        can_id & 0xFF,
                        (can_id >> 8) & 0xFF
                    ])
                
                # Add data bytes
                packet.extend(data)
                
                # Add tail byte
                packet.append(self.TAIL_BYTE)
                
                if verbose:
                    print(f"[DEBUG] Raw packet: {' '.join(f'{b:02X}' for b in packet)}")
                
                # Send
                self.ser.write(bytes(packet))
                return True
                
        except Exception as e:
            print(f"[ERR] Send failed: {e}")
            return False
    
    def send_cyclic(self, 
                    name: str,
                    can_id: int, 
                    data: bytes, 
                    period_ms: int,
                    is_extended: bool = True):
        """
        Send a CAN frame cyclically at a fixed interval
        
        Args:
            name: Unique identifier for this cyclic task
            can_id: CAN identifier
            data: Data bytes
            period_ms: Transmission period in milliseconds
            is_extended: True for extended ID
        """
        if name in self._cyclic_tasks:
            print(f"[WARN] Cyclic task '{name}' already exists")
            return
        
        stop_event = threading.Event()
        self._cyclic_stop_events[name] = stop_event
        
        def cyclic_sender():
            while not stop_event.is_set():
                self.send(can_id, data, is_extended)
                time.sleep(period_ms / 1000.0)
        
        thread = threading.Thread(target=cyclic_sender, daemon=True)
        thread.start()
        self._cyclic_tasks[name] = thread
        
        print(f"[OK] Cyclic task '{name}' started: ID 0x{can_id:X} every {period_ms}ms")
    
    def stop_cyclic(self, name: str):
        """Stop a cyclic transmission task"""
        if name not in self._cyclic_tasks:
            print(f"[WARN] Cyclic task '{name}' not found")
            return
        
        self._cyclic_stop_events[name].set()
        self._cyclic_tasks[name].join(timeout=1.0)
        del self._cyclic_tasks[name]
        del self._cyclic_stop_events[name]
        print(f"[OK] Cyclic task '{name}' stopped")
    
    def get_frame(self, timeout: Optional[float] = None) -> Optional[CANFrame]:
        """
        Get a frame from the receive queue
        
        Args:
            timeout: Wait timeout in seconds (None = blocking)
            
        Returns:
            CANFrame object or None if timeout
        """
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        """Close the connection and clean up resources"""
        # Stop listener
        self.stop_listening()
        
        # Stop all cyclic tasks
        for name in list(self._cyclic_tasks.keys()):
            self.stop_cyclic(name)
        
        # Close serial port
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[OK] Connection closed")


# Example usage
if __name__ == "__main__":
    can = WaveshareCAN()
    
    if can.open():
        can.setup(speed=CANSpeed.SPEED_500K, mode=CANMode.NORMAL)
        can.start_listening()
        
        # Example: Send a test frame
        can.send(0x12345678, bytes([0xDE, 0xAD, 0xBE, 0xEF]), is_extended=True)
        
        # Example: Receive frames for 5 seconds
        print("\nListening for frames...")
        start = time.time()
        while time.time() - start < 5:
            frame = can.get_frame(timeout=0.1)
            if frame:
                print(frame)
        
        can.close()
