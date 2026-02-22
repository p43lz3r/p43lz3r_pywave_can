#!/usr/bin/env python3
# 2026-02-12 14:30 v1.0.0
"""
Professional CAN Bus Diagnostic Tool
Features: Fixed-view display, DBC decoding, cyclic transmission, filtering
"""

import sys
import time
import argparse
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime

# Import our CAN interface
from waveshare_can import WaveshareCAN, CANFrame, CANSpeed, CANMode

# Optional: cantools for DBC file support
try:
    import cantools
    CANTOOLS_AVAILABLE = True
except ImportError:
    CANTOOLS_AVAILABLE = False
    print("âš  cantools not installed. DBC decoding disabled.")
    print("  Install with: pip install cantools")


class CANStatistics:
    """Track CAN bus statistics"""
    def __init__(self):
        self.total_frames = 0
        self.frames_per_id: Dict[int, int] = defaultdict(int)
        self.last_seen: Dict[int, float] = {}
        self.start_time = time.time()
    
    def update(self, frame: CANFrame):
        self.total_frames += 1
        self.frames_per_id[frame.can_id] += 1
        self.last_seen[frame.can_id] = frame.timestamp
    
    def get_rate(self, can_id: int) -> float:
        """Calculate message rate in Hz"""
        count = self.frames_per_id[can_id]
        elapsed = time.time() - self.start_time
        return count / elapsed if elapsed > 0 else 0.0


class FixedViewDisplay:
    """
    Fixed-view display for CAN messages
    Shows a static table that updates in place (no scrolling)
    """
    def __init__(self, max_ids: int = 20):
        self.max_ids = max_ids
        self.frame_cache: Dict[int, CANFrame] = {}
        self.stats = CANStatistics()
        self.db: Optional[cantools.database.Database] = None
    
    def load_dbc(self, dbc_path: str) -> bool:
        """Load a DBC file for signal decoding"""
        if not CANTOOLS_AVAILABLE:
            print("âœ— cantools not available")
            return False
        
        try:
            self.db = cantools.database.load_file(dbc_path)
            print(f"âœ“ Loaded DBC: {dbc_path} ({len(self.db.messages)} messages)")
            return True
        except Exception as e:
            print(f"âœ— Failed to load DBC: {e}")
            return False
    
    def update(self, frame: CANFrame):
        """Update display with new frame"""
        self.frame_cache[frame.can_id] = frame
        self.stats.update(frame)
        
        # Limit cache size
        if len(self.frame_cache) > self.max_ids:
            # Remove oldest
            oldest_id = min(self.frame_cache.keys(), 
                          key=lambda k: self.frame_cache[k].timestamp)
            del self.frame_cache[oldest_id]
    
    def decode_signals(self, frame: CANFrame) -> str:
        """Decode signals using DBC file"""
        if not self.db:
            return ""
        
        try:
            message = self.db.get_message_by_frame_id(frame.can_id)
            decoded = message.decode(frame.data)
            
            # Format signals
            signals = []
            for name, value in decoded.items():
                # Format based on type
                if isinstance(value, float):
                    signals.append(f"{name}={value:.2f}")
                else:
                    signals.append(f"{name}={value}")
            
            return " | ".join(signals)
        except:
            return ""
    
    def render(self):
        """Render the fixed display"""
        # Clear screen (ANSI escape codes)
        print("\033[2J\033[H", end="")
        
        # Header
        print("=" * 100)
        print(f"CAN Bus Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Frames: {self.stats.total_frames} | "
              f"Unique IDs: {len(self.frame_cache)}")
        print("=" * 100)
        print()
        
        # Column headers
        print(f"{'ID':>12} {'Type':>6} {'DLC':>4} {'Rate (Hz)':>10} "
              f"{'Data':>24} {'Signals/Decoded':<40}")
        print("-" * 100)
        
        # Sort by ID
        sorted_ids = sorted(self.frame_cache.keys())
        
        for can_id in sorted_ids[:self.max_ids]:
            frame = self.frame_cache[can_id]
            
            # Format ID
            if frame.is_extended:
                id_str = f"0x{can_id:08X}"
                type_str = "Ext"
            else:
                id_str = f"0x{can_id:03X}"
                type_str = "Std"
            
            # Format data
            data_hex = ' '.join(f"{b:02X}" for b in frame.data)
            
            # Calculate rate
            rate = self.stats.get_rate(can_id)
            
            # Decode signals if DBC available
            decoded = self.decode_signals(frame)
            
            print(f"{id_str:>12} {type_str:>6} {len(frame.data):>4} "
                  f"{rate:>10.1f} {data_hex:>24} {decoded:<40}")
        
        print()
        print("Press Ctrl+C to stop")


class CANDiagnosticTool:
    """Main diagnostic tool application"""
    
    def __init__(self, port: str = "/dev/ttyUSB0"):
        self.can = WaveshareCAN(port=port)
        self.display = FixedViewDisplay(max_ids=20)
        self.running = False
        
        # Register callback
        self.can.on_message_received = self._on_message
    
    def _on_message(self, frame: CANFrame):
        """Callback for received messages"""
        self.display.update(frame)
    
    def start(self, 
              speed: CANSpeed = CANSpeed.SPEED_500K,
              mode: CANMode = CANMode.NORMAL,
              dbc_file: Optional[str] = None):
        """Start the diagnostic tool"""
        
        # Open connection
        if not self.can.open():
            return False
        
        # Setup CAN parameters
        if not self.can.setup(speed=speed, mode=mode):
            return False
        
        # Load DBC if provided
        if dbc_file:
            self.display.load_dbc(dbc_file)
        
        # Start listening
        self.can.start_listening()
        
        print("\nâœ“ Diagnostic tool started")
        print("  Press Ctrl+C to stop\n")
        time.sleep(1)
        
        self.running = True
        return True
    
    def run_monitor(self, refresh_rate: float = 0.2):
        """Run in monitor mode with fixed display"""
        try:
            while self.running:
                self.display.render()
                time.sleep(refresh_rate)
        except KeyboardInterrupt:
            print("\n\nâœ“ Stopping...")
        finally:
            self.stop()
    
    def run_logger(self, log_file: Optional[str] = None):
        """Run in logging mode (scrolling output)"""
        print("\n--- CAN Bus Logger Mode ---")
        print("Press Ctrl+C to stop\n")
        
        file_handle = None
        if log_file:
            try:
                file_handle = open(log_file, 'w')
                print(f"âœ“ Logging to {log_file}\n")
            except Exception as e:
                print(f"âœ— Failed to open log file: {e}")
        
        try:
            while self.running:
                frame = self.can.get_frame(timeout=0.1)
                if frame:
                    # Console output
                    print(frame)
                    
                    # File output
                    if file_handle:
                        timestamp = datetime.fromtimestamp(frame.timestamp)
                        file_handle.write(
                            f"{timestamp.isoformat()},{frame.can_id:08X},"
                            f"{frame.data.hex()},{frame.is_extended}\n"
                        )
                        file_handle.flush()
        
        except KeyboardInterrupt:
            print("\nâœ“ Stopping...")
        finally:
            if file_handle:
                file_handle.close()
                print(f"âœ“ Log saved to {log_file}")
            self.stop()
    
    def send_message(self, can_id: int, data: bytes, is_extended: bool = True):
        """Send a single CAN message"""
        if self.can.send(can_id, data, is_extended, verbose=True):
            print(f"âœ“ Sent: ID=0x{can_id:X}, Data={data.hex(' ').upper()}")
        else:
            print(f"âœ— Failed to send message")
    
    def send_cyclic_message(self, 
                           name: str,
                           can_id: int, 
                           data: bytes, 
                           period_ms: int,
                           is_extended: bool = True):
        """Start cyclic transmission"""
        self.can.send_cyclic(name, can_id, data, period_ms, is_extended)
    
    def stop_cyclic_message(self, name: str):
        """Stop cyclic transmission"""
        self.can.stop_cyclic(name)
    
    def stop(self):
        """Stop the tool"""
        self.running = False
        self.can.close()


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Professional CAN Bus Diagnostic Tool for Waveshare USB-CAN-A"
    )
    
    parser.add_argument(
        '--port', '-p',
        default='/dev/ttyUSB0',
        help='Serial port (default: /dev/ttyUSB0)'
    )
    
    parser.add_argument(
        '--speed', '-s',
        type=int,
        default=500,
        choices=[5, 10, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 666, 800, 1000],
        help='CAN bus speed in kbps (default: 500)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['normal', 'loopback', 'silent', 'loopback-silent'],
        default='normal',
        help='CAN operating mode (default: normal)'
    )
    
    parser.add_argument(
        '--dbc',
        help='DBC file for signal decoding'
    )
    
    parser.add_argument(
        '--log',
        help='Log file for recording CAN traffic'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Use fixed-view monitor mode (default: logger mode)'
    )
    
    parser.add_argument(
        '--send',
        nargs=2,
        metavar=('ID', 'DATA'),
        help='Send single message (e.g., --send 0x123 "DE AD BE EF")'
    )
    
    args = parser.parse_args()
    
    # Map speed to enum
    speed_map = {
        5: CANSpeed.SPEED_5K, 10: CANSpeed.SPEED_10K, 20: CANSpeed.SPEED_20K,
        25: CANSpeed.SPEED_25K, 40: CANSpeed.SPEED_40K, 50: CANSpeed.SPEED_50K,
        80: CANSpeed.SPEED_80K, 100: CANSpeed.SPEED_100K, 125: CANSpeed.SPEED_125K,
        200: CANSpeed.SPEED_200K, 250: CANSpeed.SPEED_250K, 400: CANSpeed.SPEED_400K,
        500: CANSpeed.SPEED_500K, 666: CANSpeed.SPEED_666K, 800: CANSpeed.SPEED_800K,
        1000: CANSpeed.SPEED_1M
    }
    
    # Map mode to enum
    mode_map = {
        'normal': CANMode.NORMAL,
        'loopback': CANMode.LOOPBACK,
        'silent': CANMode.SILENT,
        'loopback-silent': CANMode.LOOPBACK_SILENT
    }
    
    # Create tool
    tool = CANDiagnosticTool(port=args.port)
    
    # Handle send-only mode
    if args.send:
        can_id_str, data_str = args.send
        can_id = int(can_id_str, 0)  # Auto-detect hex/decimal
        data = bytes.fromhex(data_str.replace(' ', ''))
        is_extended = can_id > 0x7FF
        
        print(f"\n--- CAN Message Transmit ---")
        print(f"ID: 0x{can_id:X} ({'Extended' if is_extended else 'Standard'})")
        print(f"Data: {data.hex(' ').upper()} ({len(data)} bytes)")
        print(f"Speed: {args.speed}kbps")
        print(f"Mode: {args.mode}")
        print("-" * 40)
        
        if tool.start(speed=speed_map[args.speed], mode=mode_map[args.mode]):
            time.sleep(0.2)  # Give setup time to complete
            tool.send_message(can_id, data, is_extended)
            time.sleep(0.2)  # Give message time to transmit
            tool.stop()
        else:
            print("\nâœ— Failed to start CAN interface")
            sys.exit(1)
        return
    
    # Start tool
    if not tool.start(
        speed=speed_map[args.speed], 
        mode=mode_map[args.mode],
        dbc_file=args.dbc
    ):
        sys.exit(1)
    
    # Run in selected mode
    if args.monitor:
        tool.run_monitor()
    else:
        tool.run_logger(log_file=args.log)


if __name__ == "__main__":
    main()
