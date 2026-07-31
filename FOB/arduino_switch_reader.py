"""Non-blocking, fail-safe Arduino digital switch serial reader."""

from __future__ import annotations
import argparse

import re
import threading
import time
from typing import Optional

try:
    import serial
except ImportError:
    serial = None


class ArduinoSwitchSource:
    """Read an active-high switch state from simple Arduino serial messages.

    ASCII ``A`` is the device's ON event. Common 0/1, ON/OFF, HIGH/LOW,
    TRUE/FALSE and forms such as ``D2=1`` are also accepted.
    """

    _TOKEN = re.compile(
        r"(?:^|[^A-Z0-9])(?:D2|SWITCH|BUTTON|STATE)?\s*[:=]?\s*"
        r"(ON|OFF|HIGH|LOW|TRUE|FALSE|1|0)(?:[^A-Z0-9]|$)",
        re.IGNORECASE,
    )
    _TRUE = {"ON", "HIGH", "TRUE", "1"}

    def __init__(self, port="/dev/ttyUSB0", baud=9600, timeout=0.1,
                 pulse_hold_s=0.35, reconnect_after_s=1.0):
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self.pulse_hold_s = float(pulse_hold_s)
        self.reconnect_after_s = float(reconnect_after_s)
        self._active = False
        self._active_until = 0.0
        self._sample_time = 0.0
        self._sample_count = 0
        self._connected = False
        self._error = "not started"
        self._serial = None
        self._buffer = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="arduino-switch-reader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.timeout * 5.0))
        self._close_serial()

    close = stop

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            active = self._active and now <= self._active_until
            sample_time = self._sample_time
            count = self._sample_count
            connected = self._connected
            error = self._error
        if not connected:
            return False, count, False, f"connection failed: {error}"
        if not sample_time:
            return False, count, True, "connected / waiting for valid data"
        return active, count, True, "connected"

    @property
    def active(self) -> bool:
        return self.snapshot()[0]

    @property
    def status(self) -> str:
        active, _count, connected, state = self.snapshot()
        if connected and _count == 0:
            return state
        return f"connected / {'ON' if active else 'OFF'}" if connected else state

    @classmethod
    def parse_state(cls, text: str):
        normalized = text.strip().upper()
        if normalized and set(normalized) == {"A"}:
            return True
        if normalized and set(normalized) <= {"0", "1"}:
            return normalized[-1] == "1"
        matches = list(cls._TOKEN.finditer(normalized))
        if not matches:
            return None
        return matches[-1].group(1).upper() in cls._TRUE

    def _publish(self, active: bool) -> None:
        with self._lock:
            self._active = bool(active)
            self._sample_time = time.monotonic()
            self._active_until = (self._sample_time + self.pulse_hold_s
                                  if active else 0.0)
            self._sample_count += 1
            self._connected = True
            self._error = ""

    def _set_error(self, error: str) -> None:
        with self._lock:
            self._active = False
            self._active_until = 0.0
            self._connected = False
            self._error = error

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _run(self) -> None:
        if serial is None:
            self._set_error("pyserial is not installed")
            return
        while not self._stop.is_set():
            try:
                if self._serial is None:
                    self._serial = serial.Serial(
                        self.port, self.baud, timeout=self.timeout,
                        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                    )
                    self._serial.reset_input_buffer()
                    with self._lock:
                        self._connected = True
                        self._error = "waiting for valid data"
                        self._active = False
                chunk = self._serial.read(max(1, self._serial.in_waiting or 1))
                if not chunk:
                    continue
                decoded_chunk = chunk.decode("ascii", errors="ignore")
                stripped_chunk = decoded_chunk.strip().upper()
                if stripped_chunk and set(stripped_chunk) == {"A"}:
                    self._publish(True)
                    self._buffer = ""
                    continue
                self._buffer = (self._buffer + decoded_chunk)[-256:]
                parts = re.split(r"[\r\n,;]+", self._buffer)
                self._buffer = parts.pop() if parts else ""
                candidates = parts + ([self._buffer] if self._buffer else [])
                for candidate in candidates:
                    state = self.parse_state(candidate)
                    if state is not None:
                        self._publish(state)
            except Exception as exc:
                self._set_error(str(exc))
                self._close_serial()
                self._buffer = ""
                self._stop.wait(self.reconnect_after_s)


def _raw_state(chunk: bytes):
    """Interpret printable messages and binary 0x00/0x01 event bytes."""
    if chunk and all(value in (0, 1) for value in chunk):
        return bool(chunk[-1])
    return ArduinoSwitchSource.parse_state(chunk.decode("ascii", errors="ignore"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor raw Arduino switch bytes and parsed ON/OFF events"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument(
        "--startup-delay", type=float, default=2.0,
        help="seconds to wait after opening a Nano that resets on serial connect",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="stop after N seconds; 0 listens until Ctrl+C",
    )
    args = parser.parse_args()
    if args.baud <= 0 or args.timeout <= 0 or args.startup_delay < 0 or args.duration < 0:
        parser.error("baud/timeout must be positive; delays must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if serial is None:
        print("[ERROR] pyserial is not installed")
        return 2
    print(f"[OPEN] {args.port} @ {args.baud} baud")
    print("[INFO] Press the switch repeatedly. Every received byte will be shown.")
    print("[INFO] Stop with Ctrl+C.")
    try:
        with serial.Serial(
            args.port, args.baud, timeout=args.timeout,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        ) as device:
            if args.startup_delay:
                print(f"[WAIT] Arduino reset delay {args.startup_delay:.1f}s")
                time.sleep(args.startup_delay)
            device.reset_input_buffer()
            started = time.monotonic()
            received = 0
            while not args.duration or time.monotonic() - started < args.duration:
                chunk = device.read(max(1, device.in_waiting or 1))
                if not chunk:
                    continue
                received += len(chunk)
                elapsed = time.monotonic() - started
                decoded = chunk.decode("ascii", errors="backslashreplace")
                state = _raw_state(chunk)
                parsed = "ON" if state is True else "OFF" if state is False else "UNKNOWN"
                print(
                    f"[{elapsed:9.3f}s] bytes={len(chunk):3d} hex={chunk.hex(' ')} "
                    f"ascii={decoded!r} parsed={parsed}", flush=True,
                )
            print(f"[DONE] received {received} byte(s)")
    except KeyboardInterrupt:
        print("\n[STOP] user interrupted")
    except Exception as exc:
        print(f"[ERROR] connection failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
