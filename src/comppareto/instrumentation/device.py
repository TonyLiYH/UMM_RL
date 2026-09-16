"""Device resolution with no silent fallback.

The pass/fail gate forbids silently substituting CPU when a different
device is requested. This layer has no ML-framework dependency (see
:mod:`comppareto.instrumentation.params`), so "device" here is a plain
string identifier the caller declares intent with; the only device this
implementation can actually execute on is ``"cpu"``. Requesting anything
else must raise rather than silently running on CPU anyway -- which is
exactly the failure mode this module exists to prevent.
"""

from __future__ import annotations

SUPPORTED_DEVICES = ("cpu",)


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested device cannot actually be honored."""


def resolve_device(requested: str) -> str:
    """Return ``requested`` if it can genuinely be honored, else raise.

    Never substitutes a different device for the one requested.
    """
    normalized = requested.strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise DeviceUnavailableError(
            f"device {requested!r} was requested but this instrumentation "
            f"layer can only execute on {SUPPORTED_DEVICES!r}; refusing to "
            "silently fall back to a different device"
        )
    return normalized
