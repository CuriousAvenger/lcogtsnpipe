"""Minimal pyraf stub for CLI help/argparse tests.

This allows importing modules that perform IRAF setup at import time,
without requiring a full IRAF installation in the host test environment.
"""


class _IrafStub:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return None


iraf = _IrafStub()
