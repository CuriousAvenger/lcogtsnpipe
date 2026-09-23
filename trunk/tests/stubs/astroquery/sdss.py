"""Minimal astroquery.sdss stub for CLI help/argparse tests."""


class _SDSSStub:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return None


SDSS = _SDSSStub()
