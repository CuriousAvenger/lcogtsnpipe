"""Minimal astroquery.gaia stub for CLI help/argparse tests."""


class _GaiaStub:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return None


Gaia = _GaiaStub()
