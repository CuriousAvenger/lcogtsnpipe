"""Minimal astroquery.vizier stub for CLI help/argparse tests."""


class _VizierStub:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return None


Vizier = _VizierStub()
