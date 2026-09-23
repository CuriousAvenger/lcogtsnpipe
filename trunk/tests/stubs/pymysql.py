"""Minimal pymysql stub for CLI help/argparse tests.

Used only by subprocess CLI tests to avoid requiring a real MySQL driver.
"""


class Error(Exception):
    pass


class _Cursors:
    class DictCursor:
        pass


cursors = _Cursors()


class _DummyCursor:
    rowcount = 0

    def execute(self, *args, **kwargs):
        return 0

    def fetchall(self):
        return []

    def close(self):
        return None


class _DummyConnection:
    def cursor(self, *args, **kwargs):
        return _DummyCursor()

    def commit(self):
        return None

    def close(self):
        return None


def connect(*args, **kwargs):
    return _DummyConnection()
