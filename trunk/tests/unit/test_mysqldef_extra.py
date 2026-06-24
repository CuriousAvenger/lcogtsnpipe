"""
Additional tests for lsc.mysqldef — covers get_snex_uid and JDnow/MJDnow edge cases.
No database access required (mocked).
"""
import pytest
import datetime
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# get_snex_uid — resolves UNIX username to SNEx user ID
# ---------------------------------------------------------------------------

class TestGetSnexUid:
    def test_returns_uid_when_user_found(self, monkeypatch):
        """When the UNIX username matches a SNEx user, returns the id."""
        from lsc.mysqldef import get_snex_uid

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'id': 42, 'firstname': 'Jane', 'lastname': 'Doe'}
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch('lsc.mysqldef.query', return_value=[
            {'id': 42, 'firstname': 'Jane', 'lastname': 'Doe'}
        ]):
            with patch('lsc.mysqldef.lsc') as mock_lsc:
                mock_lsc.myloopdef.conn = mock_conn
                result = get_snex_uid(interactive=False)
                assert result == 42

    def test_returns_fullname_when_requested(self, monkeypatch):
        """When return_fullname=True, returns (uid, fullname)."""
        from lsc.mysqldef import get_snex_uid

        mock_conn = MagicMock()
        with patch('lsc.mysqldef.query', return_value=[
            {'id': 42, 'firstname': 'Jane', 'lastname': 'Doe'}
        ]):
            with patch('lsc.mysqldef.lsc') as mock_lsc:
                mock_lsc.myloopdef.conn = mock_conn
                uid, fullname = get_snex_uid(interactive=False, return_fullname=True)
                assert uid == 42
                assert fullname == 'Jane Doe'

    def test_returns_none_when_user_not_found(self, monkeypatch):
        """When no matching user, returns None."""
        from lsc.mysqldef import get_snex_uid

        mock_conn = MagicMock()
        with patch('lsc.mysqldef.query', return_value=[]):
            with patch('lsc.mysqldef.lsc') as mock_lsc:
                mock_lsc.myloopdef.conn = mock_conn
                result = get_snex_uid(interactive=False)
                assert result is None

    def test_interactive_prompts_username(self, monkeypatch):
        """When interactive=True and user not found, prompts for username."""
        from lsc.mysqldef import get_snex_uid

        mock_conn = MagicMock()
        # First call returns empty (user not found), second call returns user
        with patch('lsc.mysqldef.query', side_effect=[
            [],  # Initial lookup fails
            [{'id': 99, 'firstname': 'Test', 'lastname': 'User'}]  # Second lookup succeeds
        ]):
            with patch('lsc.mysqldef.lsc') as mock_lsc:
                mock_lsc.myloopdef.conn = mock_conn
                with patch('lsc.mysqldef.lsc.util.userinput', return_value='testuser'):
                    result = get_snex_uid(interactive=True)
                    assert result == 99


# ---------------------------------------------------------------------------
# JDnow / MJDnow — Julian date calculations
# ---------------------------------------------------------------------------

class TestJDnow:
    def test_returns_float(self):
        from lsc.mysqldef import JDnow
        result = JDnow()
        assert isinstance(result, float)

    def test_known_date(self):
        """JD for 2000-01-01 12:00:00 UT is 2451545.0."""
        from lsc.mysqldef import JDnow
        dt = datetime.datetime(2000, 1, 1, 12, 0, 0)
        result = JDnow(datenow=dt)
        assert abs(result - 2451545.0) < 0.5

    def test_verbose_prints(self, capsys):
        from lsc.mysqldef import JDnow
        JDnow(verbose=True)
        captured = capsys.readouterr()
        assert 'JD' in captured.out


class TestMJDnow:
    def test_returns_float(self):
        from lsc.mysqldef import MJDnow
        result = MJDnow()
        assert isinstance(result, float)

    def test_known_date(self):
        """MJD for 2000-01-01 12:00:00 UT is 51544.5."""
        from lsc.mysqldef import MJDnow
        dt = datetime.datetime(2000, 1, 1, 12, 0, 0)
        result = MJDnow(datenow=dt)
        assert abs(result - 51544.5) < 0.5

    def test_jd_mjd_offset(self):
        """JD - MJD should be approximately 2400000.5."""
        from lsc.mysqldef import JDnow, MJDnow
        dt = datetime.datetime(2020, 6, 15, 0, 0, 0)
        jd = JDnow(datenow=dt)
        mjd = MJDnow(datenow=dt)
        assert abs((jd - mjd) - 2400000.5) < 0.01
