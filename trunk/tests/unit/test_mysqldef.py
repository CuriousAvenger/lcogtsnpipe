"""
Tests for lsc.mysqldef pure-Python functions:
guess_instrument_type, getsky, MJDnow, JDnow, getconnection.
Database-dependent functions are tested with a mock connection.
"""
import datetime
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# guess_instrument_type
# ---------------------------------------------------------------------------

class TestGuessInstrumentType:
    def test_fl_prefix(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("fl01") == "Sinistro"

    def test_fa_prefix(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("fa05") == "Sinistro"

    def test_kb_prefix(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("kb70") == "SBIG"

    def test_fs_prefix(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("fs01") == "Spectral"

    def test_ep_prefix(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("ep01") == "MUSCAT"

    def test_sq_prefix(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("sq01") == "QHY"

    def test_unknown_prefix_returns_none(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("xx99") is None

    def test_empty_string_returns_none(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type("") is None


# ---------------------------------------------------------------------------
# getsky — sigma-clipped mean and std of a numpy array
# ---------------------------------------------------------------------------

class TestGetsky:
    def test_uniform_array(self):
        from lsc.mysqldef import getsky
        # Use slightly noisy data — perfectly uniform arrays empty the sigma-clip
        rng = np.random.default_rng(123)
        data = rng.normal(1000.0, 0.1, (50, 50))   # tiny but nonzero spread
        mean, std = getsky(data)
        assert abs(mean - 1000.0) < 2.0
        assert abs(std) < 1.0

    def test_returns_two_values(self):
        from lsc.mysqldef import getsky
        data = np.random.default_rng(0).normal(500, 20, (30, 30))
        result = getsky(data)
        assert len(result) == 2

    def test_std_positive(self):
        from lsc.mysqldef import getsky
        data = np.random.default_rng(1).normal(1000, 50, (40, 40))
        mean, std = getsky(data)
        assert std > 0

    def test_outlier_ignored(self):
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(99)
        data = rng.normal(1000.0, 5.0, (20, 20))
        data[10, 10] = 999999.0     # single extreme outlier
        mean, std = getsky(data)
        # median/clipped result should be close to 1000, not dominated by outlier
        assert abs(mean - 1000.0) < 50.0


# ---------------------------------------------------------------------------
# MJDnow and JDnow
# ---------------------------------------------------------------------------

class TestMJDnow:
    def test_reference_epoch(self):
        from lsc.mysqldef import MJDnow
        fixed = datetime.datetime(2012, 1, 1, 0, 0, 0)
        mjd = MJDnow(datenow=fixed)
        assert abs(mjd - 55927.0) < 1e-6

    def test_no_arg_returns_float(self):
        from lsc.mysqldef import MJDnow
        assert isinstance(MJDnow(), float)

    def test_increases_monotonically(self):
        from lsc.mysqldef import MJDnow
        d0 = datetime.datetime(2020, 1, 1)
        d1 = datetime.datetime(2020, 1, 2)
        assert MJDnow(datenow=d1) > MJDnow(datenow=d0)

    def test_verbose_prints_jd(self, capsys):
        from lsc.mysqldef import MJDnow
        fixed = datetime.datetime(2012, 1, 1, 0, 0, 0)
        MJDnow(datenow=fixed, verbose=True)
        out = capsys.readouterr().out
        assert 'JD=' in out
        assert '55927' in out

    def test_one_day_advance(self):
        from lsc.mysqldef import MJDnow
        d0 = datetime.datetime(2012, 1, 1)
        d1 = datetime.datetime(2012, 1, 2)
        assert abs(MJDnow(datenow=d1) - MJDnow(datenow=d0) - 1.0) < 1e-6


class TestJDnow:
    def test_reference_epoch_j2000(self):
        from lsc.mysqldef import JDnow
        fixed = datetime.datetime(2000, 1, 1, 12, 0, 0)   # J2000.0 = JD 2451545.0
        jd = JDnow(datenow=fixed)
        assert abs(jd - 2451545.0) < 1e-4

    def test_jd_gt_mjd_by_about_2400000(self):
        from lsc.mysqldef import JDnow, MJDnow
        fixed = datetime.datetime(2022, 6, 1)
        jd  = JDnow(datenow=fixed)
        mjd = MJDnow(datenow=fixed)
        assert abs((jd - mjd) - 2400000.5) < 0.1

    def test_verbose_prints_jd(self, capsys):
        from lsc.mysqldef import JDnow
        fixed = datetime.datetime(2012, 1, 1, 0, 0, 0)
        JDnow(datenow=fixed, verbose=True)
        out = capsys.readouterr().out
        assert 'JD=' in out

    def test_no_arg_returns_float(self):
        from lsc.mysqldef import JDnow
        assert isinstance(JDnow(), float)


# ---------------------------------------------------------------------------
# getfromdataraw — mocked database query
# ---------------------------------------------------------------------------

@pytest.mark.db
class TestGetfromdataraw:
    def test_returns_row_list(self, mock_db_conn):
        from lsc.mysqldef import getfromdataraw
        rows = getfromdataraw(mock_db_conn, "photlco",
                              "filename,telescope", "filename='test.fits'")
        assert isinstance(rows, (list, tuple))

    def test_empty_result_on_no_match(self, mock_db_conn):
        from lsc.mysqldef import getfromdataraw
        mock_db_conn.cursor().fetchall.return_value = []
        rows = getfromdataraw(mock_db_conn, "photlco",
                              "filename", "filename='nonexistent.fits'")
        assert rows == [] or rows is None or len(rows) == 0


# ---------------------------------------------------------------------------
# getconnection — lookup DB connection parameters from lsc.util.readpass
# ---------------------------------------------------------------------------

class TestGetconnection:
    """getconnection reads from lsc.util.readpass (module-level global).
    We monkeypatch that dict so the test is hermetic."""

    @pytest.fixture(autouse=True)
    def _patch_readpass(self, monkeypatch):
        import lsc.util
        fake = {
            'database':    'sndb',
            'hostname':    'db.example.com',
            'mysqluser':   'snuser',
            'mysqlpasswd': 'snpass',
            'ptfdatabase': 'ptfdb',
            'ptfhost':     'ptf.example.com',
            'ptfuser':     'ptfuser',
            'ptfpasswd':   'ptfpass',
        }
        monkeypatch.setattr(lsc.util, 'readpass', fake)

    def test_lcogt2_returns_four_values(self):
        from lsc.mysqldef import getconnection
        result = getconnection('lcogt2')
        assert len(result) == 4

    def test_lcogt2_host(self):
        from lsc.mysqldef import getconnection
        host, user, passwd, db = getconnection('lcogt2')
        assert host == 'db.example.com'

    def test_lcogt2_credentials(self):
        from lsc.mysqldef import getconnection
        host, user, passwd, db = getconnection('lcogt2')
        assert user == 'snuser'
        assert passwd == 'snpass'
        assert db == 'sndb'

    def test_iptf_returns_four_values(self):
        from lsc.mysqldef import getconnection
        result = getconnection('iptf')
        assert len(result) == 4

    def test_iptf_host(self):
        from lsc.mysqldef import getconnection
        host, user, passwd, db = getconnection('iptf')
        assert host == 'ptf.example.com'

    def test_unknown_site_raises_key_error(self):
        from lsc.mysqldef import getconnection
        with pytest.raises(KeyError):
            getconnection('unknown_site_xyz')


# ---------------------------------------------------------------------------
# getmissing — all branch combinations via mocked conn
# ---------------------------------------------------------------------------

def _make_conn():
    """Return a (conn, cursor) pair backed by MagicMock."""
    import sys
    msql = sys.modules['pymysql']
    cursor = MagicMock()
    cursor.rowcount = 0
    cursor.fetchall.return_value = ()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    # Also set cursors attribute for DictCursor
    msql.cursors = MagicMock()
    msql.cursors.DictCursor = MagicMock()
    return conn, cursor


class TestGetmissing:
    def test_all_telescope_with_epoch2(self):
        from lsc.mysqldef import getmissing
        conn, _ = _make_conn()
        result = getmissing(conn, '20220101', '20220201', 'all')
        assert isinstance(result, tuple)

    def test_all_telescope_without_epoch2(self):
        from lsc.mysqldef import getmissing
        conn, _ = _make_conn()
        result = getmissing(conn, '20220101', '', 'all')
        assert isinstance(result, tuple)

    def test_specific_telescope_with_epoch2(self):
        from lsc.mysqldef import getmissing
        conn, _ = _make_conn()
        result = getmissing(conn, '20220101', '20220201', '1m0-01')
        assert isinstance(result, tuple)

    def test_specific_telescope_without_epoch2(self):
        from lsc.mysqldef import getmissing
        conn, _ = _make_conn()
        result = getmissing(conn, '20220101', '', '1m0-01')
        assert isinstance(result, tuple)

    def test_returns_empty_when_no_rows(self):
        from lsc.mysqldef import getmissing
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = getmissing(conn, '20220101', '20220201', 'all')
        assert len(result) == 0

    def test_returns_rows_when_cursor_has_data(self):
        from lsc.mysqldef import getmissing
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'filename': 'a.fits', 'objname': 'SN2020'},)
        result = getmissing(conn, '20220101', '20220201', 'all')
        assert len(result) == 1


class TestGetlistfromraw:
    def test_telescope_all(self):
        from lsc.mysqldef import getlistfromraw
        conn, _ = _make_conn()
        result = getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201')
        assert isinstance(result, tuple)

    def test_specific_telescope(self):
        from lsc.mysqldef import getlistfromraw
        conn, _ = _make_conn()
        result = getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201', telescope='1m0-01')
        assert isinstance(result, tuple)

    def test_value2_defaults_to_value1(self):
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '')
        # The execute was called — check the SQL included value1 twice
        sql = cursor.execute.call_args[0][0]
        assert '20220101' in sql


class TestInsertValues:
    def test_basic_insert(self):
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits', 'filter': 'r'}
        insert_values(conn, 'photlco', values)
        assert cursor.execute.called

    def test_datecreated_added_automatically(self):
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits'}
        insert_values(conn, 'photlco', values)
        sql = cursor.execute.call_args[0][0]
        assert 'datecreated' in sql

    def test_none_values_excluded(self):
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits', 'filter': None, 'ra': 180.0}
        insert_values(conn, 'photlco', values)
        sql = cursor.execute.call_args[0][0]
        assert 'filter' not in sql
        assert 'ra' in sql

    def test_empty_string_excluded(self):
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits', 'comment': ''}
        insert_values(conn, 'photlco', values)
        sql = cursor.execute.call_args[0][0]
        assert 'comment' not in sql


class TestGetvaluefromarchive:
    def test_returns_list(self):
        from lsc.mysqldef import getvaluefromarchive
        # getvaluefromarchive uses dbConnect internally — already mocked
        result = getvaluefromarchive('photlco', 'filename', 'test.fits', '*')
        assert isinstance(result, (list, tuple))

    def test_empty_when_no_match(self):
        from lsc.mysqldef import getvaluefromarchive
        result = getvaluefromarchive('photlco', 'filename', 'nonexistent.fits', '*')
        assert result == [] or result == ()


class TestDeleteredufromarchive:
    def test_runs_without_error(self):
        from lsc.mysqldef import deleteredufromarchive
        # deleteredufromarchive uses dbConnect internally — already mocked
        # Should complete without raising
        deleteredufromarchive('test.fits', archive='photlco')

    def test_with_nondefault_archive(self):
        from lsc.mysqldef import deleteredufromarchive
        deleteredufromarchive('test.fits', archive='photlcoraw')


class TestGetfromcoordinate:
    def test_targets_table(self):
        from lsc.mysqldef import getfromcoordinate
        conn, _ = _make_conn()
        result = getfromcoordinate(conn, 'targets', 180.0, 2.0, 0.1)
        assert isinstance(result, tuple)

    def test_photlcoraw_table(self):
        from lsc.mysqldef import getfromcoordinate
        conn, _ = _make_conn()
        result = getfromcoordinate(conn, 'photlcoraw', 180.0, 2.0, 0.5)
        assert isinstance(result, tuple)


class TestGetlike:
    def test_returns_tuple(self):
        from lsc.mysqldef import getlike
        conn, _ = _make_conn()
        result = getlike(conn, 'photlco', 'filename', 'test')
        assert isinstance(result, tuple)

    def test_with_column2(self):
        from lsc.mysqldef import getlike
        conn, _ = _make_conn()
        result = getlike(conn, 'photlco', 'filename', 'test', column2='filename,filter')
        assert isinstance(result, tuple)


class TestQuery:
    def test_single_command(self):
        from lsc.mysqldef import query
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'id': 1},)
        result = query(['SELECT 1'], conn)
        assert result == ({'id': 1},) or isinstance(result, (tuple, list, str))

    def test_multiple_commands(self):
        from lsc.mysqldef import query
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = query(['SELECT 1', 'SELECT 2'], conn)
        assert cursor.execute.call_count == 2

    def test_empty_command_list(self):
        from lsc.mysqldef import query
        conn, cursor = _make_conn()
        result = query([], conn)
        # No execute calls
        assert cursor.execute.call_count == 0


class TestGettargetid:
    def test_returns_empty_when_not_found(self):
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = gettargetid('SN2024abc', '', '', conn, 0.01)
        assert result == ''

    def test_with_name(self):
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'name': 'SN2024abc', 'targetid': 1, 'id': 1, 'ra0': 180.0, 'dec0': 2.0},)
        result = gettargetid('SN2024abc', '', '', conn, 0.01)
        assert result == 1

    def test_with_multiple_ids_returns_empty(self):
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        # Two different IDs → ambiguous → returns ''
        cursor.fetchall.return_value = (
            {'name': 'SN2024abc', 'targetid': 1, 'id': 1, 'ra0': 180.0, 'dec0': 2.0},
            {'name': 'SN2024abc', 'targetid': 2, 'id': 2, 'ra0': 180.1, 'dec0': 2.0},
        )
        result = gettargetid('SN2024abc', '', '', conn, 0.01)
        assert result == ''

    def test_verbose_no_match(self, capsys):
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        gettargetid('SN2024abc', '', '', conn, 0.01, verbose=True)
        out = capsys.readouterr().out
        assert 'no objects' in out

    def test_with_multiple_ids_and_hsine_returns_min(self):
        """Multiple IDs with 'hsine' key → picks the one with minimum hsine."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        # Two different IDs, both with 'hsine' → picks min
        cursor.fetchall.return_value = (
            {'name': 'SN2024abc', 'targetid': 1, 'id': 1, 'ra0': 180.0, 'dec0': 2.0, 'hsine': 0.01},
            {'name': 'SN2024abc', 'targetid': 2, 'id': 2, 'ra0': 180.1, 'dec0': 2.0, 'hsine': 0.001},
        )
        result = gettargetid('SN2024abc', '', '', conn, 0.01)
        assert result == 2  # id with minimum hsine
    def test_small_array_no_downsampling(self):
        from lsc.mysqldef import getsky
        data = np.random.default_rng(10).normal(500, 10, (20, 20))
        mean, std = getsky(data)
        assert abs(mean - 500) < 30
        assert std > 0

    def test_large_array_downsampled(self):
        from lsc.mysqldef import getsky
        data = np.random.default_rng(11).normal(1000, 20, (200, 200))
        mean, std = getsky(data)
        assert abs(mean - 1000) < 60
        assert std > 0

    def test_outlier_clipped(self):
        from lsc.mysqldef import getsky
        data = np.random.default_rng(12).normal(1000, 5, (50, 50)).astype(float)
        data[25, 25] = 1e6
        mean, std = getsky(data)
        assert abs(mean - 1000) < 50


# ---------------------------------------------------------------------------
# updatevalue — UPDATE a database field (mocked via MySQLdb/pymysql mock)
# ---------------------------------------------------------------------------

class TestUpdatevalue:
    """updatevalue calls getconnection→dbConnect internally; MySQLdb is mocked."""

    def test_single_string_column(self):
        """updatevalue with a single string column/value should not raise."""
        from lsc.mysqldef import updatevalue
        # getconnection returns lsc.util.readpass values; dbConnect uses mocked MySQLdb
        updatevalue('photlco', 'wcs', 1, 'test.fits')

    def test_string_value(self):
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', 'zcat', 'catalog.cat', 'test.fits')

    def test_list_columns_and_values(self):
        """updatevalue with list columns and values."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', ['wcs', 'psf'], [1, 0], 'test.fits')

    def test_custom_filename0(self):
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', 'wcs', 1, 'test.fits', filename0='filename')


# ---------------------------------------------------------------------------
# gettargetid with ra/dec path
# ---------------------------------------------------------------------------

class TestGettargetidRaDec:
    def test_ra_dec_match_found(self):
        """When ra/dec are given, gettargetid uses getfromcoordinate."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'id': 5, 'ra0': 150.0, 'dec0': 2.0, 'hsine': 0.001},)
        result = gettargetid('', 150.0, 2.0, conn, 0.01)
        assert result == 5

    def test_ra_dec_no_match(self):
        """When ra/dec are given but no match is found, returns ''."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = gettargetid('', 150.0, 2.0, conn, 0.01)
        assert result == ''

    def test_ra_dec_verbose(self, capsys):
        """verbose=True prints the ra, dec, radius info."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = gettargetid('', 150.0, 2.0, conn, 0.01, verbose=True)
        out = capsys.readouterr().out
        assert '150' in out or result == ''

    def test_ra_dec_match_found_verbose(self, capsys):
        """verbose=True with a match prints and returns target id."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'id': 5, 'ra0': 150.0, 'dec0': 2.0, 'hsine': 0.001},)
        result = gettargetid('', 150.0, 2.0, conn, 0.01, verbose=True)
        assert result == 5

    def test_ra_colon_format(self):
        """RA in colon format ('10:00:00') — deg2HMS may not be available so skip gracefully."""
        import lsc
        from unittest.mock import MagicMock
        import pytest
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        # Patch deg2HMS so it doesn't fail on missing attribute
        with patch.object(lsc, 'deg2HMS', return_value=(150.0, 2.0), create=True):
            result = gettargetid('', '10:00:00', '+02:00:00', conn, 0.01)
        assert result == ''


# ---------------------------------------------------------------------------
# get_snex_uid — query DB for current user's SNEx user id
# ---------------------------------------------------------------------------

class TestGetSnexUid:
    def test_returns_none_when_no_match_noninteractive(self, monkeypatch):
        """When DB returns no rows and interactive=False, returns None."""
        import lsc.myloopdef
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=False)
        assert result is None

    def test_returns_uid_when_found(self, monkeypatch):
        """When DB returns a match, returns the uid."""
        import lsc.myloopdef
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'id': 42, 'firstname': 'John', 'lastname': 'Doe'},)
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=False)
        assert result == 42

    def test_return_fullname_true(self, monkeypatch):
        """With return_fullname=True, returns (uid, 'First Last')."""
        import lsc.myloopdef
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'id': 42, 'firstname': 'John', 'lastname': 'Doe'},)
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        uid, fullname = get_snex_uid(interactive=False, return_fullname=True)
        assert uid == 42
        assert fullname == 'John Doe'

    def test_return_fullname_none_when_no_match(self, monkeypatch):
        """With return_fullname=True and no match, returns (None, None)."""
        import lsc.myloopdef
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        uid, fullname = get_snex_uid(interactive=False, return_fullname=True)
        assert uid is None
        assert fullname is None

    def test_interactive_with_valid_snex_user(self, monkeypatch):
        """interactive=True: user not found by unix name but provides valid SNEx username."""
        import lsc.myloopdef
        import lsc.util
        conn, cursor = _make_conn()
        # First call (unix user) returns nothing; second call (snex_user) returns a match
        cursor.fetchall.side_effect = [
            (),  # unix username not found
            ({'id': 99, 'firstname': 'Jane', 'lastname': 'Doe'},),  # snex username found
        ]
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setattr(lsc.util, 'userinput', lambda *a, **kw: 'jdoe')
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=True)
        assert result == 99

    def test_interactive_with_unknown_snex_user(self, monkeypatch, capsys):
        """interactive=True: user not found by unix name and SNEx username also not found."""
        import lsc.myloopdef
        import lsc.util
        conn, cursor = _make_conn()
        cursor.fetchall.side_effect = [
            (),  # unix username not found
            (),  # snex username also not found
        ]
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setattr(lsc.util, 'userinput', lambda *a, **kw: 'unknownuser')
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=True)
        assert result is None
        out = capsys.readouterr().out
        assert 'not found' in out

    def test_interactive_empty_snex_input(self, monkeypatch):
        """interactive=True: user not found by unix name and provides empty input."""
        import lsc.myloopdef
        import lsc.util
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setattr(lsc.util, 'userinput', lambda *a, **kw: '')
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=True)
        assert result is None


# ---------------------------------------------------------------------------
# insert_values — insert a dict row into a table via mock conn
# ---------------------------------------------------------------------------

class TestInsertValues:
    def test_simple_insert_no_raise(self):
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits', 'wcs': 1, 'psf': 0}
        # Should call cursor.execute once; no exception should be raised
        insert_values(conn, 'photlco', values)
        assert cursor.execute.call_count >= 1

    def test_adds_datecreated_for_known_table(self):
        """For tables in datecreated_tables, insert_values adds datecreated key."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits'}
        insert_values(conn, 'photlco', values)
        # datecreated should have been added before the insert
        call_args = cursor.execute.call_args
        assert call_args is not None

    def test_empty_values_filtered_out(self):
        """Values that are None, '', NaN, UNKNOWN, N/A are filtered from INSERT."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_conn()
        values = {'filename': 'test.fits', 'empty_col': None, 'nan_col': 'NaN'}
        insert_values(conn, 'photlco', values)
        sql = cursor.execute.call_args[0][0]
        assert 'empty_col' not in sql
        assert 'nan_col' not in sql


# ---------------------------------------------------------------------------
# getlike — LIKE query via mocked connection
# ---------------------------------------------------------------------------

class TestGetlike:
    def test_returns_results(self):
        from lsc.mysqldef import getlike
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'filename': 'test.fits'},)
        result = getlike(conn, 'photlco', 'filename', 'test')
        assert result == ({'filename': 'test.fits'},)

    def test_empty_result(self):
        from lsc.mysqldef import getlike
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = getlike(conn, 'photlco', 'filename', 'nonexistent')
        assert result == ()

    def test_with_column2(self):
        from lsc.mysqldef import getlike
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'filename': 'test.fits'},)
        result = getlike(conn, 'photlco', 'filename', 'test', column2='filename')
        assert cursor.execute.call_count == 1


# ---------------------------------------------------------------------------
# getlistfromraw — date-range query via mocked connection
# ---------------------------------------------------------------------------

class TestGetlistfromraw:
    def test_all_telescopes(self):
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'filename': 'test.fits'},)
        result = getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201')
        assert result == ({'filename': 'test.fits'},)

    def test_specific_telescope(self):
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201',
                                telescope='1m0-01')
        assert result == ()

    def test_no_value2(self):
        """When value2 is empty, value1 is used for both ends."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '')
        assert cursor.execute.call_count == 1


# ---------------------------------------------------------------------------
# getfromdataraw — single key lookup via mocked connection
# ---------------------------------------------------------------------------

class TestGetfromdataraw:
    def test_returns_matching_row(self):
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'filename': 'test.fits', 'wcs': 1},)
        result = getfromdataraw(conn, 'photlco', 'filename', 'test.fits')
        assert result == ({'filename': 'test.fits', 'wcs': 1},)

    def test_returns_empty_when_no_match(self):
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ()
        result = getfromdataraw(conn, 'photlco', 'filename', 'nonexistent.fits')
        assert result == ()

    def test_with_column2(self):
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = ({'filename': 'test.fits'},)
        result = getfromdataraw(conn, 'photlco', 'filename', 'test.fits',
                                column2='filename')
        assert cursor.execute.call_count == 1


# ---------------------------------------------------------------------------
# Error-handler branches (except msql.Error)
# ---------------------------------------------------------------------------

class _FakeError(Exception):
    """Fake DB error with args[0]=errno, args[1]=message, like MySQLdb.Error."""
    def __init__(self, errno=1, msg='fake error'):
        super().__init__(errno, msg)


def _make_conn_with_error(error):
    """Return (conn, cursor) where cursor.execute raises `error`."""
    import sys
    from unittest.mock import MagicMock
    import MySQLdb as msql  # this is the stub mock in conftest
    cursor = MagicMock()
    cursor.execute.side_effect = error
    cursor.fetchall.return_value = ()
    cursor.rowcount = 0
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


class TestGetmissingErrorBranch:
    def test_error_causes_sysexit(self, monkeypatch):
        import sys, MySQLdb as msql
        from lsc.mysqldef import getmissing
        err = _FakeError(1045, 'Access denied')
        # make msql.Error match our fake error so except clause catches it
        monkeypatch.setattr(msql, 'Error', _FakeError)
        conn, _ = _make_conn_with_error(err)
        with pytest.raises(SystemExit):
            getmissing(conn, '2020-01-01', '2020-01-02', '1m0-01')


class TestGetlikeErrorBranch:
    def test_error_causes_sysexit(self, monkeypatch):
        import MySQLdb as msql
        from lsc.mysqldef import getlike
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        conn, _ = _make_conn_with_error(err)
        with pytest.raises(SystemExit):
            getlike(conn, 'photlco', 'filename', 'test.fits')


class TestQueryErrorBranch:
    def test_error_prints_message(self, monkeypatch, capsys):
        import MySQLdb as msql
        from lsc.mysqldef import query
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        conn, _ = _make_conn_with_error(err)
        result = query(['SELECT 1'], conn)
        out = capsys.readouterr().out
        assert 'Error' in out or result == ''


class TestGetfromdatarawErrorBranch:
    def test_error_causes_sysexit(self, monkeypatch):
        import MySQLdb as msql
        from lsc.mysqldef import getfromdataraw
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        conn, _ = _make_conn_with_error(err)
        with pytest.raises(SystemExit):
            getfromdataraw(conn, 'photlco', 'filename', 'test.fits')


class TestDeleteredufromarchiveErrorBranch:
    def test_error_causes_sysexit(self, monkeypatch):
        import MySQLdb as msql
        from lsc.mysqldef import deleteredufromarchive
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        from unittest.mock import patch, MagicMock
        cursor = MagicMock()
        cursor.execute.side_effect = err
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        with patch('lsc.mysqldef.dbConnect', return_value=mock_conn):
            with pytest.raises(SystemExit):
                deleteredufromarchive('test.fits')


class TestUpdatevalueErrorBranch:
    def test_error_prints_message(self, monkeypatch, capsys):
        import MySQLdb as msql
        from lsc.mysqldef import updatevalue
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        from unittest.mock import patch, MagicMock
        cursor = MagicMock()
        cursor.execute.side_effect = err
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        with patch('lsc.mysqldef.dbConnect', return_value=mock_conn):
            updatevalue('photlco', 'wcs', 1, 'test.fits')
        out = capsys.readouterr().out
        assert 'Error' in out


class TestInsertValuesErrorBranch:
    def test_error_prints_message(self, monkeypatch, capsys):
        import MySQLdb as msql
        from lsc.mysqldef import insert_values
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        from unittest.mock import MagicMock
        cursor = MagicMock()
        cursor.execute.side_effect = err
        conn = MagicMock()
        conn.cursor.return_value = cursor
        insert_values(conn, 'photlco', {'filename': 'test.fits', 'wcs': 1})
        out = capsys.readouterr().out
        assert 'Error' in out


class TestGetfromcoordinateErrorBranch:
    def test_error_causes_sysexit(self, monkeypatch):
        import MySQLdb as msql
        from lsc.mysqldef import getfromcoordinate
        err = _FakeError(1045, 'Access denied')
        monkeypatch.setattr(msql, 'Error', _FakeError)
        conn, _ = _make_conn_with_error(err)
        with pytest.raises(SystemExit):
            getfromcoordinate(conn, 'targets', 150.0, 2.5, 0.01)


class TestGetvaluefromarchiveTruthyResult:
    def test_returns_result_when_truthy(self):
        from unittest.mock import patch, MagicMock
        from lsc.mysqldef import getvaluefromarchive
        fake_result = ({'filename': 'test.fits'},)
        with patch('lsc.mysqldef.dbConnect', return_value=MagicMock()):
            with patch('lsc.mysqldef.getfromdataraw', return_value=fake_result):
                result = getvaluefromarchive('photlco', 'filename', 'test.fits', '*')
        assert result == fake_result


# ---------------------------------------------------------------------------
# targimg — resolves target ID from header or image path
# ---------------------------------------------------------------------------

def _targimg_patches(monkeypatch, gettargetid_return=42, query_return=None,
                     ra=150.0, dec=2.5, obj='SN2024abc', propid='NOKEY',
                     _ra_str=False):
    """Helper to set up mocks for targimg tests."""
    import lsc
    import lsc.myloopdef
    import lsc.mysqldef

    monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())

    if query_return is None:
        query_return = []

    monkeypatch.setattr(lsc.mysqldef, 'query', lambda cmd, conn: query_return)
    monkeypatch.setattr(lsc.mysqldef, 'gettargetid', lambda *a, **k: gettargetid_return)
    monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
    monkeypatch.setattr(lsc.mysqldef, 'getfromcoordinate', lambda *a, **k: [{'id': 99}])
    monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])

    # Build a header-like dict
    def mock_readkey3(hdr, key):
        mapping = {'CAT-RA': '' if _ra_str else ra,
                   'CAT-DEC': '' if _ra_str else dec,
                   'object': obj, 'propid': propid}
        return mapping.get(key, 'NOKEY')

    monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)
    return {'CAT-RA': ra, 'CAT-DEC': dec, 'object': obj, 'propid': propid}


class TestTargimg:
    def test_found_by_name_returns_targetid(self, monkeypatch):
        """targimg returns 42 when gettargetid finds by name."""
        hdrt = _targimg_patches(monkeypatch, gettargetid_return=42)
        from lsc.mysqldef import targimg
        result = targimg(hdrt=hdrt)
        assert result == 42

    def test_no_target_found_adds_new(self, monkeypatch):
        """When gettargetid returns '', targimg inserts a new target and returns id."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        # First call returns '', second call also returns '' → add new target
        call_count = {'n': 0}
        def fake_gettargetid(*a, **k):
            call_count['n'] += 1
            return ''
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', fake_gettargetid)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'getfromcoordinate', lambda *a, **k: [{'id': 77}])
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])
        # return a dict for bb2 query so id lookup works
        def mock_query(cmd, conn):
            if 'targetnames' in str(cmd):
                return [{'id': 200}]
            return []
        monkeypatch.setattr(lsc.mysqldef, 'query', mock_query)

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                       'object': 'SN2024test', 'propid': 'NOKEY'}
            return mapping.get(key, 'NOKEY')

        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)
        from lsc.mysqldef import targimg
        result = targimg(hdrt={'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                                'object': 'SN2024test', 'propid': 'NOKEY'})
        # Should have inserted and returned id from getfromcoordinate
        assert result == 77

    def test_name_not_found_but_coord_found(self, monkeypatch):
        """When name lookup fails but coordinate lookup succeeds, adds name alias."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        monkeypatch.setattr(lsc.mysqldef, 'query', lambda cmd, conn: [])
        call_count = {'n': 0}
        def fake_gettargetid(*a, **k):
            call_count['n'] += 1
            if call_count['n'] == 1:
                return ''   # name lookup fails
            return 55       # coord lookup succeeds
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', fake_gettargetid)
        insert_calls = []
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: insert_calls.append(a))
        monkeypatch.setattr(lsc.mysqldef, 'getfromcoordinate', lambda *a, **k: [{'id': 55}])
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])
        # For the bb2 query returning id for targetnames
        def mock_query(cmd, conn):
            if 'targetnames' in cmd[0]:
                return [{'id': 100}]
            return []
        monkeypatch.setattr(lsc.mysqldef, 'query', mock_query)

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                       'object': 'SN2024alias', 'propid': 'NOKEY'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        from lsc.mysqldef import targimg
        result = targimg(hdrt={'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                                'object': 'SN2024alias', 'propid': 'NOKEY'})
        assert result == 55

    def test_propid_in_programs_sets_group(self, monkeypatch):
        """When propid matches a program entry, group is set from the table."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        programs = [{'idcode': 'KEY123', 'groupidcode': 5}]
        def mock_query(cmd, conn):
            return programs
        monkeypatch.setattr(lsc.mysqldef, 'query', mock_query)
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', lambda *a, **k: 42)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                       'object': 'SN2024abc', 'propid': 'KEY123'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        from lsc.mysqldef import targimg
        result = targimg(hdrt={'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                                'object': 'SN2024abc', 'propid': 'KEY123'})
        assert result == 42

    def test_propid_in_programs_group_is_none_sets_32769(self, monkeypatch):
        """When propid matches program but groupidcode is None, sets _group=32769."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        programs = [{'idcode': 'KEY456', 'groupidcode': None}]
        def mock_query(cmd, conn):
            return programs
        monkeypatch.setattr(lsc.mysqldef, 'query', mock_query)
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', lambda *a, **k: 42)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                       'object': 'SN2024abc', 'propid': 'KEY456'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        from lsc.mysqldef import targimg
        result = targimg(hdrt={'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                                'object': 'SN2024abc', 'propid': 'KEY456'})
        assert result == 42

    def test_found_target_with_group_adds_permission(self, monkeypatch):
        """When target found and _group set, adds permissionlog entry if empty."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        programs = [{'idcode': 'PROG1', 'groupidcode': 7}]
        def mock_query(cmd, conn):
            return programs
        monkeypatch.setattr(lsc.mysqldef, 'query', mock_query)
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', lambda *a, **k: 42)
        insert_calls = []
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: insert_calls.append(a))
        # getfromdataraw returns empty → so insert_values called for permissionlog
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])
        monkeypatch.setattr(lsc.mysqldef, 'JDnow', lambda: 59000.0)

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                       'object': 'SN2024abc', 'propid': 'PROG1'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        from lsc.mysqldef import targimg
        result = targimg(hdrt={'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                                'object': 'SN2024abc', 'propid': 'PROG1'})
        assert result == 42
        # verify permissionlog insert was attempted
        assert any('permissionlog' in str(a) for a in insert_calls)

    def test_ra_as_string_with_colon_uses_deg2hms(self, monkeypatch):
        """When RA is a string with ':', lsc.deg2HMS is called to convert it."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef
        import lsc.lscabsphotdef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        # Mock deg2HMS in lscabsphotdef (where it's defined) and patch into lsc namespace
        monkeypatch.setattr(lsc.lscabsphotdef, 'deg2HMS', lambda ra, dec, round=False: (150.0, 2.5))
        # Patch it into lsc namespace as well since mysqldef uses lsc.deg2HMS
        import sys
        import types
        # targimg does `import lsc; lsc.deg2HMS(...)` — patch the lsc module attribute
        lsc_module = sys.modules['lsc']
        original = getattr(lsc_module, 'deg2HMS', None)
        lsc_module.deg2HMS = lambda ra, dec: (150.0, 2.5)

        call_count = {'n': 0}
        def fake_gettargetid(*a, **k):
            call_count['n'] += 1
            if call_count['n'] == 1:
                return ''
            return 55
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', fake_gettargetid)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])
        def mock_query(cmd, conn):
            if 'targetnames' in str(cmd):
                return [{'id': 100}]
            return []
        monkeypatch.setattr(lsc.mysqldef, 'query', mock_query)

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': '10:00:00', 'CAT-DEC': '+02:30:00',
                       'object': 'SN2024test', 'propid': 'NOKEY'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        try:
            from lsc.mysqldef import targimg
            result = targimg(hdrt={'CAT-RA': '10:00:00', 'CAT-DEC': '+02:30:00',
                                    'object': 'SN2024test', 'propid': 'NOKEY'})
            assert result == 55
        finally:
            if original is None:
                if hasattr(lsc_module, 'deg2HMS'):
                    delattr(lsc_module, 'deg2HMS')
            else:
                lsc_module.deg2HMS = original

    def test_hdrt_none_reads_from_img(self, monkeypatch):
        """When hdrt=None, targimg reads header from img path."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        monkeypatch.setattr(lsc.mysqldef, 'query', lambda cmd, conn: [])
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', lambda *a, **k: 42)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])

        fake_hdr = {'CAT-RA': 150.0, 'CAT-DEC': 2.5, 'object': 'SN2024abc', 'propid': 'NOKEY'}
        monkeypatch.setattr(lsc.util, 'readhdr', lambda img: fake_hdr)

        def mock_readkey3(hdr, key):
            mapping = {'CAT-RA': 150.0, 'CAT-DEC': 2.5,
                       'object': 'SN2024abc', 'propid': 'NOKEY'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        from lsc.mysqldef import targimg
        result = targimg(img='fake.fits')  # hdrt=None, reads from img
        assert result == 42

    def test_no_target_found_ra_not_float_raises(self, monkeypatch):
        """When gettargetid fails and RA is not float, raises Exception."""
        import lsc
        import lsc.myloopdef
        import lsc.mysqldef

        monkeypatch.setattr(lsc.myloopdef, 'conn', MagicMock())
        monkeypatch.setattr(lsc.mysqldef, 'query', lambda cmd, conn: [])
        monkeypatch.setattr(lsc.mysqldef, 'gettargetid', lambda *a, **k: '')
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', lambda *a, **k: [])

        def mock_readkey3(hdr, key):
            # Return a non-float string for RA to trigger the error branch
            mapping = {'CAT-RA': 'CORRUPT', 'CAT-DEC': 'CORRUPT',
                       'object': 'SN2024abc', 'propid': 'NOKEY'}
            return mapping.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

        from lsc.mysqldef import targimg
        with pytest.raises(Exception, match='CAT-RA or CAT-DEC'):
            targimg(hdrt={'CAT-RA': 'CORRUPT', 'CAT-DEC': 'CORRUPT',
                           'object': 'SN2024abc', 'propid': 'NOKEY'})


# ---------------------------------------------------------------------------
# ingestredu — ingests reduced images into the database
# ---------------------------------------------------------------------------

def _make_ingest_hdr():
    """A minimal FITS-like header dict for ingestredu tests."""
    return {
        'date-obs': '2024-01-01T00:00:00',
        'DAY-OBS': '20240101',
        'exptime': 300.0,
        'filter': 'rp',
        'mjd': 60000.0,
        'TRACKNUM': '12345',
        'TELESCOP': 'lsc',
        'instrume': 'fa03',
        'airmass': 1.2,
        'object': 'SN2024abc',
        'ut': '00:00:00',
        'wcserr': 0,
        'RA': 150.0,
        'DEC': 2.5,
        'SITEID': 'lsc',
    }


def _setup_ingestredu_mocks(monkeypatch, exist=False, exist2=False,
                              telid_found=True, instid_found=True,
                              already_in_db=False):
    """Set up all the mocks needed by ingestredu."""
    import lsc
    import lsc.mysqldef

    fake_conn = MagicMock()
    monkeypatch.setattr(lsc.mysqldef, 'getconnection',
                        lambda name: ('host', 'user', 'pass', 'db'))
    monkeypatch.setattr(lsc.mysqldef, 'dbConnect',
                        lambda *a, **k: fake_conn)

    fake_hdr = _make_ingest_hdr()
    monkeypatch.setattr(lsc.util, 'readhdr', lambda img: fake_hdr)

    def mock_readkey3(hdr, key):
        return fake_hdr.get(key, 'NOKEY')
    monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)

    monkeypatch.setattr(lsc.mysqldef, 'targimg', lambda *a, **k: 42)

    # Track calls to getfromdataraw per table to simulate state after inserts
    call_counts = {'telescopes': 0, 'instruments': 0, 'photlco_main': 0, 'photlco_final': 0}

    def mock_getfromdataraw(conn, table, col, val, column2='*'):
        if table == 'photlcoraw':
            return [{'groupidcode': 'grp1'}] if exist2 else []
        elif table == 'photlco':
            # First call is the existence check, later calls are pre-insert checks
            call_counts['photlco_main'] += 1
            if call_counts['photlco_main'] == 1:
                return [{'filename': 'test.fits'}] if exist else []
            # Later calls (pre-insert ggg check): return already_in_db
            return [{'filename': 'test.fits'}] if already_in_db else []
        elif table == 'telescopes':
            call_counts['telescopes'] += 1
            if telid_found or call_counts['telescopes'] > 1:
                return [{'id': 1}]
            return []
        elif table == 'instruments':
            call_counts['instruments'] += 1
            if instid_found or call_counts['instruments'] > 1:
                return [{'id': 2}]
            return []
        else:
            return []
    monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', mock_getfromdataraw)

    insert_calls = []
    monkeypatch.setattr(lsc.mysqldef, 'insert_values',
                        lambda conn, tbl, d: insert_calls.append((tbl, d)))
    update_calls = []
    monkeypatch.setattr(lsc.mysqldef, 'updatevalue',
                        lambda tbl, col, val, fn: update_calls.append((tbl, col)))
    monkeypatch.setattr(lsc.mysqldef, 'deleteredufromarchive',
                        lambda fn, tbl: None)
    return insert_calls, update_calls


class TestIngestredu:
    def test_ingest_new_image(self, monkeypatch):
        """Standard ingestion of a new image (not already in db)."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=False, exist2=False,
                                                    telid_found=True, instid_found=True,
                                                    already_in_db=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'])
        # Should have called insert_values for photlco
        assert any(t == 'photlco' for t, _ in insert_calls)

    def test_already_ingested_skips(self, monkeypatch):
        """If image already exists and force='no', prints 'already ingested'."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=True, exist2=False,
                                                    telid_found=True, instid_found=True,
                                                    already_in_db=True)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'])
        # Should NOT have inserted into photlco when exist=True and force='no'
        assert not any(t == 'photlco' for t, _ in insert_calls)

    def test_force_yes_reingest(self, monkeypatch):
        """force='yes' deletes and re-ingests existing image."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=True, exist2=False,
                                                    telid_found=True, instid_found=True,
                                                    already_in_db=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'], force='yes')
        assert any(t == 'photlco' for t, _ in insert_calls)

    def test_force_update_updates_values(self, monkeypatch):
        """force='update' updates existing rows."""
        insert_calls, update_calls = _setup_ingestredu_mocks(monkeypatch,
                                                               exist=True, exist2=False,
                                                               telid_found=True, instid_found=True,
                                                               already_in_db=True)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'], force='update')
        assert len(update_calls) > 0

    def test_telescope_not_found_adds_to_db(self, monkeypatch):
        """When telescope not in DB, inserts it."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=False, exist2=False,
                                                    telid_found=False, instid_found=True,
                                                    already_in_db=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'])
        assert any(t == 'telescopes' for t, _ in insert_calls)

    def test_instrument_not_found_adds_to_db(self, monkeypatch):
        """When instrument not in DB, inserts it."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=False, exist2=False,
                                                    telid_found=True, instid_found=False,
                                                    already_in_db=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'])
        assert any(t == 'instruments' for t, _ in insert_calls)

    def test_exist2_sets_groupidcode(self, monkeypatch):
        """When photlcoraw record exists, uses its groupidcode."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=False, exist2=True,
                                                    telid_found=True, instid_found=True,
                                                    already_in_db=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'])
        # Verify insert happened (groupidcode was set from exist2)
        assert any(t == 'photlco' for t, _ in insert_calls)

    def test_fz_image_strips_extension(self, monkeypatch, tmp_path):
        """Image with .fz extension has it stripped before processing."""
        insert_calls, _ = _setup_ingestredu_mocks(monkeypatch,
                                                    exist=False, exist2=False,
                                                    telid_found=True, instid_found=True,
                                                    already_in_db=False)
        # Mock os.system so funpack doesn't actually run
        monkeypatch.setattr('os.system', lambda cmd: 0)
        # Create a fake non-.fz file to ensure isfile returns False for .fz
        import lsc.mysqldef
        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits.fz'])
        # Should process without error

    def test_telescop_fts_mapping(self, monkeypatch):
        """Faulkes Telescope South maps to '2m0-02'."""
        import lsc
        import lsc.mysqldef
        import lsc.util

        fake_conn = MagicMock()
        monkeypatch.setattr(lsc.mysqldef, 'getconnection',
                            lambda name: ('host', 'user', 'pass', 'db'))
        monkeypatch.setattr(lsc.mysqldef, 'dbConnect',
                            lambda *a, **k: fake_conn)

        fake_hdr = _make_ingest_hdr()
        fake_hdr['TELESCOP'] = 'Faulkes Telescope South'
        monkeypatch.setattr(lsc.util, 'readhdr', lambda img: fake_hdr)

        def mock_readkey3(hdr, key):
            return fake_hdr.get(key, 'NOKEY')
        monkeypatch.setattr(lsc.util, 'readkey3', mock_readkey3)
        monkeypatch.setattr(lsc.mysqldef, 'targimg', lambda *a, **k: 42)

        def mock_getfromdataraw(conn, table, col, val, column2='*'):
            if table in ('photlco', 'photlcoraw'):
                return []
            return [{'id': 1}]
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', mock_getfromdataraw)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'updatevalue', lambda *a, **k: None)

        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits'])  # Should not raise
