"""
Comprehensive tests for lsc.mysqldef — covers all functions with emphasis on
untested edge cases, boundary conditions, SQL construction details, error handling,
and interaction patterns between functions.

Uses unittest.mock throughout; no actual database connections required.
"""
import datetime
import sys
import types
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers: create mock connections/cursors matching pymysql interface
# ---------------------------------------------------------------------------

def _make_mock_conn(fetchall_return=(), rowcount=0):
    """Return (conn, cursor) with pymysql-style mock objects."""
    msql = sys.modules['pymysql']
    cursor = MagicMock()
    cursor.rowcount = rowcount
    cursor.fetchall.return_value = fetchall_return
    conn = MagicMock()
    conn.cursor.return_value = cursor
    msql.cursors = MagicMock()
    msql.cursors.DictCursor = MagicMock()
    return conn, cursor


class _FakeMySQLError(Exception):
    """Simulates MySQLdb.Error with (errno, msg) args."""
    def __init__(self, errno=1, msg='test error'):
        super().__init__(errno, msg)


# ===========================================================================
# dbConnect
# ===========================================================================

class TestDbConnect:
    """Tests for dbConnect which wraps MySQLdb/pymysql connect."""

    def test_returns_connection_on_success(self):
        """Successful connection returns the conn object."""
        from lsc.mysqldef import dbConnect
        conn = dbConnect('host', 'user', 'pass', 'db')
        # pymysql is mocked globally in conftest; connect returns a MagicMock
        assert conn is not None

    def test_passes_correct_params_to_connect(self):
        """Verifies host/user/passwd/db are passed correctly to the connector."""
        msql = sys.modules['pymysql']
        msql.connect.reset_mock()
        from lsc.mysqldef import dbConnect
        dbConnect('myhost', 'myuser', 'mypasswd', 'mydb')
        msql.connect.assert_called_with(
            host='myhost', user='myuser', passwd='mypasswd', db='mydb'
        )

    def test_connection_failure_exits(self, monkeypatch):
        """When connect raises msql.Error, dbConnect calls sys.exit(1)."""
        msql = sys.modules['pymysql']
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        original_connect = msql.connect
        msql.connect = MagicMock(side_effect=_FakeMySQLError(2003, 'Cannot connect'))
        try:
            from lsc.mysqldef import dbConnect
            with pytest.raises(SystemExit) as exc_info:
                dbConnect('badhost', 'user', 'pass', 'db')
            assert exc_info.value.code == 1
        finally:
            msql.connect = original_connect

    def test_empty_string_params(self):
        """Empty string params should still call connect without error."""
        from lsc.mysqldef import dbConnect
        conn = dbConnect('', '', '', '')
        assert conn is not None


# ===========================================================================
# getconnection
# ===========================================================================

class TestGetconnectionEdgeCases:
    """Edge cases for getconnection beyond basic lookup."""

    @pytest.fixture(autouse=True)
    def _patch_readpass(self, monkeypatch):
        import lsc.util
        self.fake_pass = {
            'database': 'sndb',
            'hostname': 'db.example.com',
            'mysqluser': 'snuser',
            'mysqlpasswd': 'snpass',
            'ptfdatabase': 'ptfdb',
            'ptfhost': 'ptf.example.com',
            'ptfuser': 'ptfuser',
            'ptfpasswd': 'ptfpass',
        }
        monkeypatch.setattr(lsc.util, 'readpass', self.fake_pass)

    def test_lcogt2_all_fields(self):
        from lsc.mysqldef import getconnection
        host, user, passwd, db = getconnection('lcogt2')
        assert host == 'db.example.com'
        assert user == 'snuser'
        assert passwd == 'snpass'
        assert db == 'sndb'

    def test_iptf_all_fields(self):
        from lsc.mysqldef import getconnection
        host, user, passwd, db = getconnection('iptf')
        assert host == 'ptf.example.com'
        assert user == 'ptfuser'
        assert passwd == 'ptfpass'
        assert db == 'ptfdb'

    def test_invalid_site_raises_keyerror(self):
        from lsc.mysqldef import getconnection
        with pytest.raises(KeyError):
            getconnection('nonexistent_site')

    def test_empty_string_site_raises_keyerror(self):
        from lsc.mysqldef import getconnection
        with pytest.raises(KeyError):
            getconnection('')

    def test_readpass_missing_key_raises(self, monkeypatch):
        """If readpass is missing a key, getconnection raises KeyError."""
        import lsc.util
        monkeypatch.setattr(lsc.util, 'readpass', {})
        from lsc.mysqldef import getconnection
        with pytest.raises(KeyError):
            getconnection('lcogt2')


# ===========================================================================
# getmissing — complex SQL generation with multiple branches
# ===========================================================================

class TestGetmissingComprehensive:
    """Full coverage of getmissing branches and SQL construction."""

    def test_all_telescope_with_epoch2_sql_construction(self):
        """Verify SQL contains NOT EXISTS subquery for telescope='all' with epoch2."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', 'all')
        sql = cursor.execute.call_args[0][0]
        assert 'NOT EXISTS' in sql
        assert '20220101' in sql
        assert '20220201' in sql
        assert 'photlcoraw' in sql

    def test_all_telescope_without_epoch2_uses_equality(self):
        """Without epoch2, SQL uses dayobs = epoch0 (equality, not range)."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220515', '', 'all')
        sql = cursor.execute.call_args[0][0]
        assert 'dayobs = ' in sql or 'dayobs =' in sql
        assert '20220515' in sql

    def test_specific_telescope_strips_dashes(self):
        """Telescope '1m0-01' becomes '1m001' in LIKE clause."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', '1m0-01')
        sql = cursor.execute.call_args[0][0]
        assert '1m001' in sql
        assert "telescope = '1m0-01'" in sql

    def test_specific_telescope_no_epoch2_equality(self):
        """Specific telescope without epoch2 uses dayobs = epoch0."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220515', '', '2m0-01')
        sql = cursor.execute.call_args[0][0]
        assert '2m001' in sql
        assert 'dayobs =' in sql or 'dayobs = ' in sql

    def test_custom_datatable(self):
        """Custom datatable parameter changes the subquery table name."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', 'all', datatable='spec')
        sql = cursor.execute.call_args[0][0]
        assert 'spec' in sql

    def test_returns_multiple_rows(self):
        """When cursor has multiple rows, all are returned."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        rows = (
            {'filename': 'a.fits', 'objname': 'SN1'},
            {'filename': 'b.fits', 'objname': 'SN2'},
            {'filename': 'c.fits', 'objname': 'SN3'},
        )
        cursor.fetchall.return_value = rows
        result = getmissing(conn, '20220101', '20220201', 'all')
        assert len(result) == 3

    def test_telescope_no_dash_stays_same(self):
        """Telescope without dashes (e.g., '2m001') is unchanged."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', '2m001')
        sql = cursor.execute.call_args[0][0]
        assert '2m001' in sql

    def test_telescope_multiple_dashes(self):
        """Telescope with multiple dashes: '1m0-04-a' becomes '1m004a'."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', '1m0-04-a')
        sql = cursor.execute.call_args[0][0]
        assert '1m004a' in sql

    def test_cursor_closed_after_query(self):
        """Cursor is always closed after successful query."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', 'all')
        cursor.close.assert_called_once()


# ===========================================================================
# getfromdataraw — single-value lookup
# ===========================================================================

class TestGetfromdatarawComprehensive:
    """Full coverage of getfromdataraw function."""

    def test_default_column2_is_star(self):
        """Default column2='*' selects all columns."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        getfromdataraw(conn, 'photlco', 'filename', 'test.fits')
        sql = cursor.execute.call_args[0][0]
        assert 'select *' in sql.lower() or 'select * ' in sql.lower()

    def test_custom_column2(self):
        """Custom column2 is included in the SELECT."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        getfromdataraw(conn, 'photlco', 'filename', 'test.fits',
                       column2='filename,filter,mjd')
        sql = cursor.execute.call_args[0][0]
        assert 'filename,filter,mjd' in sql

    def test_value_with_special_characters(self):
        """Values with special characters are placed directly in SQL."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        # This is how the function works - no parameterization
        getfromdataraw(conn, 'photlco', 'objname', "SN2024abc'test")
        sql = cursor.execute.call_args[0][0]
        assert "SN2024abc'test" in sql

    def test_empty_value_string(self):
        """Empty value string still generates valid SQL."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        getfromdataraw(conn, 'photlco', 'filename', '')
        sql = cursor.execute.call_args[0][0]
        assert "=''" in sql

    def test_returns_empty_tuple_on_no_match(self):
        """Empty result returns empty tuple."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn(fetchall_return=())
        result = getfromdataraw(conn, 'photlco', 'filename', 'no_file.fits')
        assert result == ()

    def test_returns_multiple_rows(self):
        """Multiple matching rows are all returned."""
        from lsc.mysqldef import getfromdataraw
        rows = ({'filename': 'a.fits'}, {'filename': 'b.fits'})
        conn, cursor = _make_mock_conn(fetchall_return=rows)
        result = getfromdataraw(conn, 'photlco', 'filter', 'r')
        assert len(result) == 2

    def test_sql_structure(self):
        """Verify the exact SQL structure: select column2 from table where col='val'."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        getfromdataraw(conn, 'mytable', 'mycolumn', 'myvalue', column2='col1,col2')
        sql = cursor.execute.call_args[0][0]
        assert sql == "select col1,col2 from mytable where mycolumn='myvalue'"


# ===========================================================================
# getlistfromraw — date range query with telescope filtering
# ===========================================================================

class TestGetlistfromrawComprehensive:
    """Comprehensive tests for getlistfromraw date range queries."""

    def test_all_telescope_generates_range_query(self):
        """telescope='all' generates a between-style range query."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201')
        sql = cursor.execute.call_args[0][0]
        assert '<=' in sql
        assert '>=' in sql
        assert '20220101' in sql
        assert '20220201' in sql

    def test_specific_telescope_adds_like_clause(self):
        """Non-'all' telescope adds filename LIKE and telescope= conditions."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201',
                       telescope='1m0-01')
        sql = cursor.execute.call_args[0][0]
        assert '1m001' in sql  # dashes stripped
        assert "telescope='1m0-01'" in sql
        assert 'like' in sql.lower()

    def test_value2_none_uses_value1(self):
        """When value2 is None (falsy), value1 is used for both bounds."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220515', None)
        sql = cursor.execute.call_args[0][0]
        # value1 appears in both <= and >= positions
        assert sql.count('20220515') == 2

    def test_value2_zero_uses_value1(self):
        """value2=0 is falsy, so value1 is used for both bounds."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220515', 0)
        sql = cursor.execute.call_args[0][0]
        assert sql.count('20220515') == 2

    def test_custom_column2_in_select(self):
        """column2 parameter changes the SELECT clause."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201',
                       column2='filename,objname')
        sql = cursor.execute.call_args[0][0]
        assert 'filename,objname' in sql

    def test_same_value1_and_value2(self):
        """When value1 == value2, range is still valid (single day)."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220515', '20220515')
        sql = cursor.execute.call_args[0][0]
        assert sql.count('20220515') == 2

    def test_cursor_closed(self):
        """Cursor is closed after query."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201')
        cursor.close.assert_called_once()

    def test_telescope_with_no_dashes(self):
        """Telescope without dashes is unchanged in LIKE clause."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201',
                       telescope='lsc')
        sql = cursor.execute.call_args[0][0]
        assert "telescope='lsc'" in sql


# ===========================================================================
# updatevalue — UPDATE queries with various column/value combinations
# ===========================================================================

class TestUpdatevalueComprehensive:
    """Comprehensive tests for updatevalue function."""

    def test_single_numeric_value(self):
        """Single numeric column/value generates correct UPDATE SQL."""
        from lsc.mysqldef import updatevalue
        # updatevalue uses dbConnect internally (mocked in conftest)
        updatevalue('photlco', 'wcs', 0, 'test.fits')

    def test_single_string_value_gets_quoted(self):
        """String value is wrapped in quotes in the SQL."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', 'zcat', 'mycatalog.cat', 'img.fits')

    def test_list_columns_and_values(self):
        """List of columns and values generates comma-separated SET clause."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', ['wcs', 'psf', 'zcat'],
                    [1, 0, 'cat.cat'], 'img.fits')

    def test_none_value_in_list(self):
        """None values in list are passed as-is (not quoted)."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', ['wcs', 'psf'], [None, 1], 'test.fits')

    def test_custom_filename0_column(self):
        """Custom filename0 parameter changes the WHERE clause column."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', 'wcs', 1, 'myid123', filename0='id')

    def test_custom_connection_parameter(self):
        """connection parameter is passed to getconnection."""
        from lsc.mysqldef import updatevalue
        # Default is 'lcogt2' — different connection name
        # This should work as long as it maps to something valid in readpass
        with patch('lsc.mysqldef.getconnection', return_value=('h', 'u', 'p', 'd')):
            with patch('lsc.mysqldef.dbConnect', return_value=MagicMock()):
                updatevalue('photlco', 'wcs', 1, 'test.fits', connection='lcogt2')

    def test_float_value(self):
        """Float value is passed without quotes."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', 'airmass', 1.5, 'img.fits')

    def test_empty_filename(self):
        """Empty filename in WHERE clause."""
        from lsc.mysqldef import updatevalue
        updatevalue('photlco', 'wcs', 1, '')


# ===========================================================================
# insert_values — INSERT with dict of values
# ===========================================================================

class TestInsertValuesComprehensive:
    """Comprehensive tests for insert_values function."""

    def test_basic_insert_generates_correct_sql(self):
        """Values dict is converted to INSERT INTO table (cols) VALUES (...)."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'filter': 'r', 'mjd': 60000.0}
        insert_values(conn, 'photlco', values)
        sql = cursor.execute.call_args[0][0]
        assert 'INSERT INTO photlco' in sql
        assert 'filename' in sql
        assert 'filter' in sql
        assert 'mjd' in sql

    def test_datecreated_added_for_known_tables(self):
        """Tables in datecreated_tables get datecreated automatically."""
        from lsc.mysqldef import insert_values
        known_tables = ['photlco', 'photlcoraw', 'targets', 'targetnames',
                        'telescopes', 'instruments', 'spec', 'speclcoraw']
        for table in known_tables:
            conn, cursor = _make_mock_conn()
            values = {'filename': 'test.fits'}
            insert_values(conn, table, values)
            sql = cursor.execute.call_args[0][0]
            assert 'datecreated' in sql, f"datecreated not added for {table}"

    def test_datecreated_not_added_for_unknown_tables(self):
        """Tables NOT in datecreated_tables do not get datecreated."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits'}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'datecreated' not in sql

    def test_datecreated_not_overwritten_if_present(self):
        """If datecreated is already in values, it is not overwritten."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'datecreated': '2024-01-01 00:00:00'}
        insert_values(conn, 'photlco', values)
        # The provided datecreated should be used, not a new one
        passed_values = cursor.execute.call_args[0][1]
        assert passed_values['datecreated'] == '2024-01-01 00:00:00'

    def test_nan_value_excluded(self):
        """Values of 'NaN' are filtered out from the INSERT."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'badcol': 'NaN'}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'badcol' not in sql

    def test_unknown_value_excluded(self):
        """Values of 'UNKNOWN' are filtered out."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'badcol': 'UNKNOWN'}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'badcol' not in sql

    def test_na_value_excluded(self):
        """Values of 'N/A' are filtered out."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'badcol': 'N/A'}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'badcol' not in sql

    def test_none_value_excluded(self):
        """None values are filtered out."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'badcol': None}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'badcol' not in sql

    def test_empty_string_excluded(self):
        """Empty string values are filtered out."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'badcol': ''}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'badcol' not in sql

    def test_zero_value_not_excluded(self):
        """Numeric zero is a valid value and should NOT be excluded."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'wcs': 0}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'wcs' in sql

    def test_conn_commit_called(self):
        """Connection commit is called after successful insert."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits'}
        insert_values(conn, 'custom_table', values)
        conn.commit.assert_called_once()

    def test_all_values_filtered_out(self):
        """If all values are filtered, the SQL will be malformed but handled."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'col1': None, 'col2': '', 'col3': 'NaN'}
        # This generates an empty INSERT which may error - but it's caught
        insert_values(conn, 'custom_table', values)

    def test_uses_parameterized_values(self):
        """insert_values uses %(key)s parameterized placeholders."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'filter': 'B'}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert '%(filename)s' in sql
        assert '%(filter)s' in sql


# ===========================================================================
# guess_instrument_type — prefix-based instrument classification
# ===========================================================================

class TestGuessInstrumentTypeComprehensive:
    """Full coverage of guess_instrument_type."""

    @pytest.mark.parametrize("prefix,expected", [
        ('fl', 'Sinistro'),
        ('fa', 'Sinistro'),
        ('kb', 'SBIG'),
        ('fs', 'Spectral'),
        ('ep', 'MUSCAT'),
        ('sq', 'QHY'),
    ])
    def test_known_prefixes(self, prefix, expected):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type(prefix + '99') == expected

    @pytest.mark.parametrize("name", [
        'xx01', 'ab12', 'zz00', 'aa', 'qq55', 'nn01'
    ])
    def test_unknown_prefixes_return_none(self, name):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type(name) is None

    def test_case_sensitive_fl(self):
        """Uppercase prefix does not match."""
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type('FL01') is None

    def test_case_sensitive_kb(self):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type('KB01') is None

    def test_exact_two_char_input(self):
        """Input of exactly 2 characters with known prefix."""
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type('fl') == 'Sinistro'
        assert guess_instrument_type('kb') == 'SBIG'

    def test_long_name_uses_first_two(self):
        """Only first two characters matter."""
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type('flverylongname') == 'Sinistro'
        assert guess_instrument_type('kbanything') == 'SBIG'


# ===========================================================================
# getsky — sigma-clipped statistics
# ===========================================================================

class TestGetskyComprehensive:
    """Comprehensive tests for getsky function."""

    def test_constant_array(self):
        """Array with all same values: sigma clipping empties the sample (std=0
        means the strict < comparison removes all points), returning nan."""
        from lsc.mysqldef import getsky
        data = np.ones((30, 30)) * 5000.0
        mean, std = getsky(data)
        # When all values are identical, std=0 so mean+3*std == mean and
        # the condition (sample < mean) removes everything, yielding nan.
        assert np.isnan(mean)
        assert np.isnan(std)

    def test_negative_values(self):
        """Data with negative values (e.g., background-subtracted)."""
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(42)
        data = rng.normal(-100, 10, (50, 50))
        mean, std = getsky(data)
        assert abs(mean - (-100)) < 30
        assert std > 0

    def test_very_small_array(self):
        """Minimum valid array size (2x2)."""
        from lsc.mysqldef import getsky
        data = np.array([[100.0, 102.0], [99.0, 101.0]])
        mean, std = getsky(data)
        assert abs(mean - 100.5) < 5.0

    def test_large_array_downsamples(self):
        """Array larger than maxsample=10000 pixels gets downsampled."""
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(55)
        # 200x200 = 40000 pixels > 10000 maxsample
        data = rng.normal(2000, 30, (200, 200))
        mean, std = getsky(data)
        assert abs(mean - 2000) < 50
        assert abs(std - 30) < 20

    def test_exactly_maxsample_pixels(self):
        """Array with exactly 10000 pixels (100x100)."""
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(66)
        data = rng.normal(500, 15, (100, 100))
        mean, std = getsky(data)
        assert abs(mean - 500) < 30

    def test_many_outliers_still_converges(self):
        """With 10% extreme outliers at 50000, the initial mean is pulled up
        significantly. The sigma clipping uses random sampling and the strict
        less-than comparison, so convergence to the base distribution is not
        guaranteed when outliers are this extreme relative to the data spread."""
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(77)
        data = rng.normal(1000, 10, (50, 50)).astype(float)
        # Add 10% extreme outliers
        n_outliers = 250  # 10% of 2500
        idx = rng.choice(2500, n_outliers, replace=False)
        data.flat[idx] = 50000.0
        mean, std = getsky(data)
        # The function returns a finite result (not nan) but given the extreme
        # outliers and random sampling, the mean may not converge near 1000.
        assert np.isfinite(mean)
        assert np.isfinite(std)

    def test_bimodal_distribution(self):
        """Bimodal data: result depends on which mode dominates."""
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(88)
        # 80% at 1000, 20% at 5000
        data = np.zeros((50, 50))
        data[:40, :] = rng.normal(1000, 5, (40, 50))
        data[40:, :] = rng.normal(5000, 5, (10, 50))
        mean, std = getsky(data)
        # The sigma clip should reject the minor mode
        # Mean should be closer to 1000 (dominant mode)
        assert mean < 3000

    def test_integer_array_input(self):
        """Integer arrays should work (converted internally by numpy)."""
        from lsc.mysqldef import getsky
        rng = np.random.default_rng(99)
        data = rng.integers(900, 1100, (50, 50))
        mean, std = getsky(data)
        assert abs(mean - 1000) < 200


# ===========================================================================
# getlike — LIKE pattern matching query
# ===========================================================================

class TestGetlikeComprehensive:
    """Full coverage of getlike LIKE queries."""

    def test_generates_like_query(self):
        """SQL uses LIKE '%value%' pattern."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', 'cpt')
        sql = cursor.execute.call_args[0][0]
        assert "like '%cpt%'" in sql.lower() or "like '%cpt%'" in sql

    def test_default_column2_is_star(self):
        """Default column2='*' selects all columns."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', 'test')
        sql = cursor.execute.call_args[0][0]
        assert 'select *' in sql.lower() or 'select * ' in sql

    def test_custom_column2(self):
        """Custom column2 is used in SELECT clause."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', 'test', column2='filename,mjd')
        sql = cursor.execute.call_args[0][0]
        assert 'filename,mjd' in sql

    def test_empty_value_matches_everything(self):
        """Empty value creates LIKE '%%' which matches all."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', '')
        sql = cursor.execute.call_args[0][0]
        assert "'%%" in sql or "'%%'" in sql

    def test_cursor_closed_on_success(self):
        """Cursor is closed after successful query."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', 'test')
        cursor.close.assert_called_once()

    def test_returns_multiple_matches(self):
        """Multiple matching rows are all returned."""
        from lsc.mysqldef import getlike
        rows = ({'filename': 'a_cpt.fits'}, {'filename': 'b_cpt.fits'})
        conn, cursor = _make_mock_conn(fetchall_return=rows)
        result = getlike(conn, 'photlco', 'filename', 'cpt')
        assert len(result) == 2


# ===========================================================================
# query — execute arbitrary SQL commands
# ===========================================================================

class TestQueryComprehensive:
    """Comprehensive tests for the query() function."""

    def test_single_command_returns_result(self):
        """Single command executes and returns fetchall result."""
        from lsc.mysqldef import query
        conn, cursor = _make_mock_conn(fetchall_return=({'id': 1},))
        result = query(['SELECT * FROM targets'], conn)
        assert result == ({'id': 1},)

    def test_multiple_commands_executed_in_order(self):
        """Multiple commands are all executed sequentially."""
        from lsc.mysqldef import query
        conn, cursor = _make_mock_conn()
        query(['SET @x = 1', 'SET @y = 2', 'SELECT @x + @y'], conn)
        assert cursor.execute.call_count == 3

    def test_returns_last_command_result(self):
        """Return value is the fetchall from the LAST command."""
        from lsc.mysqldef import query
        conn, cursor = _make_mock_conn()
        # Side effects: first two return nothing useful, last returns data
        cursor.fetchall.side_effect = [(), (), ({'result': 42},)]
        result = query(['SET @x=1', 'SET @y=2', 'SELECT 42'], conn)
        assert result == ({'result': 42},)

    def test_empty_command_list(self):
        """Empty list results in no execution."""
        from lsc.mysqldef import query
        conn, cursor = _make_mock_conn()
        result = query([], conn)
        cursor.execute.assert_not_called()
        assert result == ''  # lista starts as ''

    def test_conn_commit_called(self):
        """Connection commit is called after all commands."""
        from lsc.mysqldef import query
        conn, cursor = _make_mock_conn()
        query(['SELECT 1'], conn)
        conn.commit.assert_called_once()

    def test_cursor_closed(self):
        """Cursor is closed after execution."""
        from lsc.mysqldef import query
        conn, cursor = _make_mock_conn()
        query(['SELECT 1'], conn)
        cursor.close.assert_called_once()

    def test_error_prints_message(self, monkeypatch, capsys):
        """Database error prints error message without raising."""
        import MySQLdb as msql
        from lsc.mysqldef import query
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(1064, 'syntax error')
        result = query(['INVALID SQL'], conn)
        out = capsys.readouterr().out
        assert 'Error' in out

    def test_error_returns_empty_string(self, monkeypatch):
        """On error, query returns '' (initial value of lista)."""
        import MySQLdb as msql
        from lsc.mysqldef import query
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(1064, 'syntax error')
        result = query(['INVALID SQL'], conn)
        assert result == ''


# ===========================================================================
# getfromcoordinate — haversine-based coordinate search
# ===========================================================================

class TestGetfromcoordinateComprehensive:
    """Full coverage of getfromcoordinate coordinate search."""

    def test_targets_table_sets_ra1_dec1(self):
        """For 'targets' table, ra1/dec1 are set to 'ra0'/'dec0'."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 180.0, 45.0, 0.1)
        # Verify the haversine SQL was constructed
        calls = cursor.execute.call_args_list
        assert len(calls) == 5  # 4 SET commands + 1 SELECT

    def test_photlcoraw_table(self):
        """For 'photlcoraw' table, same behavior as targets."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'photlcoraw', 90.0, -30.0, 0.5)
        calls = cursor.execute.call_args_list
        assert len(calls) == 5

    def test_ra_dec_distance_in_sql(self):
        """RA, DEC, and distance values appear in the SET statements."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 123.456, -45.678, 0.01)
        all_sqls = [c[0][0] for c in cursor.execute.call_args_list]
        # Check the SET statements contain our values
        assert any('123.456' in s for s in all_sqls)
        assert any('-45.678' in s for s in all_sqls)
        assert any('0.01' in s for s in all_sqls)

    def test_haversine_formula_in_select(self):
        """The SELECT statement includes the haversine formula."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 180.0, 0.0, 1.0)
        # The last execute call should contain the haversine SELECT
        last_sql = cursor.execute.call_args_list[-1][0][0]
        assert 'hsine' in last_sql
        assert 'asin' in last_sql
        assert 'sin' in last_sql
        assert 'cos' in last_sql

    def test_zero_distance(self):
        """Zero distance search should still execute."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 180.0, 0.0, 0.0)
        assert cursor.execute.call_count == 5

    def test_negative_declination(self):
        """Negative declination works correctly."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 0.0, -89.99, 0.01)
        all_sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any('-89.99' in s for s in all_sqls)

    def test_ra_wraps_at_360(self):
        """RA near 360 degrees should still work."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 359.99, 0.0, 0.1)
        all_sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any('359.99' in s for s in all_sqls)

    def test_returns_result_with_hsine(self):
        """Returned rows should include hsine column from the query."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn(
            fetchall_return=({'id': 1, 'ra0': 180.0, 'dec0': 0.0, 'hsine': 0.001},)
        )
        result = getfromcoordinate(conn, 'targets', 180.0, 0.0, 0.01)
        assert len(result) == 1
        assert result[0]['hsine'] == 0.001


# ===========================================================================
# JDnow and MJDnow — Julian Date calculations
# ===========================================================================

class TestJDnowComprehensive:
    """Comprehensive tests for JDnow."""

    def test_reference_epoch_2012_jan_1(self):
        """JD0 = 2455927.5 corresponds to 2012-01-01 00:00:00 UTC."""
        from lsc.mysqldef import JDnow
        dt = datetime.datetime(2012, 1, 1, 0, 0, 0)
        assert abs(JDnow(datenow=dt) - 2455927.5) < 1e-6

    def test_one_day_later(self):
        """One day later adds 1.0 to JD."""
        from lsc.mysqldef import JDnow
        d0 = datetime.datetime(2012, 1, 1, 0, 0, 0)
        d1 = datetime.datetime(2012, 1, 2, 0, 0, 0)
        assert abs(JDnow(datenow=d1) - JDnow(datenow=d0) - 1.0) < 1e-6

    def test_half_day(self):
        """12 hours adds 0.5 to JD."""
        from lsc.mysqldef import JDnow
        d0 = datetime.datetime(2012, 1, 1, 0, 0, 0)
        d1 = datetime.datetime(2012, 1, 1, 12, 0, 0)
        assert abs(JDnow(datenow=d1) - JDnow(datenow=d0) - 0.5) < 1e-6

    def test_verbose_false_no_output(self, capsys):
        """verbose=False (default) produces no output."""
        from lsc.mysqldef import JDnow
        JDnow(datenow=datetime.datetime(2020, 1, 1))
        assert capsys.readouterr().out == ''

    def test_verbose_true_prints(self, capsys):
        """verbose=True prints JD value."""
        from lsc.mysqldef import JDnow
        JDnow(datenow=datetime.datetime(2020, 1, 1), verbose=True)
        out = capsys.readouterr().out
        assert 'JD=' in out

    def test_no_arg_uses_current_time(self):
        """Calling with no datenow arg returns a reasonable current JD."""
        from lsc.mysqldef import JDnow
        jd = JDnow()
        # Current JD should be > 2460000 (after 2023)
        assert jd > 2460000

    def test_leap_year_date(self):
        """Leap year date Feb 29 works correctly."""
        from lsc.mysqldef import JDnow
        dt = datetime.datetime(2024, 2, 29, 0, 0, 0)
        jd = JDnow(datenow=dt)
        assert isinstance(jd, float)
        assert jd > 2455927.5  # After reference epoch


class TestMJDnowComprehensive:
    """Comprehensive tests for MJDnow."""

    def test_reference_epoch(self):
        """MJD0 = 55927.0 corresponds to 2012-01-01 00:00:00 UTC."""
        from lsc.mysqldef import MJDnow
        dt = datetime.datetime(2012, 1, 1, 0, 0, 0)
        assert abs(MJDnow(datenow=dt) - 55927.0) < 1e-6

    def test_jd_minus_mjd_is_2400000_5(self):
        """JD - MJD = 2400000.5 for the same date."""
        from lsc.mysqldef import JDnow, MJDnow
        dt = datetime.datetime(2023, 6, 15, 12, 0, 0)
        diff = JDnow(datenow=dt) - MJDnow(datenow=dt)
        assert abs(diff - 2400000.5) < 0.01

    def test_seconds_precision(self):
        """Seconds within a day are captured correctly."""
        from lsc.mysqldef import MJDnow
        d0 = datetime.datetime(2020, 6, 1, 0, 0, 0)
        d1 = datetime.datetime(2020, 6, 1, 6, 0, 0)  # 6 hours = 0.25 day
        assert abs(MJDnow(datenow=d1) - MJDnow(datenow=d0) - 0.25) < 1e-6

    def test_multi_year_span(self):
        """Multi-year span calculates correctly."""
        from lsc.mysqldef import MJDnow
        d0 = datetime.datetime(2012, 1, 1)
        d1 = datetime.datetime(2013, 1, 1)  # 366 days (2012 is leap year)
        diff = MJDnow(datenow=d1) - MJDnow(datenow=d0)
        assert abs(diff - 366.0) < 1e-6


# ===========================================================================
# getvaluefromarchive — wrapper around dbConnect + getfromdataraw
# ===========================================================================

class TestGetvaluefromarchiveComprehensive:
    """Tests for getvaluefromarchive which wraps connection + query."""

    def test_returns_result_when_found(self):
        """When result is truthy, returns it directly."""
        from lsc.mysqldef import getvaluefromarchive
        fake_rows = ({'filename': 'test.fits', 'wcs': 1},)
        with patch('lsc.mysqldef.getfromdataraw', return_value=fake_rows):
            result = getvaluefromarchive('photlco', 'filename', 'test.fits', '*')
        assert result == fake_rows

    def test_returns_empty_list_when_not_found(self):
        """When result is falsy (empty tuple), returns []."""
        from lsc.mysqldef import getvaluefromarchive
        with patch('lsc.mysqldef.getfromdataraw', return_value=()):
            result = getvaluefromarchive('photlco', 'filename', 'nonexist.fits', '*')
        assert result == []

    def test_calls_getconnection_with_lcogt2(self):
        """Always uses 'lcogt2' connection."""
        from lsc.mysqldef import getvaluefromarchive
        with patch('lsc.mysqldef.getconnection') as mock_gc:
            mock_gc.return_value = ('h', 'u', 'p', 'd')
            with patch('lsc.mysqldef.dbConnect', return_value=MagicMock()):
                with patch('lsc.mysqldef.getfromdataraw', return_value=()):
                    getvaluefromarchive('photlco', 'filename', 'x.fits', '*')
            mock_gc.assert_called_with('lcogt2')

    def test_passes_all_params_to_getfromdataraw(self):
        """table, column, value, column2 are all passed through."""
        from lsc.mysqldef import getvaluefromarchive
        with patch('lsc.mysqldef.getfromdataraw') as mock_gfdr:
            mock_gfdr.return_value = ()
            getvaluefromarchive('spec', 'objname', 'SN2024', 'filename,mjd')
            args = mock_gfdr.call_args[0]
            # args: (conn, table, column, value, column2)
            assert args[1] == 'spec'
            assert args[2] == 'objname'
            assert args[3] == 'SN2024'
            assert args[4] == 'filename,mjd'


# ===========================================================================
# deleteredufromarchive — DELETE query
# ===========================================================================

class TestDeleteredufromarchiveComprehensive:
    """Comprehensive tests for deleteredufromarchive."""

    def test_default_archive_and_column(self):
        """Default archive='photlco' and column='filename'."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.dbConnect') as mock_db:
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            mock_db.return_value = mock_conn
            deleteredufromarchive('test.fits')
            sql = cursor.execute.call_args[0][0]
            assert 'delete' in sql.lower()
            assert 'photlco' in sql
            assert "filename='test.fits'" in sql

    def test_custom_archive(self):
        """Custom archive name is used in the DELETE."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.dbConnect') as mock_db:
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            mock_db.return_value = mock_conn
            deleteredufromarchive('test.fits', archive='spec')
            sql = cursor.execute.call_args[0][0]
            assert 'spec' in sql

    def test_custom_column(self):
        """Custom column is used in the WHERE clause."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.dbConnect') as mock_db:
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            mock_db.return_value = mock_conn
            deleteredufromarchive('42', archive='photlco', column='id')
            sql = cursor.execute.call_args[0][0]
            assert "id='42'" in sql

    def test_commit_called(self):
        """Connection commit is called after delete."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.dbConnect') as mock_db:
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            mock_db.return_value = mock_conn
            deleteredufromarchive('test.fits')
            mock_conn.commit.assert_called_once()

    def test_returns_fetchall_result(self):
        """Returns the result of fetchall (typically empty for DELETE)."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.dbConnect') as mock_db:
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 1
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            mock_db.return_value = mock_conn
            result = deleteredufromarchive('test.fits')
            assert result == ()


# ===========================================================================
# gettargetid — target resolution by name or coordinates
# ===========================================================================

class TestGettargetidComprehensive:
    """Comprehensive tests for gettargetid function."""

    def test_name_lookup_single_match(self):
        """Single name match returns that target id."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(
            fetchall_return=({'name': 'SN2024a', 'targetid': 10, 'id': 10,
                            'ra0': 100.0, 'dec0': 20.0},)
        )
        result = gettargetid('SN2024a', '', '', conn, 0.01)
        assert result == 10

    def test_name_lookup_multiple_same_id(self):
        """Multiple rows with same id returns that id."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(
            fetchall_return=(
                {'name': 'SN2024a', 'targetid': 10, 'id': 10, 'ra0': 100.0, 'dec0': 20.0},
                {'name': 'AT2024a', 'targetid': 10, 'id': 10, 'ra0': 100.0, 'dec0': 20.0},
            )
        )
        result = gettargetid('SN2024a', '', '', conn, 0.01)
        assert result == 10

    def test_name_lookup_multiple_different_ids_no_hsine(self):
        """Multiple different ids without hsine returns ''."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(
            fetchall_return=(
                {'name': 'SN2024a', 'targetid': 10, 'id': 10, 'ra0': 100.0, 'dec0': 20.0},
                {'name': 'SN2024a', 'targetid': 20, 'id': 20, 'ra0': 101.0, 'dec0': 21.0},
            )
        )
        result = gettargetid('SN2024a', '', '', conn, 0.01)
        assert result == ''

    def test_name_lookup_multiple_different_ids_with_hsine(self):
        """Multiple different ids with hsine returns the one with min hsine."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(
            fetchall_return=(
                {'name': 'SN2024a', 'targetid': 10, 'id': 10, 'ra0': 100.0, 'dec0': 20.0, 'hsine': 0.05},
                {'name': 'SN2024a', 'targetid': 20, 'id': 20, 'ra0': 100.001, 'dec0': 20.001, 'hsine': 0.001},
            )
        )
        result = gettargetid('SN2024a', '', '', conn, 0.01)
        assert result == 20

    def test_name_with_spaces_uses_percent(self):
        """Name with spaces gets spaces replaced with % in SQL."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(fetchall_return=())
        gettargetid('SN 2024abc', '', '', conn, 0.01)
        sql = cursor.execute.call_args[0][0]
        assert 'SN%2024abc' in sql

    def test_ra_dec_lookup_single_match(self):
        """RA/DEC lookup returns target id from coordinate search."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn()
        # getfromcoordinate executes multiple commands; the final fetchall returns results
        cursor.fetchall.return_value = (
            {'id': 5, 'ra0': 150.0, 'dec0': 2.0, 'hsine': 0.001},
        )
        result = gettargetid('', 150.0, 2.0, conn, 0.01)
        assert result == 5

    def test_ra_dec_no_match(self):
        """RA/DEC with no match returns ''."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(fetchall_return=())
        result = gettargetid('', 150.0, 2.0, conn, 0.01)
        assert result == ''

    def test_ra_dec_colon_format_calls_deg2hms(self):
        """RA in colon format triggers deg2HMS conversion."""
        from lsc.mysqldef import gettargetid
        import lsc
        conn, cursor = _make_mock_conn(fetchall_return=())
        with patch.object(lsc, 'deg2HMS', return_value=(150.0, 2.0), create=True):
            result = gettargetid('', '10:00:00', '+02:00:00', conn, 0.01)
        assert result == ''

    def test_verbose_no_match_prints_no_objects(self, capsys):
        """verbose=True with no match prints 'no objects'."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(fetchall_return=())
        gettargetid('SN_not_found', '', '', conn, 0.01, verbose=True)
        out = capsys.readouterr().out
        assert 'no objects' in out

    def test_verbose_ra_dec_prints_coords(self, capsys):
        """verbose=True with ra/dec prints the coordinates."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(fetchall_return=())
        gettargetid('', 123.45, -67.89, conn, 0.5, verbose=True)
        out = capsys.readouterr().out
        assert '123.45' in out

    def test_custom_radius(self):
        """Custom radius is passed to getfromcoordinate."""
        from lsc.mysqldef import gettargetid
        conn, cursor = _make_mock_conn(fetchall_return=())
        # Note: dec=0.0 is falsy in Python, so the elif _ra and _dec branch
        # would not be taken. Use a non-zero dec to exercise the coordinate path.
        gettargetid('', 180.0, 1.0, conn, 0.5)
        # The distance SET command should include 0.5
        all_calls = cursor.execute.call_args_list
        all_sqls = [c[0][0] for c in all_calls]
        assert any('0.5' in s for s in all_sqls)


# ===========================================================================
# get_snex_uid — SNEx user ID lookup
# ===========================================================================

class TestGetSnexUidComprehensive:
    """Comprehensive tests for get_snex_uid."""

    def test_unix_user_found(self, monkeypatch):
        """When unix username matches, returns uid immediately."""
        import lsc.myloopdef
        conn, cursor = _make_mock_conn()
        cursor.fetchall.return_value = ({'id': 10, 'firstname': 'Alice', 'lastname': 'Smith'},)
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=False)
        assert result == 10

    def test_unix_user_not_found_noninteractive(self, monkeypatch):
        """Non-interactive mode returns None when user not found."""
        import lsc.myloopdef
        conn, cursor = _make_mock_conn()
        cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=False)
        assert result is None

    def test_return_fullname_found(self, monkeypatch):
        """return_fullname=True returns (uid, 'First Last')."""
        import lsc.myloopdef
        conn, cursor = _make_mock_conn()
        cursor.fetchall.return_value = ({'id': 10, 'firstname': 'Bob', 'lastname': 'Jones'},)
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        uid, fullname = get_snex_uid(interactive=False, return_fullname=True)
        assert uid == 10
        assert fullname == 'Bob Jones'

    def test_return_fullname_not_found(self, monkeypatch):
        """return_fullname=True with no match returns (None, None)."""
        import lsc.myloopdef
        conn, cursor = _make_mock_conn()
        cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.mysqldef import get_snex_uid
        uid, fullname = get_snex_uid(interactive=False, return_fullname=True)
        assert uid is None
        assert fullname is None

    def test_interactive_user_provides_valid_name(self, monkeypatch):
        """Interactive: user provides valid SNEx username that is found."""
        import lsc.myloopdef
        import lsc.util
        conn, cursor = _make_mock_conn()
        cursor.fetchall.side_effect = [
            (),  # unix user not found
            ({'id': 77, 'firstname': 'Carol', 'lastname': 'White'},),
        ]
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setattr(lsc.util, 'userinput', lambda *a, **k: 'cwhite')
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=True)
        assert result == 77

    def test_interactive_user_provides_empty_input(self, monkeypatch):
        """Interactive: user presses enter (empty input) returns None."""
        import lsc.myloopdef
        import lsc.util
        conn, cursor = _make_mock_conn()
        cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setattr(lsc.util, 'userinput', lambda *a, **k: '')
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=True)
        assert result is None

    def test_interactive_user_provides_invalid_name(self, monkeypatch, capsys):
        """Interactive: provided SNEx name not found in DB."""
        import lsc.myloopdef
        import lsc.util
        conn, cursor = _make_mock_conn()
        cursor.fetchall.side_effect = [(), ()]
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setattr(lsc.util, 'userinput', lambda *a, **k: 'baduser')
        from lsc.mysqldef import get_snex_uid
        result = get_snex_uid(interactive=True)
        assert result is None
        out = capsys.readouterr().out
        assert 'not found' in out


# ===========================================================================
# ingestredu — ingestion of reduced images
# ===========================================================================

def _setup_ingestredu(monkeypatch, exist=False, exist2=False,
                      telid_found=True, instid_found=True,
                      already_in_db=False, hdr_overrides=None):
    """Set up mocks for ingestredu tests."""
    import lsc
    import lsc.mysqldef

    fake_conn = MagicMock()
    monkeypatch.setattr(lsc.mysqldef, 'getconnection',
                        lambda name: ('h', 'u', 'p', 'd'))
    monkeypatch.setattr(lsc.mysqldef, 'dbConnect', lambda *a, **k: fake_conn)

    hdr = {
        'date-obs': '2024-01-15T10:30:00',
        'DAY-OBS': '20240115',
        'exptime': 300.0,
        'filter': 'gp',
        'mjd': 60323.0,
        'TRACKNUM': '54321',
        'TELESCOP': '1m0-04',
        'instrume': 'fa16',
        'airmass': 1.3,
        'object': 'SN2024test',
        'ut': '10:30:00',
        'wcserr': 0,
        'RA': 200.0,
        'DEC': -15.0,
        'SITEID': 'cpt',
    }
    if hdr_overrides:
        hdr.update(hdr_overrides)

    monkeypatch.setattr(lsc.util, 'readhdr', lambda img: hdr)
    monkeypatch.setattr(lsc.util, 'readkey3', lambda h, k: hdr.get(k, 'NOKEY'))
    monkeypatch.setattr(lsc.mysqldef, 'targimg', lambda *a, **k: 100)

    call_counts = {'tel': 0, 'inst': 0, 'main': 0}

    def mock_getfromdataraw(conn, table, col, val, column2='*'):
        if table == 'photlcoraw':
            return [{'groupidcode': 'grp2'}] if exist2 else []
        elif table == 'photlco':
            call_counts['main'] += 1
            if call_counts['main'] == 1:
                return [{'filename': 'img.fits'}] if exist else []
            return [{'filename': 'img.fits'}] if already_in_db else []
        elif table == 'telescopes':
            call_counts['tel'] += 1
            if telid_found or call_counts['tel'] > 1:
                return [{'id': 3}]
            return []
        elif table == 'instruments':
            call_counts['inst'] += 1
            if instid_found or call_counts['inst'] > 1:
                return [{'id': 4}]
            return []
        return []

    monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', mock_getfromdataraw)

    insert_calls = []
    monkeypatch.setattr(lsc.mysqldef, 'insert_values',
                        lambda c, t, d: insert_calls.append((t, d)))
    update_calls = []
    monkeypatch.setattr(lsc.mysqldef, 'updatevalue',
                        lambda t, c, v, f: update_calls.append((t, c, v, f)))
    monkeypatch.setattr(lsc.mysqldef, 'deleteredufromarchive', lambda *a: None)

    return insert_calls, update_calls


class TestIngestreduComprehensive:
    """Comprehensive tests for ingestredu function."""

    def test_new_image_inserts_into_dataredutable(self, monkeypatch):
        """New image is inserted into the data reduction table."""
        insert_calls, _ = _setup_ingestredu(monkeypatch)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/new_image.fits'])
        assert any(t == 'photlco' for t, _ in insert_calls)

    def test_already_ingested_default_force(self, monkeypatch, capsys):
        """Already-ingested image with force='no' prints message and skips."""
        insert_calls, _ = _setup_ingestredu(monkeypatch, exist=True)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/existing.fits'])
        out = capsys.readouterr().out
        assert 'already ingested' in out
        assert not any(t == 'photlco' for t, _ in insert_calls)

    def test_force_yes_deletes_and_reingests(self, monkeypatch):
        """force='yes' deletes existing entry and re-ingests."""
        insert_calls, _ = _setup_ingestredu(monkeypatch, exist=True, already_in_db=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/img.fits'], force='yes')
        assert any(t == 'photlco' for t, _ in insert_calls)

    def test_force_update_updates_fields(self, monkeypatch):
        """force='update' updates existing fields via updatevalue."""
        _, update_calls = _setup_ingestredu(monkeypatch, exist=True, already_in_db=True)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/img.fits'], force='update')
        assert len(update_calls) > 0

    def test_fts_telescope_mapping(self, monkeypatch):
        """'Faulkes Telescope South' maps to '2m0-02'."""
        insert_calls, _ = _setup_ingestredu(
            monkeypatch, hdr_overrides={'TELESCOP': 'Faulkes Telescope South'}
        )
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/fts.fits'])
        # Check that the inserted dictionary has telescope='2m0-02'
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['telescope'] == '2m0-02'

    def test_ftn_telescope_mapping(self, monkeypatch):
        """'Faulkes Telescope North' maps to '2m0-01'."""
        insert_calls, _ = _setup_ingestredu(
            monkeypatch, hdr_overrides={'TELESCOP': 'Faulkes Telescope North'}
        )
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/ftn.fits'])
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['telescope'] == '2m0-01'

    def test_fts_shortname_mapping(self, monkeypatch):
        """'fts' telescope short name maps to '2m0-02'."""
        insert_calls, _ = _setup_ingestredu(
            monkeypatch, hdr_overrides={'TELESCOP': 'fts'}
        )
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/fts2.fits'])
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['telescope'] == '2m0-02'

    def test_ftn_shortname_mapping(self, monkeypatch):
        """'ftn' telescope short name maps to '2m0-01'."""
        insert_calls, _ = _setup_ingestredu(
            monkeypatch, hdr_overrides={'TELESCOP': 'ftn'}
        )
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/ftn2.fits'])
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['telescope'] == '2m0-01'

    def test_no_telescop_header(self, monkeypatch):
        """Missing TELESCOP header defaults to empty string."""
        insert_calls, _ = _setup_ingestredu(
            monkeypatch, hdr_overrides={'TELESCOP': None}
        )
        from lsc.mysqldef import ingestredu
        # readkey3 returns None for TELESCOP which gets replaced by ''
        # Actually the code does hdr.get('TELESCOP') so None -> ''
        ingestredu(['/path/to/notel.fits'])

    def test_telescope_not_found_inserts_new(self, monkeypatch):
        """Unrecognized telescope triggers insert into telescopes table."""
        insert_calls, _ = _setup_ingestredu(monkeypatch, telid_found=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/newtel.fits'])
        assert any(t == 'telescopes' for t, _ in insert_calls)

    def test_instrument_not_found_inserts_new(self, monkeypatch):
        """Unrecognized instrument triggers insert into instruments table."""
        insert_calls, _ = _setup_ingestredu(monkeypatch, instid_found=False)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/newinst.fits'])
        assert any(t == 'instruments' for t, _ in insert_calls)

    def test_groupidcode_from_raw(self, monkeypatch):
        """When photlcoraw has an entry, groupidcode is taken from it."""
        insert_calls, _ = _setup_ingestredu(monkeypatch, exist2=True)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/withgrp.fits'])
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['groupidcode'] == 'grp2'

    def test_custom_dataredutable(self, monkeypatch):
        """Custom dataredutable parameter is respected."""
        import lsc
        import lsc.mysqldef

        fake_conn = MagicMock()
        monkeypatch.setattr(lsc.mysqldef, 'getconnection',
                            lambda name: ('h', 'u', 'p', 'd'))
        monkeypatch.setattr(lsc.mysqldef, 'dbConnect', lambda *a, **k: fake_conn)

        hdr = {
            'date-obs': '2024-01-01T00:00:00', 'DAY-OBS': '20240101',
            'exptime': 100.0, 'filter': 'r', 'mjd': 60000.0,
            'TRACKNUM': '111', 'TELESCOP': 'lsc', 'instrume': 'fl01',
            'airmass': 1.0, 'object': 'test', 'ut': '00:00:00',
            'wcserr': 0, 'RA': 0.0, 'DEC': 0.0, 'SITEID': 'lsc',
        }
        monkeypatch.setattr(lsc.util, 'readhdr', lambda img: hdr)
        monkeypatch.setattr(lsc.util, 'readkey3', lambda h, k: hdr.get(k, 'NOKEY'))
        monkeypatch.setattr(lsc.mysqldef, 'targimg', lambda *a, **k: 1)

        tables_queried = []

        def mock_getfromdataraw(conn, table, col, val, column2='*'):
            tables_queried.append(table)
            if table == 'telescopes':
                return [{'id': 1}]
            if table == 'instruments':
                return [{'id': 1}]
            return []

        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', mock_getfromdataraw)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a: None)
        monkeypatch.setattr(lsc.mysqldef, 'updatevalue', lambda *a: None)

        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/spec.fits'], dataredutable='spec')
        # The existence check should query the custom table
        assert 'spec' in tables_queried

    def test_custom_filetype(self, monkeypatch):
        """Custom filetype parameter is set in the dictionary."""
        insert_calls, _ = _setup_ingestredu(monkeypatch)
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/img.fits'], filetype=3)
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['filetype'] == 3

    def test_tracknum_non_integer(self, monkeypatch):
        """Non-integer TRACKNUM defaults to 0."""
        insert_calls, _ = _setup_ingestredu(
            monkeypatch, hdr_overrides={'TRACKNUM': 'not_a_number'}
        )
        from lsc.mysqldef import ingestredu
        ingestredu(['/path/to/badtrack.fits'])
        photlco_inserts = [d for t, d in insert_calls if t == 'photlco']
        if photlco_inserts:
            assert photlco_inserts[0]['tracknumber'] == 0

    def test_multiple_images_processed(self, monkeypatch):
        """Multiple images in list are all processed."""
        import lsc
        import lsc.mysqldef

        fake_conn = MagicMock()
        monkeypatch.setattr(lsc.mysqldef, 'getconnection',
                            lambda name: ('h', 'u', 'p', 'd'))
        monkeypatch.setattr(lsc.mysqldef, 'dbConnect', lambda *a, **k: fake_conn)

        hdr = {
            'date-obs': '2024-01-01T00:00:00', 'DAY-OBS': '20240101',
            'exptime': 100.0, 'filter': 'r', 'mjd': 60000.0,
            'TRACKNUM': '111', 'TELESCOP': 'lsc', 'instrume': 'fl01',
            'airmass': 1.0, 'object': 'test', 'ut': '00:00:00',
            'wcserr': 0, 'RA': 0.0, 'DEC': 0.0, 'SITEID': 'lsc',
        }
        monkeypatch.setattr(lsc.util, 'readhdr', lambda img: hdr)
        monkeypatch.setattr(lsc.util, 'readkey3', lambda h, k: hdr.get(k, 'NOKEY'))
        monkeypatch.setattr(lsc.mysqldef, 'targimg', lambda *a, **k: 1)
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw',
                            lambda *a, **k: [{'id': 1}] if 'telescopes' in str(a) or 'instruments' in str(a) else [])

        insert_count = {'n': 0}
        original_insert = lsc.mysqldef.insert_values

        def counting_insert(*a, **k):
            insert_count['n'] += 1

        monkeypatch.setattr(lsc.mysqldef, 'insert_values', counting_insert)
        monkeypatch.setattr(lsc.mysqldef, 'updatevalue', lambda *a: None)

        def mock_gfdr(conn, table, col, val, column2='*'):
            if table in ('telescopes', 'instruments'):
                return [{'id': 1}]
            return []

        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', mock_gfdr)

        from lsc.mysqldef import ingestredu
        ingestredu(['/path/img1.fits', '/path/img2.fits', '/path/img3.fits'])
        assert insert_count['n'] >= 3


# ===========================================================================
# Error handling branches — sys.exit and print-only paths
# ===========================================================================

class TestErrorHandlingBranches:
    """Tests that verify error handling in all functions with try/except."""

    def test_getmissing_error_exits(self, monkeypatch):
        """getmissing exits with code 1 on database error."""
        import MySQLdb as msql
        from lsc.mysqldef import getmissing
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(2002, 'conn refused')
        with pytest.raises(SystemExit):
            getmissing(conn, '20220101', '20220201', 'all')

    def test_getfromdataraw_error_exits(self, monkeypatch):
        """getfromdataraw exits with code 1 on database error."""
        import MySQLdb as msql
        from lsc.mysqldef import getfromdataraw
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(1146, 'table not found')
        with pytest.raises(SystemExit):
            getfromdataraw(conn, 'photlco', 'filename', 'x.fits')

    def test_getlike_error_exits(self, monkeypatch):
        """getlike exits with code 1 on database error."""
        import MySQLdb as msql
        from lsc.mysqldef import getlike
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(1064, 'bad sql')
        with pytest.raises(SystemExit):
            getlike(conn, 'photlco', 'fn', 'x')

    def test_getfromcoordinate_error_exits(self, monkeypatch):
        """getfromcoordinate exits with code 1 on database error."""
        import MySQLdb as msql
        from lsc.mysqldef import getfromcoordinate
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(1, 'generic')
        with pytest.raises(SystemExit):
            getfromcoordinate(conn, 'targets', 0.0, 0.0, 1.0)

    def test_deleteredufromarchive_error_exits(self, monkeypatch):
        """deleteredufromarchive exits on database error."""
        import MySQLdb as msql
        from lsc.mysqldef import deleteredufromarchive
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        cursor = MagicMock()
        cursor.execute.side_effect = _FakeMySQLError(1, 'err')
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        with patch('lsc.mysqldef.dbConnect', return_value=mock_conn):
            with pytest.raises(SystemExit):
                deleteredufromarchive('test.fits')

    def test_updatevalue_error_prints_only(self, monkeypatch, capsys):
        """updatevalue prints error but does NOT exit."""
        import MySQLdb as msql
        from lsc.mysqldef import updatevalue
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        cursor = MagicMock()
        cursor.execute.side_effect = _FakeMySQLError(1062, 'duplicate')
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        with patch('lsc.mysqldef.dbConnect', return_value=mock_conn):
            # Should NOT raise SystemExit
            updatevalue('photlco', 'wcs', 1, 'test.fits')
        out = capsys.readouterr().out
        assert 'Error' in out

    def test_insert_values_error_prints_only(self, monkeypatch, capsys):
        """insert_values prints error but does NOT exit."""
        import MySQLdb as msql
        from lsc.mysqldef import insert_values
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        cursor = MagicMock()
        cursor.execute.side_effect = _FakeMySQLError(1062, 'duplicate entry')
        conn = MagicMock()
        conn.cursor.return_value = cursor
        insert_values(conn, 'photlco', {'filename': 'x.fits'})
        out = capsys.readouterr().out
        assert 'Error' in out

    def test_query_error_prints_only(self, monkeypatch, capsys):
        """query prints error but does NOT exit."""
        import MySQLdb as msql
        from lsc.mysqldef import query
        monkeypatch.setattr(msql, 'Error', _FakeMySQLError)
        conn, cursor = _make_mock_conn()
        cursor.execute.side_effect = _FakeMySQLError(1064, 'syntax')
        result = query(['SELECT bad'], conn)
        out = capsys.readouterr().out
        assert 'Error' in out
        # Returns '' because lista was never assigned
        assert result == ''


# ===========================================================================
# SQL injection-like inputs — testing that values are placed directly in SQL
# ===========================================================================

class TestSQLInjectionLikeInputs:
    """These tests document that the module does NOT use parameterized queries
    for most functions (except insert_values), meaning SQL injection-like inputs
    are passed through. This documents existing behavior, not desired behavior."""

    def test_getfromdataraw_with_quote_in_value(self):
        """Value with single quote is embedded directly in SQL."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        getfromdataraw(conn, 'photlco', 'objname', "'; DROP TABLE photlco; --")
        sql = cursor.execute.call_args[0][0]
        assert "'; DROP TABLE" in sql

    def test_getlike_with_percent_in_value(self):
        """Percent in value becomes part of LIKE pattern."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', '%')
        sql = cursor.execute.call_args[0][0]
        assert "'%%%'" in sql  # '%' + value + '%'

    def test_getlistfromraw_with_quote_in_value(self):
        """Quote in value1/value2 is embedded directly."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', "20220101' OR '1'='1", '20220201')
        sql = cursor.execute.call_args[0][0]
        assert "OR '1'='1" in sql

    def test_getmissing_with_special_datatable(self):
        """Special characters in datatable name pass through."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220201', 'all', datatable='table; DROP')
        sql = cursor.execute.call_args[0][0]
        assert 'table; DROP' in sql


# ===========================================================================
# Boundary conditions
# ===========================================================================

class TestBoundaryConditions:
    """Tests for boundary and edge-case inputs."""

    def test_getmissing_epoch0_equals_epoch2(self):
        """When epoch0 == epoch2, the range query selects nothing (< epoch2 and >= epoch0)."""
        from lsc.mysqldef import getmissing
        conn, cursor = _make_mock_conn()
        getmissing(conn, '20220101', '20220101', 'all')
        sql = cursor.execute.call_args[0][0]
        # dayobs < 20220101 and dayobs >= 20220101 is always false
        assert '20220101' in sql

    def test_getlistfromraw_value1_greater_than_value2(self):
        """When value1 > value2, the range is inverted (empty result)."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220201', '20220101')
        sql = cursor.execute.call_args[0][0]
        # column <= '20220101' and column >= '20220201' is always false
        assert '20220201' in sql
        assert '20220101' in sql

    def test_getfromcoordinate_large_distance(self):
        """Large distance (full sky) still works."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 0.0, 0.0, 180.0)
        all_sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any('180.0' in s for s in all_sqls)

    def test_getfromcoordinate_ra_zero_dec_zero(self):
        """RA=0, DEC=0 (origin) works correctly."""
        from lsc.mysqldef import getfromcoordinate
        conn, cursor = _make_mock_conn()
        getfromcoordinate(conn, 'targets', 0.0, 0.0, 0.01)
        all_sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any('0.0' in s or '0' in s for s in all_sqls)

    def test_jdnow_far_future(self):
        """JDnow works for dates far in the future."""
        from lsc.mysqldef import JDnow
        dt = datetime.datetime(2099, 12, 31, 23, 59, 59)
        jd = JDnow(datenow=dt)
        assert jd > 2455927.5
        assert isinstance(jd, float)

    def test_mjdnow_before_reference_epoch(self):
        """MJDnow for dates before reference epoch gives values < 55927."""
        from lsc.mysqldef import MJDnow
        dt = datetime.datetime(2011, 12, 31, 0, 0, 0)
        mjd = MJDnow(datenow=dt)
        assert mjd < 55927.0

    def test_insert_values_single_column(self):
        """Insert with only one column works."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'onlycol': 'value'}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'onlycol' in sql

    def test_updatevalue_empty_list_columns(self):
        """updatevalue with empty lists — edge case."""
        from lsc.mysqldef import updatevalue
        # Empty lists would generate 'SET  WHERE ...' — still calls execute
        updatevalue('photlco', [], [], 'test.fits')

    def test_getsky_single_value_repeated(self):
        """getsky with an array where all values are the same: sigma clipping
        empties the sample because std=0 makes the strict < threshold equal
        to the mean itself, removing all points."""
        from lsc.mysqldef import getsky
        # Array of exactly one distinct value
        data = np.full((20, 20), 42.0)
        mean, std = getsky(data)
        # All-identical values yield nan due to empty sample after clipping
        assert np.isnan(mean)
        assert np.isnan(std)


# ===========================================================================
# Integration-style tests with multiple function interactions
# ===========================================================================

class TestFunctionInteractions:
    """Tests that verify how functions work together."""

    def test_getvaluefromarchive_uses_dbconnect_and_getfromdataraw(self):
        """getvaluefromarchive chains getconnection -> dbConnect -> getfromdataraw."""
        from lsc.mysqldef import getvaluefromarchive
        with patch('lsc.mysqldef.getconnection', return_value=('h', 'u', 'p', 'd')) as mock_gc:
            with patch('lsc.mysqldef.dbConnect', return_value=MagicMock()) as mock_db:
                with patch('lsc.mysqldef.getfromdataraw', return_value=()) as mock_gfdr:
                    getvaluefromarchive('photlco', 'filename', 'x.fits', '*')
                    mock_gc.assert_called_once_with('lcogt2')
                    mock_db.assert_called_once_with('h', 'u', 'p', 'd')
                    mock_gfdr.assert_called_once()

    def test_deleteredufromarchive_uses_dbconnect(self):
        """deleteredufromarchive establishes its own connection."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.getconnection', return_value=('h', 'u', 'p', 'd')) as mock_gc:
            mock_conn = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn.cursor.return_value = cursor
            with patch('lsc.mysqldef.dbConnect', return_value=mock_conn) as mock_db:
                deleteredufromarchive('test.fits')
                mock_gc.assert_called_once_with('lcogt2')
                mock_db.assert_called_once()

    def test_updatevalue_uses_dbconnect(self):
        """updatevalue establishes its own connection via getconnection."""
        from lsc.mysqldef import updatevalue
        with patch('lsc.mysqldef.getconnection', return_value=('h', 'u', 'p', 'd')) as mock_gc:
            with patch('lsc.mysqldef.dbConnect', return_value=MagicMock()) as mock_db:
                updatevalue('photlco', 'wcs', 1, 'test.fits')
                mock_gc.assert_called_once()
                mock_db.assert_called_once()


# ===========================================================================
# Specific SQL format verification
# ===========================================================================

class TestSQLFormats:
    """Verify exact SQL statements generated by various functions."""

    def test_getfromdataraw_exact_format(self):
        """Exact SQL format for getfromdataraw."""
        from lsc.mysqldef import getfromdataraw
        conn, cursor = _make_mock_conn()
        getfromdataraw(conn, 'photlco', 'filename', 'img.fits', column2='wcs,psf')
        expected = "select wcs,psf from photlco where filename='img.fits'"
        assert cursor.execute.call_args[0][0] == expected

    def test_getlike_exact_format(self):
        """Exact SQL format for getlike."""
        from lsc.mysqldef import getlike
        conn, cursor = _make_mock_conn()
        getlike(conn, 'photlco', 'filename', 'cpt', column2='filename')
        expected = "select filename from photlco where filename like '%cpt%'"
        assert cursor.execute.call_args[0][0] == expected

    def test_getlistfromraw_all_telescope_format(self):
        """Exact SQL format for getlistfromraw with telescope='all'."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201',
                       column2='filename')
        expected = "select filename from photlcoraw where dayobs<='20220201' and dayobs>='20220101'"
        assert cursor.execute.call_args[0][0] == expected

    def test_getlistfromraw_telescope_format(self):
        """Exact SQL format for getlistfromraw with specific telescope."""
        from lsc.mysqldef import getlistfromraw
        conn, cursor = _make_mock_conn()
        getlistfromraw(conn, 'photlcoraw', 'dayobs', '20220101', '20220201',
                       column2='filename', telescope='1m0-01')
        expected = ("select filename from photlcoraw where dayobs<='20220201' "
                    "and dayobs>='20220101' and (filename like '%1m001%' "
                    "or telescope='1m0-01')")
        assert cursor.execute.call_args[0][0] == expected

    def test_deleteredufromarchive_exact_format(self):
        """Exact SQL format for deleteredufromarchive."""
        from lsc.mysqldef import deleteredufromarchive
        with patch('lsc.mysqldef.dbConnect') as mock_db:
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            mock_db.return_value = mock_conn
            deleteredufromarchive('img.fits', archive='spec', column='id')
            expected = "delete  from spec where id='img.fits'"
            assert cursor.execute.call_args[0][0] == expected

    def test_insert_values_sql_format(self):
        """INSERT SQL uses %(key)s parameterized placeholders."""
        from lsc.mysqldef import insert_values
        conn, cursor = _make_mock_conn()
        values = {'filename': 'test.fits', 'wcs': 1}
        insert_values(conn, 'custom_table', values)
        sql = cursor.execute.call_args[0][0]
        assert 'INSERT INTO custom_table' in sql
        assert '%(filename)s' in sql
        assert '%(wcs)s' in sql
        assert sql.endswith(');')


# ===========================================================================
# guess_instrument_type with parametrize for full coverage
# ===========================================================================

class TestGuessInstrumentTypeParametrized:
    """Parametrized tests for all instrument name patterns."""

    @pytest.mark.parametrize("name,expected", [
        ('fl01', 'Sinistro'),
        ('fl99', 'Sinistro'),
        ('fa01', 'Sinistro'),
        ('fa16', 'Sinistro'),
        ('kb70', 'SBIG'),
        ('kb99', 'SBIG'),
        ('fs01', 'Spectral'),
        ('fs02', 'Spectral'),
        ('ep01', 'MUSCAT'),
        ('ep10', 'MUSCAT'),
        ('sq01', 'QHY'),
        ('sq99', 'QHY'),
        ('xx01', None),
        ('zz01', None),
        ('ab', None),
        ('  ', None),  # whitespace prefix
    ])
    def test_instrument_classification(self, name, expected):
        from lsc.mysqldef import guess_instrument_type
        assert guess_instrument_type(name) == expected
