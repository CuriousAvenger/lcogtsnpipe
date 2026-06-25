"""
Tests targeting specific uncovered lines in lsc/mysqldef.py:
- pymysql fallback imports (when MySQLdb is not available)
- Line 246: os.remove(.fz) in ingestredu when funpack fails to remove
- Lines 365-435: the updateDatabase function
- Line 738: gettargetid with multiple ids and no 'hsine' key
"""
import sys
import types
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, mock_open
import importlib

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: reload mysqldef with MySQLdb removed to hit pymysql fallback
# ---------------------------------------------------------------------------

@contextmanager
def _reload_mysqldef_without_MySQLdb():
    """
    Temporarily remove MySQLdb from sys.modules so that when functions
    inside mysqldef do `try: import MySQLdb ... except: import pymysql`,
    the except branch is taken (hitting the pymysql fallback lines).

    Returns the reloaded module. Caller should restore state after.
    """
    # Save original state
    original_mysqldb = sys.modules.pop('MySQLdb', None)

    # Make MySQLdb import raise ImportError
    class _MySQLdbBlocker:
        """Import hook that blocks MySQLdb."""
        def find_module(self, name, path=None):
            if name == 'MySQLdb':
                return self
            return None
        def load_module(self, name):
            raise ImportError("Blocked MySQLdb for testing pymysql fallback")

    blocker = _MySQLdbBlocker()
    sys.meta_path.insert(0, blocker)

    try:
        # pymysql mock should still be available
        if 'pymysql' not in sys.modules:
            _pmysql_mock = MagicMock()
            _pmysql_mock.cursors = MagicMock()
            _pmysql_mock.cursors.DictCursor = MagicMock()
            _pmysql_mock.connect.return_value.cursor.return_value = MagicMock()
            sys.modules['pymysql'] = _pmysql_mock

        yield sys.modules['pymysql']
    finally:
        sys.meta_path.remove(blocker)
        # Restore MySQLdb
        if original_mysqldb is not None:
            sys.modules['MySQLdb'] = original_mysqldb


# ---------------------------------------------------------------------------
# Tests: pymysql fallback import paths
# ---------------------------------------------------------------------------

class TestPymysqlFallbackImports:
    """
    Each function in mysqldef.py has its own try/except import block.
    To hit the `except: import pymysql as msql` fallback, we need MySQLdb
    to fail to import. We test each function under that condition.
    """

    def test_dbConnect_pymysql_fallback(self):
        """Lines 9-10: dbConnect uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.connect.return_value = MagicMock()
            # Import the function fresh - it does its own internal import
            from lsc.mysqldef import dbConnect
            conn = dbConnect('localhost', 'user', 'pass', 'db')
            # The function should have used pymysql.connect
            assert conn is not None

    def test_getmissing_pymysql_fallback(self):
        """Lines 45-46: getmissing uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import getmissing
            result = getmissing(mock_conn, '20220101', '20220201', 'all')
            assert isinstance(result, tuple)

    def test_getfromdataraw_pymysql_fallback(self):
        """Lines 93-94: getfromdataraw uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import getfromdataraw
            result = getfromdataraw(mock_conn, 'photlco', 'filename', 'test.fits')
            assert isinstance(result, tuple)

    def test_getlistfromraw_pymysql_fallback(self):
        """Lines 112-113: getlistfromraw uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import getlistfromraw
            result = getlistfromraw(mock_conn, 'photlcoraw', 'dayobs', '20220101', '20220201')
            assert isinstance(result, tuple)

    def test_updatevalue_pymysql_fallback(self):
        """Lines 135-136: updatevalue uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import updatevalue
            with patch('lsc.mysqldef.dbConnect', return_value=mock_conn):
                updatevalue('photlco', 'wcs', 1, 'test.fits')

    def test_insert_values_pymysql_fallback(self):
        """Lines 170-171: insert_values uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import insert_values
            insert_values(mock_conn, 'photlco', {'filename': 'test.fits', 'wcs': 1})

    def test_deleteredufromarchive_pymysql_fallback(self):
        """Lines 344-345: deleteredufromarchive uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import deleteredufromarchive
            with patch('lsc.mysqldef.dbConnect', return_value=mock_conn):
                deleteredufromarchive('test.fits')

    def test_getfromcoordinate_pymysql_fallback(self):
        """Lines 445-446: getfromcoordinate uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import getfromcoordinate
            result = getfromcoordinate(mock_conn, 'targets', 150.0, 2.0, 0.01)
            assert isinstance(result, tuple)

    def test_getlike_pymysql_fallback(self):
        """Lines 642-643: getlike uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import getlike
            result = getlike(mock_conn, 'photlco', 'filename', 'test')
            assert isinstance(result, tuple)

    def test_query_pymysql_fallback(self):
        """Lines 664-665: query uses pymysql when MySQLdb unavailable."""
        with _reload_mysqldef_without_MySQLdb() as pmysql:
            pmysql.cursors = MagicMock()
            pmysql.cursors.DictCursor = MagicMock()
            cursor = MagicMock()
            cursor.fetchall.return_value = ()
            cursor.rowcount = 0
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = cursor
            from lsc.mysqldef import query
            result = query(['SELECT 1'], mock_conn)
            assert result == () or result == ''


# ---------------------------------------------------------------------------
# Test: Line 246 - os.remove(.fz) when funpack fails to remove the file
# ---------------------------------------------------------------------------

class TestIngestreduFzRemoval:
    """
    Line 246: After funpack is called on a .fz file, if the .fz file still
    exists (os.path.isfile returns True), os.remove is called to clean it up.
    """

    def test_fz_file_removed_when_still_exists(self, monkeypatch):
        """Line 246: os.remove(fullpath + '.fz') called when file persists."""
        import lsc
        import lsc.mysqldef
        import lsc.util

        fake_conn = MagicMock()
        monkeypatch.setattr(lsc.mysqldef, 'getconnection',
                            lambda name: ('host', 'user', 'pass', 'db'))
        monkeypatch.setattr(lsc.mysqldef, 'dbConnect',
                            lambda *a, **k: fake_conn)

        fake_hdr = {
            'date-obs': '2024-01-01T00:00:00', 'DAY-OBS': '20240101',
            'exptime': 300.0, 'filter': 'rp', 'mjd': 60000.0,
            'TRACKNUM': '12345', 'TELESCOP': '1m0-01', 'instrume': 'fa03',
            'airmass': 1.2, 'object': 'SN2024abc', 'ut': '00:00:00',
            'wcserr': 0, 'RA': 150.0, 'DEC': 2.5, 'SITEID': 'lsc',
        }
        monkeypatch.setattr(lsc.util, 'readhdr', lambda img: fake_hdr)
        monkeypatch.setattr(lsc.util, 'readkey3', lambda hdr, key: fake_hdr.get(key, 'NOKEY'))
        monkeypatch.setattr(lsc.mysqldef, 'targimg', lambda *a, **k: 42)

        def mock_getfromdataraw(conn, table, col, val, column2='*'):
            if table in ('photlco', 'photlcoraw'):
                return []
            return [{'id': 1}]
        monkeypatch.setattr(lsc.mysqldef, 'getfromdataraw', mock_getfromdataraw)
        monkeypatch.setattr(lsc.mysqldef, 'insert_values', lambda *a, **k: None)
        monkeypatch.setattr(lsc.mysqldef, 'updatevalue', lambda *a, **k: None)

        # Mock os.system (for funpack call)
        monkeypatch.setattr('os.system', lambda cmd: 0)

        # Mock os.path.isfile to return True for the .fz file (line 245 condition)
        original_isfile = __import__('os').path.isfile
        def mock_isfile(path):
            if path.endswith('.fz'):
                return True  # .fz still exists after funpack
            return original_isfile(path)
        monkeypatch.setattr('os.path.isfile', mock_isfile)

        # Track os.remove calls
        remove_calls = []
        monkeypatch.setattr('os.remove', lambda path: remove_calls.append(path))

        from lsc.mysqldef import ingestredu
        ingestredu(['/data/test.fits.fz'])

        # Verify os.remove was called on the .fz file
        assert any('.fz' in call for call in remove_calls), \
            f"Expected os.remove to be called with .fz path, got: {remove_calls}"


# ---------------------------------------------------------------------------
# Test: Lines 365-435 - updateDatabase function
# ---------------------------------------------------------------------------

class TestUpdateDatabase:
    """
    The updateDatabase function:
    - Extracts a tarfile
    - Reads a pickle file containing command dictionaries
    - Dispatches to ingestredu, getvaluefromarchive, updatevalue
    - Copies fits files to destination directories
    """

    def _setup_updateDatabase_mocks(self, monkeypatch, tmp_path):
        """Set up all mocks needed by updateDatabase."""
        import pickle

        # Create a mock ntt module with ntt.util
        ntt_mock = types.ModuleType('ntt')
        ntt_util_mock = types.ModuleType('ntt.util')
        ntt_util_mock.readkey3 = MagicMock(return_value='ARCFILE_VALUE')
        ntt_util_mock.readhdr = MagicMock(return_value={})
        ntt_mock.util = ntt_util_mock
        sys.modules['ntt'] = ntt_mock
        sys.modules['ntt.util'] = ntt_util_mock

        # Mock the bare 'mysqldef' import used in updateDatabase
        # (from mysqldef import updatevalue, getvaluefromarchive, ...)
        mysqldef_bare = types.ModuleType('mysqldef')
        mysqldef_bare.updatevalue = MagicMock()
        mysqldef_bare.getvaluefromarchive = MagicMock(return_value={'filepath': '/data/raw/', 'col1': 'val1'})
        mysqldef_bare.dbConnect = MagicMock()
        mysqldef_bare.getfromdataraw = MagicMock(return_value=())
        mysqldef_bare.ingestredu = MagicMock()
        sys.modules['mysqldef'] = mysqldef_bare

        return mysqldef_bare

    def _cleanup_mocks(self):
        """Remove mock modules."""
        for mod in ('ntt', 'ntt.util', 'mysqldef'):
            sys.modules.pop(mod, None)

    def test_updateDatabase_ingestredu_command(self, monkeypatch, tmp_path):
        """Lines 381-386: updateDatabase dispatches 'ingestredu' command."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        # Create pickle data with ingestredu command
        mydict = {
            0: {
                'command': 'ingestredu',
                'lista': ['/data/img.fits'],
                'instument': 'fa03',
                'output': 'output.fits',
            }
        }

        # Write pickle file
        pkl_path = str(tmp_path / 'test_a_b_20240101_part1_part2.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(mydict, f)

        # The tarfile name determines directory structure
        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')

        # Mock os.system, os.path.isdir, os.mkdir, glob
        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        import glob as glob_mod
        monkeypatch.setattr(glob_mod, 'glob', lambda p: [pkl_path] if 'pkl' in p else [])

        # Mock ntt.util.readkey3 and readhdr for the fitsfile loop
        ntt_util = sys.modules['ntt.util']
        ntt_util.readkey3 = MagicMock(return_value='ARCFILE_VALUE')
        ntt_util.readhdr = MagicMock(return_value={})

        # Mock getvaluefromarchive to return empty (bbbb falsy) -> hits else branch
        mysqldef_bare.getvaluefromarchive = MagicMock(return_value='')

        try:
            from lsc.mysqldef import updateDatabase
            # We need to patch glob.glob and re.sub within the function scope
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            # ingestredu should have been called
            mysqldef_bare.ingestredu.assert_called_once_with(['/data/img.fits'])
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_getvaluefromarchive_command(self, monkeypatch, tmp_path):
        """Lines 387-392: updateDatabase dispatches 'getvaluefromarchive' command."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'getvaluefromarchive',
                'table': 'datarawNTT',
                'column': 'filename',
                'value': '*',
                'input': 'somefile.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        mysqldef_bare.getvaluefromarchive = MagicMock(return_value={'filepath': '/data/raw/', 'col1': 'val1'})

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            mysqldef_bare.getvaluefromarchive.assert_called_once_with(
                'datarawNTT', 'filename', 'somefile.fits', '*')
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_updatevalue_command_normal(self, monkeypatch, tmp_path):
        """Lines 393-399: updateDatabase dispatches 'updatevalue' with normal value."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'updatevalue',
                'table': 'redulogNTT',
                'column': 'wcs',
                'value': '1',
                'input': 'test.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            mysqldef_bare.updatevalue.assert_called_once_with(
                'redulogNTT', 'wcs', '1', 'test.fits')
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_updatevalue_voce_value_voce_column(self, monkeypatch, tmp_path):
        """Lines 400-404: value='voce' and column='voce' iterates over hh dict."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        # Need getvaluefromarchive first to set hh, then updatevalue with voce
        mydict = {
            0: {
                'command': 'getvaluefromarchive',
                'table': 'datarawNTT',
                'column': 'filename',
                'value': '*',
                'input': 'somefile.fits',
            },
            1: {
                'command': 'updatevalue',
                'table': 'redulogNTT',
                'column': 'voce',
                'value': 'voce',
                'input': 'test.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        # hh will be the return of getvaluefromarchive
        mysqldef_bare.getvaluefromarchive = MagicMock(
            return_value={'id': 1, 'filename': 'x.fits', 'wcs': 0, 'psf': 'done'})

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            # updatevalue should have been called for wcs and psf (not id, not filename)
            calls = mysqldef_bare.updatevalue.call_args_list
            called_columns = [c[0][1] for c in calls]
            assert 'wcs' in called_columns
            assert 'psf' in called_columns
            assert 'id' not in called_columns
            assert 'filename' not in called_columns
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_updatevalue_voce_value_other_column(self, monkeypatch, tmp_path):
        """Lines 405-406: value='voce' but column is NOT 'voce' -> uses hh[column]."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'getvaluefromarchive',
                'table': 'datarawNTT',
                'column': 'filename',
                'value': '*',
                'input': 'somefile.fits',
            },
            1: {
                'command': 'updatevalue',
                'table': 'redulogNTT',
                'column': 'wcs',
                'value': 'voce',
                'input': 'test.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        mysqldef_bare.getvaluefromarchive = MagicMock(
            return_value={'wcs': 99, 'psf': 'ok'})

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            # Should call updatevalue('redulogNTT', 'wcs', 99, 'test.fits')
            mysqldef_bare.updatevalue.assert_called_once_with(
                'redulogNTT', 'wcs', 99, 'test.fits')
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_unknown_command(self, monkeypatch, tmp_path, capsys):
        """Lines 407-408: unknown command prints warning."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'unknown_command_xyz',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            out = capsys.readouterr().out
            assert 'warning' in out.lower() or 'not recognise' in out
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_exception_handling(self, monkeypatch, tmp_path, capsys):
        """Lines 409-412: exception in command dispatch prints error info."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        # Command that will fail because it's missing required keys
        mydict = {
            0: {
                'command': 'ingestredu',
                # Missing 'lista', 'instument', 'output' keys
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)
        monkeypatch.setattr('os.path.isdir', lambda p: True)

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            out = capsys.readouterr().out
            assert 'problems' in out or '###' in out
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_no_pkl_file(self, monkeypatch, tmp_path):
        """Lines 373: when glob returns empty, no processing occurs."""
        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')

        monkeypatch.setattr('os.system', lambda cmd: 0)

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[]):
                # Should complete without error when no pkl file found
                updateDatabase(tarfile_name)
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_fitsfile_loop_with_bbbb(self, monkeypatch, tmp_path):
        """Lines 418-435: fitsfile loop when getvaluefromarchive returns data."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'ingestredu',
                'lista': ['/data/img.fits'],
                'instument': 'fa03',
                'output': 'output.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)

        dirs_created = []
        def mock_mkdir(path):
            dirs_created.append(path)
        monkeypatch.setattr('os.mkdir', mock_mkdir)
        monkeypatch.setattr('os.path.isdir', lambda p: False)

        # getvaluefromarchive returns truthy data (bbbb has 'filepath')
        mysqldef_bare.getvaluefromarchive = MagicMock(
            return_value={'filepath': '/data/raw/20240101/'})

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            # updatevalue should be called for filepath updates
            assert mysqldef_bare.updatevalue.call_count >= 2
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_fitsfile_loop_no_bbbb(self, monkeypatch, tmp_path):
        """Lines 427-430: fitsfile loop when getvaluefromarchive returns empty."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'ingestredu',
                'lista': ['/data/img.fits'],
                'instument': 'fa03',
                'output': 'output.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)

        dirs_created = []
        def mock_mkdir(path):
            dirs_created.append(path)
        monkeypatch.setattr('os.mkdir', mock_mkdir)
        monkeypatch.setattr('os.path.isdir', lambda p: False)

        # getvaluefromarchive returns falsy (empty string/list)
        mysqldef_bare.getvaluefromarchive = MagicMock(return_value='')

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            # dir3 should have been created (the else branch for no bbbb)
            assert len(dirs_created) > 0
        finally:
            self._cleanup_mocks()

    def test_updateDatabase_fitsfile_getvaluefromarchive_exception(self, monkeypatch, tmp_path):
        """Line 423: except branch when getvaluefromarchive raises in fitsfile loop."""
        import pickle

        mysqldef_bare = self._setup_updateDatabase_mocks(monkeypatch, tmp_path)

        mydict = {
            0: {
                'command': 'ingestredu',
                'lista': ['/data/img.fits'],
                'instument': 'fa03',
                'output': 'output.fits',
            }
        }

        tarfile_name = str(tmp_path / 'test_a_b_20240101_part1_part2.tar.gz')
        pkl_path = tarfile_name.replace('tar.gz', 'pkl')

        monkeypatch.setattr('os.system', lambda cmd: 0)

        dirs_created = []
        def mock_mkdir(path):
            dirs_created.append(path)
        monkeypatch.setattr('os.mkdir', mock_mkdir)
        monkeypatch.setattr('os.path.isdir', lambda p: False)

        # getvaluefromarchive RAISES an exception -> hits except: bbbb='' (line 423)
        mysqldef_bare.getvaluefromarchive = MagicMock(
            side_effect=Exception("DB connection error"))

        try:
            from lsc.mysqldef import updateDatabase
            with patch('glob.glob', return_value=[pkl_path]):
                with patch('builtins.open', mock_open(read_data=pickle.dumps(mydict))):
                    with patch('pickle.load', return_value=mydict):
                        updateDatabase(tarfile_name)

            # Since bbbb='' (falsy), the else branch is taken -> dir3 created
            assert len(dirs_created) > 0
        finally:
            self._cleanup_mocks()


# ---------------------------------------------------------------------------
# Test: Line 738 - gettargetid with multiple IDs but no 'hsine' key
# ---------------------------------------------------------------------------

class TestGettargetidNoHsine:
    """
    Line 735-736: When len(set(ll0['id'])) > 1 and 'hsine' is NOT in
    ll0.keys(), _targetid is set to ''.
    """

    def test_multiple_ids_no_hsine_returns_empty(self):
        """
        Line 736: multiple distinct ids without 'hsine' column -> returns ''.
        This is the branch: elif len(set(ll0['id']))>1 / else: _targetid=''
        """
        from unittest.mock import MagicMock, patch
        import lsc.mysqldef

        # Create mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Return rows with multiple distinct IDs but NO 'hsine' key
        mock_cursor.fetchall.return_value = (
            {'name': 'SN2024a', 'targetid': 1, 'id': 1, 'ra0': 150.0, 'dec0': 2.0},
            {'name': 'SN2024b', 'targetid': 2, 'id': 2, 'ra0': 150.1, 'dec0': 2.1},
        )
        mock_cursor.rowcount = 2
        mock_conn.cursor.return_value = mock_cursor

        # Patch the query function so we control what lista looks like
        with patch.object(lsc.mysqldef, 'query', return_value=[
            {'name': 'SN2024a', 'targetid': 1, 'id': 1, 'ra0': 150.0, 'dec0': 2.0},
            {'name': 'SN2024b', 'targetid': 2, 'id': 2, 'ra0': 150.1, 'dec0': 2.1},
        ]):
            result = lsc.mysqldef.gettargetid('SN2024abc', '', '', mock_conn, 0.01)

        # Should return '' since multiple IDs and no 'hsine'
        assert result == ''

    def test_multiple_ids_with_hsine_returns_closest(self):
        """
        Line 734: multiple distinct ids WITH 'hsine' -> returns id with min hsine.
        (Control test to verify the other branch.)
        """
        from unittest.mock import patch
        import lsc.mysqldef

        mock_conn = MagicMock()

        with patch.object(lsc.mysqldef, 'query', return_value=[
            {'name': 'SN2024a', 'targetid': 1, 'id': 1, 'ra0': 150.0, 'dec0': 2.0, 'hsine': 0.05},
            {'name': 'SN2024b', 'targetid': 2, 'id': 2, 'ra0': 150.1, 'dec0': 2.1, 'hsine': 0.001},
        ]):
            result = lsc.mysqldef.gettargetid('SN2024abc', '', '', mock_conn, 0.01)

        # Should return id=2 (minimum hsine)
        assert result == 2
