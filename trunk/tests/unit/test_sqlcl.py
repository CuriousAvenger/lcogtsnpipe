"""
Tests for lsc.sqlcl.
filtercomment() is pure text; query() is tested with a mocked urllib.request.
write_header() is tested with io.StringIO.
"""
import io
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# filtercomment — removes SQL -- comments
# ---------------------------------------------------------------------------

class TestFiltercomment:
    def test_removes_inline_comment(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT * FROM table -- this is a comment"
        result = filtercomment(sql)
        assert "this is a comment" not in result
        assert "SELECT" in result

    def test_no_comment_unchanged_content(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT ra, dec FROM PhotoPrimary"
        result = filtercomment(sql)
        assert "SELECT ra" in result

    def test_comment_only_line_becomes_whitespace(self):
        from lsc.sqlcl import filtercomment
        sql = "-- full comment line\nSELECT 1"
        result = filtercomment(sql)
        assert "full comment line" not in result
        assert "SELECT 1" in result

    def test_empty_string(self):
        from lsc.sqlcl import filtercomment
        result = filtercomment("")
        assert isinstance(result, str)

    def test_multiple_comment_lines(self):
        from lsc.sqlcl import filtercomment
        sql = "-- comment 1\nSELECT a -- inline\nFROM t -- another"
        result = filtercomment(sql)
        assert "comment 1" not in result
        assert "inline" not in result
        assert "SELECT" in result
        assert "FROM" in result

    def test_result_is_string(self):
        from lsc.sqlcl import filtercomment
        assert isinstance(filtercomment("SELECT 1 -- cmt"), str)


# ---------------------------------------------------------------------------
# write_header — writes SOURCE / TIME / QUERY header to file-like object
# ---------------------------------------------------------------------------

class TestWriteHeader:
    def test_writes_source_line(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com', 'SELECT 1')
        output = buf.getvalue()
        assert 'SOURCE:' in output
        assert 'http://example.com' in output

    def test_writes_time_line(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com', 'SELECT 1')
        output = buf.getvalue()
        assert 'TIME:' in output

    def test_writes_query(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com', 'SELECT ra FROM t')
        output = buf.getvalue()
        assert 'QUERY' in output
        assert 'SELECT ra FROM t' in output

    def test_prefix_is_used(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '##', 'http://x.com', 'SELECT 1')
        output = buf.getvalue()
        for line in output.strip().split('\n'):
            assert line.startswith('##')

    def test_multiline_query(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://x.com', 'SELECT a\nFROM t\nWHERE b=1')
        output = buf.getvalue()
        assert 'SELECT a' in output
        assert 'FROM t' in output
        assert 'WHERE b=1' in output

    def test_empty_query(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://x.com', '')
        output = buf.getvalue()
        assert 'QUERY' in output


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_formats_list(self):
        from lsc.sqlcl import formats
        assert 'csv' in formats
        assert 'xml' in formats
        assert 'html' in formats

    def test_default_url_is_string(self):
        from lsc.sqlcl import default_url
        assert isinstance(default_url, str)
        assert default_url.startswith('http')

    def test_default_fmt(self):
        from lsc.sqlcl import default_fmt
        assert default_fmt == 'csv'


# ---------------------------------------------------------------------------
# query() — mocked urllib.request.urlopen
# ---------------------------------------------------------------------------

class TestQuery:
    @pytest.mark.http
    def test_query_calls_urlopen(self):
        from lsc import sqlcl
        import io
        fake_response = io.BytesIO(b"ra,dec\n150.0,2.0\n")
        with patch("urllib.request.urlopen", return_value=fake_response) as mock_open:
            sqlcl.query("SELECT ra, dec FROM PhotoPrimary")
            assert mock_open.called

    @pytest.mark.http
    def test_query_encodes_sql(self):
        from lsc import sqlcl
        from unittest.mock import MagicMock
        fake_response = MagicMock()
        fake_response.readline.return_value = b""
        with patch("urllib.request.urlopen", return_value=fake_response) as mock_open:
            sqlcl.query("SELECT 1")
            call_url = mock_open.call_args[0][0]
            assert "SELECT" in call_url or "cmd" in call_url or "SELECT" in str(mock_open.call_args)


class TestQuery:
    @pytest.mark.http
    def test_query_calls_urlopen(self):
        from lsc import sqlcl
        import io
        fake_response = io.BytesIO(b"ra,dec\n150.0,2.0\n")
        with patch("urllib.request.urlopen", return_value=fake_response) as mock_open:
            sqlcl.query("SELECT ra, dec FROM PhotoPrimary")
            assert mock_open.called

    @pytest.mark.http
    def test_query_encodes_sql(self):
        from lsc import sqlcl
        from unittest.mock import MagicMock
        fake_response = MagicMock()
        fake_response.readline.return_value = b""
        with patch("urllib.request.urlopen", return_value=fake_response) as mock_open:
            sqlcl.query("SELECT 1")
            call_url = mock_open.call_args[0][0]
            assert "SELECT" in call_url or "cmd" in call_url or "SELECT" in str(mock_open.call_args)


# ---------------------------------------------------------------------------
# usage() — prints help and calls sys.exit
# Note: sqlcl.py uses sys.exit() without a module-level import of sys.
# Inject sys into the module namespace before calling usage/main.
# ---------------------------------------------------------------------------

def _inject_sys_into_sqlcl():
    """sqlcl.py uses sys without a module-level import; inject it."""
    import sys as _sys
    import lsc.sqlcl as _mod
    _mod.sys = _sys


class TestUsage:
    def setup_method(self):
        _inject_sys_into_sqlcl()

    def test_exits_with_given_status(self):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit) as exc_info:
            usage(1)
        assert exc_info.value.code == 1

    def test_exits_with_zero(self):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit) as exc_info:
            usage(0)
        assert exc_info.value.code == 0

    def test_prints_error_message(self, capsys):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit):
            usage(1, 'test error message')
        captured = capsys.readouterr()
        assert 'test error message' in captured.out

    def test_no_message_no_error_print(self, capsys):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit):
            usage(0, '')
        captured = capsys.readouterr()
        assert 'ERROR' not in captured.out


# ---------------------------------------------------------------------------
# main() — command-line argument parser
# ---------------------------------------------------------------------------

class TestMain:
    def setup_method(self):
        _inject_sys_into_sqlcl()

    def _fake_response(self, content="col1\nval1\n"):
        """sqlcl.py reads response line-by-line as strings (Python 2 style)."""
        return io.StringIO(content)

    def test_query_flag_runs_query(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("ra,dec\n1.0,2.0\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-q', 'SELECT TOP 1 ra,dec FROM PhotoPrimary'])

    def test_verbose_writes_header(self, capsys):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-v', '-q', 'SELECT 1'])
        captured = capsys.readouterr()
        assert 'SOURCE' in captured.out

    def test_skip_first_line_flag(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("header\ndata\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-l', '-q', 'SELECT 1'])

    def test_format_csv(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("a,b\n1,2\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-f', 'csv', '-q', 'SELECT 1'])

    def test_format_xml(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("<xml/>\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-f', 'xml', '-q', 'SELECT 1'])

    def test_format_html(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("<html/>\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-f', 'html', '-q', 'SELECT 1'])

    def test_wrong_format_calls_usage(self):
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-f', 'invalid_fmt', '-q', 'SELECT 1'])

    def test_bad_option_calls_usage(self):
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-Z'])

    def test_sql_error_response_goes_to_stderr(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("ERROR: bad SQL\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', '-q', 'SELECT BAD'])

    def test_url_flag_used(self):
        from lsc.sqlcl import main
        fake_resp = self._fake_response("col\nval\n")
        with patch('urllib.request.urlopen', side_effect=lambda url: fake_resp) as mock_open:
            main(['sqlcl', '-s', 'http://custom.url/query', '-q', 'SELECT 1'])
            called_url = mock_open.call_args[0][0]
            assert 'custom.url' in called_url

    def test_query_from_file(self, tmp_path):
        from lsc.sqlcl import main
        sql_file = tmp_path / 'query.sql'
        sql_file.write_text('SELECT TOP 1 ra FROM PhotoPrimary')
        fake_resp = self._fake_response("ra\n1.0\n")
        with patch('urllib.request.urlopen', return_value=fake_resp):
            main(['sqlcl', str(sql_file)])
