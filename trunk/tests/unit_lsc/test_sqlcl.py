"""
Tests for lsc.sqlcl — SQL client utilities for SDSS queries.
Covers filtercomment, query, write_header, usage, main, and edge cases.
"""
import importlib.util
import io
import os
import sys
import runpy
import pytest
from unittest.mock import patch, MagicMock, mock_open

pytestmark = pytest.mark.unit

# Load lsc.sqlcl directly from its file path to avoid triggering lsc/__init__.py
# which imports modules requiring unavailable dependencies (astroquery).
_SQLCL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'src', 'lsc', 'sqlcl.py'
)
_spec = importlib.util.spec_from_file_location("lsc.sqlcl", os.path.abspath(_SQLCL_PATH))
_sqlcl_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sqlcl_mod)
# Inject sys into the module (sqlcl.py uses sys.exit in usage() and sys.stdout in main()
# but only imports sys at module level when run as __main__)
_sqlcl_mod.sys = sys
sys.modules.setdefault("lsc.sqlcl", _sqlcl_mod)


def _inject_sys():
    """sqlcl.py uses sys without a module-level import in some functions.
    Now handled at module load time above, but kept for compatibility."""
    import sys as _sys
    import lsc.sqlcl as _mod
    _mod.sys = _sys


# ---------------------------------------------------------------------------
# filtercomment — removes SQL comments starting with --
# ---------------------------------------------------------------------------

class TestFiltercommentBasic:
    """Basic functionality of filtercomment."""

    def test_removes_inline_comment(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT * FROM table -- this is a comment"
        result = filtercomment(sql)
        assert "SELECT * FROM table" in result
        assert "this is a comment" not in result

    def test_removes_full_line_comment(self):
        from lsc.sqlcl import filtercomment
        sql = "-- full line comment\nSELECT 1"
        result = filtercomment(sql)
        assert "full line comment" not in result
        assert "SELECT 1" in result

    def test_preserves_sql_without_comments(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT ra, dec FROM PhotoPrimary WHERE ra > 150"
        result = filtercomment(sql)
        assert "SELECT ra, dec FROM PhotoPrimary WHERE ra > 150" in result

    def test_handles_multiple_comment_lines(self):
        from lsc.sqlcl import filtercomment
        sql = "-- comment 1\n-- comment 2\nSELECT 1"
        result = filtercomment(sql)
        assert "comment 1" not in result
        assert "comment 2" not in result
        assert "SELECT 1" in result

    def test_handles_inline_and_full_comments(self):
        from lsc.sqlcl import filtercomment
        sql = "-- header\nSELECT a -- inline\nFROM t -- table comment"
        result = filtercomment(sql)
        assert "header" not in result
        assert "inline" not in result
        assert "table comment" not in result
        assert "SELECT a" in result
        assert "FROM t" in result

    def test_result_is_string(self):
        from lsc.sqlcl import filtercomment
        assert isinstance(filtercomment("SELECT 1 -- cmt"), str)


class TestFiltercommentEdgeCases:
    """Edge cases for filtercomment."""

    def test_empty_string(self):
        from lsc.sqlcl import filtercomment
        result = filtercomment("")
        assert isinstance(result, str)

    def test_only_comment(self):
        from lsc.sqlcl import filtercomment
        result = filtercomment("-- only a comment")
        assert "--" not in result
        assert result.strip() == ""

    def test_double_dash_in_string_literal_still_removed(self):
        """filtercomment is naive — it removes after -- regardless of context."""
        from lsc.sqlcl import filtercomment
        sql = "SELECT '2020--01' FROM t"
        result = filtercomment(sql)
        assert "01' FROM t" not in result

    def test_preserves_newline_structure(self):
        """Each input line produces output with os.linesep."""
        from lsc.sqlcl import filtercomment
        sql = "SELECT a\nFROM b\nWHERE c=1"
        result = filtercomment(sql)
        assert os.linesep in result

    def test_single_line_no_newline(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT 1"
        result = filtercomment(sql)
        assert "SELECT 1" in result

    def test_whitespace_only(self):
        from lsc.sqlcl import filtercomment
        result = filtercomment("   \n   \n   ")
        assert isinstance(result, str)

    def test_many_dashes_not_double(self):
        """Single dash should not be treated as comment."""
        from lsc.sqlcl import filtercomment
        sql = "SELECT a - b FROM t"
        result = filtercomment(sql)
        assert "SELECT a - b FROM t" in result

    def test_triple_dash(self):
        """Triple dash still triggers comment removal (starts with --)."""
        from lsc.sqlcl import filtercomment
        sql = "SELECT 1 --- triple dash"
        result = filtercomment(sql)
        assert "triple dash" not in result
        assert "SELECT 1" in result

    def test_comment_at_start_of_line(self):
        from lsc.sqlcl import filtercomment
        sql = "--comment\nSELECT 1"
        result = filtercomment(sql)
        assert "comment" not in result

    def test_very_long_query(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT " + ", ".join([f"col{i}" for i in range(1000)]) + " FROM table"
        result = filtercomment(sql)
        assert "col999" in result

    def test_special_characters_in_sql(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT * FROM t WHERE name = 'O''Brien' -- comment"
        result = filtercomment(sql)
        assert "O''Brien" in result
        assert "comment" not in result

    def test_unicode_in_query(self):
        from lsc.sqlcl import filtercomment
        sql = "SELECT * FROM t WHERE name = 'alpha' -- unicode test"
        result = filtercomment(sql)
        assert "alpha" in result
        assert "unicode test" not in result

    def test_returns_string_type(self):
        from lsc.sqlcl import filtercomment
        assert isinstance(filtercomment("SELECT 1"), str)
        assert isinstance(filtercomment(""), str)
        assert isinstance(filtercomment("-- comment only"), str)


# ---------------------------------------------------------------------------
# write_header — writes formatted header to a file-like object
# ---------------------------------------------------------------------------

class TestWriteHeaderBasic:
    """Test write_header output format."""

    def test_writes_source(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com/sql', 'SELECT 1')
        output = buf.getvalue()
        assert 'SOURCE:' in output
        assert 'http://example.com/sql' in output

    def test_writes_time(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com', 'SELECT 1')
        output = buf.getvalue()
        assert 'TIME:' in output

    def test_writes_query_label(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com', 'SELECT ra FROM stars')
        output = buf.getvalue()
        assert 'QUERY' in output

    def test_writes_query_content(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com', 'SELECT ra, dec FROM PhotoPrimary')
        output = buf.getvalue()
        assert 'SELECT ra, dec FROM PhotoPrimary' in output

    def test_prefix_applied_to_all_lines(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '##', 'http://x.com', 'SELECT 1')
        output = buf.getvalue()
        for line in output.strip().split('\n'):
            assert line.startswith('##')


class TestWriteHeaderEdgeCases:
    """Edge cases for write_header."""

    def test_empty_prefix(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '', 'http://x.com', 'SELECT 1')
        output = buf.getvalue()
        assert 'SOURCE:' in output

    def test_multiline_query(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        query = "SELECT a\nFROM b\nWHERE c > 1\nORDER BY a"
        write_header(buf, '#', 'http://x.com', query)
        output = buf.getvalue()
        assert 'SELECT a' in output
        assert 'FROM b' in output
        assert 'WHERE c > 1' in output
        assert 'ORDER BY a' in output

    def test_empty_query(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://x.com', '')
        output = buf.getvalue()
        assert 'QUERY' in output

    def test_empty_url(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', '', 'SELECT 1')
        output = buf.getvalue()
        assert 'SOURCE:' in output

    def test_long_prefix(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '####', 'http://x.com', 'SELECT 1')
        output = buf.getvalue()
        for line in output.strip().split('\n'):
            assert line.startswith('####')

    def test_special_chars_in_url(self):
        from lsc.sqlcl import write_header
        buf = io.StringIO()
        write_header(buf, '#', 'http://example.com/q?a=1&b=2', 'SELECT 1')
        output = buf.getvalue()
        assert 'http://example.com/q?a=1&b=2' in output


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Test module-level attributes."""

    def test_formats_contains_csv_xml_html(self):
        from lsc.sqlcl import formats
        assert formats == ['csv', 'xml', 'html']

    def test_formats_is_list(self):
        from lsc.sqlcl import formats
        assert isinstance(formats, list)

    def test_default_url_is_public(self):
        from lsc.sqlcl import default_url, public_url
        assert default_url == public_url

    def test_default_url_contains_sdss(self):
        from lsc.sqlcl import default_url
        assert 'sdss' in default_url.lower() or 'skyserver' in default_url.lower()

    def test_default_url_starts_with_http(self):
        from lsc.sqlcl import default_url
        assert default_url.startswith('http')

    def test_default_fmt_is_csv(self):
        from lsc.sqlcl import default_fmt
        assert default_fmt == 'csv'

    def test_astro_url_is_string(self):
        from lsc.sqlcl import astro_url
        assert isinstance(astro_url, str)
        assert astro_url.startswith('http')

    def test_public_url_is_string(self):
        from lsc.sqlcl import public_url
        assert isinstance(public_url, str)
        assert public_url.startswith('http')


# ---------------------------------------------------------------------------
# query() — mock urllib to test query construction
# ---------------------------------------------------------------------------

class TestQueryFunction:
    """Test the query() function with mocked network access."""

    @pytest.mark.http
    def test_calls_urlopen(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"ra,dec\n1.0,2.0\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT ra, dec FROM stars")
            assert mock_open.called

    @pytest.mark.http
    def test_default_url_used(self):
        from lsc.sqlcl import query, default_url
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT 1")
            call_url = mock_open.call_args[0][0]
            assert call_url.startswith(default_url)

    @pytest.mark.http
    def test_custom_url_used(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT 1", url="http://custom.server/sql")
            call_url = mock_open.call_args[0][0]
            assert 'custom.server' in call_url

    @pytest.mark.http
    def test_format_parameter_encoded(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"<xml/>")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT 1", fmt='xml')
            call_url = mock_open.call_args[0][0]
            assert 'xml' in call_url

    @pytest.mark.http
    def test_sql_is_url_encoded(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT * FROM t WHERE a > 5")
            call_url = mock_open.call_args[0][0]
            assert ' ' not in call_url.split('?')[1] or '%20' in call_url or '+' in call_url

    @pytest.mark.http
    def test_comments_stripped_before_query(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT 1 -- comment")
            call_url = mock_open.call_args[0][0]
            assert 'comment' not in call_url

    @pytest.mark.http
    def test_returns_file_object(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response):
            result = query("SELECT 1")
            assert result is fake_response

    @pytest.mark.http
    def test_empty_query(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("")
            assert mock_open.called

    @pytest.mark.http
    def test_very_long_query(self):
        from lsc.sqlcl import query
        long_sql = "SELECT " + ", ".join([f"col{i}" for i in range(500)]) + " FROM t"
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query(long_sql)
            assert mock_open.called

    @pytest.mark.http
    def test_special_characters_in_query(self):
        from lsc.sqlcl import query
        fake_response = io.BytesIO(b"col\nval\n")
        with patch('urllib.request.urlopen', return_value=fake_response) as mock_open:
            query("SELECT * FROM t WHERE name = 'a&b'")
            assert mock_open.called


# ---------------------------------------------------------------------------
# usage() — help/error function
# ---------------------------------------------------------------------------

class TestUsage:
    """Test the usage() function."""

    def setup_method(self):
        _inject_sys()

    def test_exits_with_status_0(self):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit) as exc_info:
            usage(0)
        assert exc_info.value.code == 0

    def test_exits_with_status_1(self):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit) as exc_info:
            usage(1)
        assert exc_info.value.code == 1

    def test_exits_with_arbitrary_status(self):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit) as exc_info:
            usage(42)
        assert exc_info.value.code == 42

    def test_prints_error_message(self, capsys):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit):
            usage(1, 'Something went wrong')
        out = capsys.readouterr().out
        assert 'Something went wrong' in out

    def test_no_message_when_empty(self, capsys):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit):
            usage(0, '')
        out = capsys.readouterr().out
        assert 'ERROR' not in out

    def test_prints_docstring(self, capsys):
        from lsc.sqlcl import usage
        with pytest.raises(SystemExit):
            usage(0)
        out = capsys.readouterr().out
        assert 'sqlcl' in out or 'Usage' in out or 'Options' in out


# ---------------------------------------------------------------------------
# main() — command-line interface
# ---------------------------------------------------------------------------

class TestMainBasic:
    """Test main() with various argument combinations."""

    def setup_method(self):
        _inject_sys()

    def _fake_response(self, content="col1\nval1\n"):
        return io.StringIO(content)

    def test_query_flag_executes_query(self):
        from lsc.sqlcl import main
        resp = self._fake_response("ra,dec\n1.0,2.0\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-q', 'SELECT TOP 1 ra, dec FROM PhotoPrimary'])

    def test_multiple_queries(self):
        """Multiple -q flags should run multiple queries."""
        from lsc.sqlcl import main
        call_count = [0]

        def mock_urlopen(url):
            call_count[0] += 1
            return io.StringIO("col\nval\n")

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            main(['sqlcl', '-q', 'SELECT 1', '-q', 'SELECT 2'])
        assert call_count[0] == 2

    def test_verbose_writes_header(self, capsys):
        from lsc.sqlcl import main
        resp = self._fake_response("col\nval\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-v', '-q', 'SELECT 1'])
        out = capsys.readouterr().out
        assert 'SOURCE' in out
        assert 'TIME' in out

    def test_skip_first_line(self, capsys):
        from lsc.sqlcl import main
        resp = self._fake_response("header_line\ndata_line\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-l', '-q', 'SELECT 1'])
        out = capsys.readouterr().out
        assert 'header_line' not in out
        assert 'data_line' in out


class TestMainFormats:
    """Test main() with different output format flags."""

    def setup_method(self):
        _inject_sys()

    def _fake_response(self, content="col\nval\n"):
        return io.StringIO(content)

    def test_csv_format(self):
        from lsc.sqlcl import main
        resp = self._fake_response("a,b\n1,2\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-f', 'csv', '-q', 'SELECT 1'])

    def test_xml_format(self):
        from lsc.sqlcl import main
        resp = self._fake_response("<xml/>\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-f', 'xml', '-q', 'SELECT 1'])

    def test_html_format(self):
        from lsc.sqlcl import main
        resp = self._fake_response("<html/>\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-f', 'html', '-q', 'SELECT 1'])

    def test_invalid_format_exits(self):
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-f', 'json', '-q', 'SELECT 1'])

    def test_invalid_format_uppercase(self):
        """Format check is case-sensitive — CSV (uppercase) is invalid."""
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-f', 'CSV', '-q', 'SELECT 1'])


class TestMainURLFlag:
    """Test main() with custom URL."""

    def setup_method(self):
        _inject_sys()

    def test_custom_url(self):
        from lsc.sqlcl import main
        resp = io.StringIO("col\nval\n")
        with patch('urllib.request.urlopen', return_value=resp) as mock_open:
            main(['sqlcl', '-s', 'http://my.server/sql', '-q', 'SELECT 1'])
            called_url = mock_open.call_args[0][0]
            assert 'my.server' in called_url

    def test_env_variable_url(self):
        """SQLCLURL environment variable should be used as default."""
        from lsc.sqlcl import main
        resp = io.StringIO("col\nval\n")
        with patch.dict(os.environ, {'SQLCLURL': 'http://env.server/sql'}):
            with patch('urllib.request.urlopen', return_value=resp) as mock_open:
                main(['sqlcl', '-q', 'SELECT 1'])
                called_url = mock_open.call_args[0][0]
                assert 'env.server' in called_url


class TestMainErrorHandling:
    """Test main() error handling."""

    def setup_method(self):
        _inject_sys()

    def test_bad_option_exits(self):
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-Z'])

    def test_help_flag_exits(self):
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-h'])

    def test_question_mark_flag_exits(self):
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '-?'])

    def test_sql_error_response_to_stderr(self, capsys):
        """When SDSS returns ERROR, output goes to stderr."""
        from lsc.sqlcl import main
        resp = io.StringIO("ERROR: syntax error near 'SLECT'\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-q', 'SLECT 1'])
        captured = capsys.readouterr()
        assert 'ERROR' in captured.err

    def test_nonexistent_file_exits(self):
        """Passing a nonexistent file should trigger usage/exit."""
        from lsc.sqlcl import main
        with pytest.raises(SystemExit):
            main(['sqlcl', '/nonexistent/path/to/query.sql'])


class TestMainFileInput:
    """Test main() reading queries from files."""

    def setup_method(self):
        _inject_sys()

    def test_query_from_file(self, tmp_path):
        from lsc.sqlcl import main
        sql_file = tmp_path / 'query.sql'
        sql_file.write_text('SELECT TOP 1 ra FROM PhotoPrimary')
        resp = io.StringIO("ra\n1.0\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', str(sql_file)])

    def test_multiple_files(self, tmp_path):
        from lsc.sqlcl import main
        f1 = tmp_path / 'q1.sql'
        f2 = tmp_path / 'q2.sql'
        f1.write_text('SELECT 1')
        f2.write_text('SELECT 2')
        call_count = [0]

        def mock_urlopen(url):
            call_count[0] += 1
            return io.StringIO("col\nval\n")

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            main(['sqlcl', str(f1), str(f2)])
        assert call_count[0] == 2

    def test_file_and_query_flag_together(self, tmp_path):
        """Both -q and file arguments should all be executed."""
        from lsc.sqlcl import main
        f1 = tmp_path / 'q.sql'
        f1.write_text('SELECT 1')
        call_count = [0]

        def mock_urlopen(url):
            call_count[0] += 1
            return io.StringIO("col\nval\n")

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            main(['sqlcl', '-q', 'SELECT 2', str(f1)])
        assert call_count[0] == 2


class TestMainOutput:
    """Test main() output behavior."""

    def setup_method(self):
        _inject_sys()

    def test_normal_output_goes_to_stdout(self, capsys):
        from lsc.sqlcl import main
        resp = io.StringIO("col1,col2\nval1,val2\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-q', 'SELECT 1'])
        out = capsys.readouterr().out
        assert 'col1' in out
        assert 'val1' in out

    def test_no_query_no_output(self, capsys):
        """With no queries and no files, nothing should be written."""
        from lsc.sqlcl import main
        main(['sqlcl'])
        out = capsys.readouterr().out
        assert 'SOURCE' not in out

    def test_multiline_response(self, capsys):
        from lsc.sqlcl import main
        resp = io.StringIO("col\nrow1\nrow2\nrow3\n")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-q', 'SELECT 1'])
        out = capsys.readouterr().out
        assert 'row1' in out
        assert 'row2' in out
        assert 'row3' in out

    def test_empty_response(self, capsys):
        """Empty response should not crash."""
        from lsc.sqlcl import main
        resp = io.StringIO("")
        with patch('urllib.request.urlopen', return_value=resp):
            main(['sqlcl', '-q', 'SELECT 1'])


# ---------------------------------------------------------------------------
# __main__ guard coverage (from test_small_cov100.py)
# ---------------------------------------------------------------------------

class TestSqlclMainGuard:
    def test_lines_109_110_main_guard(self):
        """Run sqlcl.py as __main__ via runpy to cover the if __name__ == '__main__' guard."""
        with patch.object(sys, 'argv', ['sqlcl.py', '-h']):
            try:
                runpy.run_module('lsc.sqlcl', run_name='__main__', alter_sys=True)
            except SystemExit:
                pass
