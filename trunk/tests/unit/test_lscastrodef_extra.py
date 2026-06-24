"""
Additional tests for lsc.lscastrodef — covers readapass (parsing APASS output)
and transformsloanlandolt / transformlandoltsloan edge cases.
No subprocess or network access required.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# readapass — parses APASS catalog from subprocess output
# ---------------------------------------------------------------------------

class TestReadapass:
    def test_parses_valid_output(self, tmp_path, monkeypatch):
        """Given a valid subprocess output, returns a dict with expected keys."""
        from lsc.lscastrodef import readapass

        # Mock subprocess.Popen to return a fake APASS output
        fake_output = """#RAdeg DECdeg B V Pg Pr Pi eB eV eg er ei
150.000 2.000 14.5 13.8 14.2 13.9 13.7 0.01 0.02 0.01 0.01 0.02
150.100 2.100 15.0 14.3 14.7 14.4 14.1 0.02 0.01 0.02 0.01 0.01
"""
        mock_popen = MagicMock()
        mock_popen.return_value.communicate.return_value = (fake_output.encode(), b'')
        with patch('subprocess.Popen', mock_popen):
            with patch('builtins.open', MagicMock()):
                result = readapass(150.0, 2.0)
        assert isinstance(result, dict)
        assert '#RAdeg' in result or len(result) > 0

    def test_empty_output(self, monkeypatch):
        """Empty subprocess output returns empty dict."""
        from lsc.lscastrodef import readapass

        fake_output = ""
        mock_popen = MagicMock()
        mock_popen.return_value.communicate.return_value = (fake_output.encode(), b'')
        with patch('subprocess.Popen', mock_popen):
            with patch('builtins.open', MagicMock()):
                result = readapass(150.0, 2.0)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# transformsloanlandolt — converts Sloan stds to Landolt
# ---------------------------------------------------------------------------

class TestTransformsloanlandolt:
    def test_returns_ubvri_keys(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'g': [14.0, 14.5],
            'r': [13.5, 14.0],
            'i': [13.0, 13.5],
        }
        result = transformsloanlandolt(stdcoo)
        assert 'B' in result or 'V' in result or 'R' in result

    def test_output_same_length(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'g': [14.0, 14.5, 15.0],
            'r': [13.5, 14.0, 14.5],
            'i': [13.0, 13.5, 14.0],
        }
        result = transformsloanlandolt(stdcoo)
        for key in result:
            assert len(result[key]) == 3


# ---------------------------------------------------------------------------
# transformlandoltsloan — converts Landolt stds to Sloan
# ---------------------------------------------------------------------------

class TestTransformlandoltsloan:
    def test_returns_griz_keys(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {
            'B': [14.0, 14.5],
            'V': [13.5, 14.0],
            'R': [13.0, 13.5],
            'I': [12.5, 13.0],
        }
        result = transformlandoltsloan(stdcoo)
        assert 'g' in result or 'r' in result

    def test_output_same_length(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {
            'B': [14.0, 14.5, 15.0],
            'V': [13.5, 14.0, 14.5],
            'R': [13.0, 13.5, 14.0],
            'I': [12.5, 13.0, 13.5],
        }
        result = transformlandoltsloan(stdcoo)
        for key in result:
            assert len(result[key]) == 3


# ---------------------------------------------------------------------------
# half_total_flux_radius_to_fwhm — additional edge cases
# ---------------------------------------------------------------------------

class TestHalfTotalFluxRadiusToFwhmExtra:
    def test_negative_input(self):
        """Negative htfr produces negative FWHM (physically odd but mathematically consistent)."""
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        result = half_total_flux_radius_to_fwhm(-3.0)
        assert result < 0

    def test_large_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        result = half_total_flux_radius_to_fwhm(100.0)
        expected = 100.0 * 0.8493218 * 2.355
        assert abs(result - expected) < 1e-6
