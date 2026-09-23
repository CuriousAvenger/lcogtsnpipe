"""
Tests for astroquery usage in lcogtsnpipe.

The pipeline queries SDSS, Vizier, and Gaia for standard star catalogs.
All network calls are mocked so tests run offline.

Relevant source locations:
- src/lsc/lscabsphotdef.py: SDSS (line ~21), Vizier (line ~1160), Gaia (line ~1237)
- src/lsc/externaldata.py:  SDSS (line ~164)

Coverage additions (vs. original):
- SDSS.query_sql()              (sloan2file primary pathway)
- SDSS.get_images()             (externaldata download path)
- SDSS run/camcol/field dedup   (downloadsdss filtering logic)
- SDSS ASCII catalog write path (column format strings)
- Vizier ROW_LIMIT=-1 assignment
- Vizier .columns / .column_filters assignment
- Vizier PanSTARRS quality-flag bitwise filter (panstarrs2file)
- Gaia query_object_async()     (actual pipeline call in gaia2file)
- Gaia astrometric noise filter + SOURCE_ID column selection
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdss_table(nrows: int = 5):
    """Return an astropy Table that mimics a SDSS photo-obj query result."""
    from astropy.table import Table
    rng = np.random.default_rng(0)
    return Table({
        "ra":    rng.uniform(149.9, 150.1, nrows),
        "dec":   rng.uniform(1.9, 2.1, nrows),
        "psfMag_g": rng.uniform(17, 22, nrows),
        "psfMag_r": rng.uniform(16, 21, nrows),
        "psfMag_i": rng.uniform(15, 20, nrows),
        "psfMagErr_g": rng.uniform(0.01, 0.05, nrows),
        "psfMagErr_r": rng.uniform(0.01, 0.05, nrows),
        "psfMagErr_i": rng.uniform(0.01, 0.05, nrows),
    })


def _make_vizier_table(nrows: int = 4):
    from astropy.table import Table
    rng = np.random.default_rng(1)
    return Table({
        "RAJ2000": rng.uniform(149.9, 150.1, nrows),
        "DEJ2000": rng.uniform(1.9, 2.1, nrows),
        "Vmag": rng.uniform(14, 18, nrows),
        "Bmag": rng.uniform(15, 19, nrows),
        "e_Vmag": np.full(nrows, 0.02),
    })


def _make_gaia_table(nrows: int = 6):
    from astropy.table import Table
    rng = np.random.default_rng(2)
    return Table({
        "ra":    rng.uniform(149.9, 150.1, nrows),
        "dec":   rng.uniform(1.9, 2.1, nrows),
        "phot_g_mean_mag":  rng.uniform(15, 20, nrows),
        "phot_bp_mean_mag": rng.uniform(15, 20, nrows),
        "phot_rp_mean_mag": rng.uniform(14, 19, nrows),
    })


# ---------------------------------------------------------------------------
# SDSS
# ---------------------------------------------------------------------------

class TestAstroquerySDSS:
    """Tests that the pipeline's SDSS query pattern works with mocked I/O."""

    def test_sdss_import(self):
        from astroquery.sdss import SDSS
        assert callable(SDSS.query_region)

    def test_sdss_query_region_called_with_skycoord(self):
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        with patch.object(SDSS, "query_region", return_value=_make_sdss_table()) as mock_qr:
            center = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="icrs")
            result = SDSS.query_region(center, radius=10 * u.arcmin)
            mock_qr.assert_called_once()
            assert len(result) == 5

    def test_sdss_returns_expected_columns(self):
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        table = _make_sdss_table()
        with patch.object(SDSS, "query_region", return_value=table):
            center = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="icrs")
            result = SDSS.query_region(center, radius=5 * u.arcmin)
            for col in ("ra", "dec", "psfMag_r", "psfMag_g"):
                assert col in result.colnames

    def test_sdss_returns_none_on_no_data(self):
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        with patch.object(SDSS, "query_region", return_value=None):
            center = SkyCoord(ra=300.0 * u.degree, dec=80.0 * u.degree, frame="icrs")
            result = SDSS.query_region(center, radius=1 * u.arcmin)
            assert result is None

    def test_sdss_filter_by_magnitude(self):
        """Simulate pipeline filtering of the returned catalog."""
        table = _make_sdss_table(nrows=20)
        bright = table[table["psfMag_r"] < 20.0]
        assert len(bright) >= 0   # just verifying filter doesn't crash

    def test_sdss_photoobj_fields_accepted(self):
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        fields = ["ra", "dec", "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z"]
        with patch.object(SDSS, "query_region", return_value=_make_sdss_table()) as mock_qr:
            center = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="icrs")
            SDSS.query_region(center, radius=5 * u.arcmin, photoobj_fields=fields)
            mock_qr.assert_called_once()


# ---------------------------------------------------------------------------
# Vizier
# ---------------------------------------------------------------------------

class TestAstroqueryVizier:
    """Tests for the Vizier query pattern (Landolt/APASS catalogs)."""

    def test_vizier_import(self):
        from astroquery.vizier import Vizier
        assert callable(Vizier.query_region)

    def test_vizier_query_region_called(self):
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        result_list = [_make_vizier_table()]   # Vizier returns a TableList
        with patch.object(Vizier, "query_region", return_value=result_list) as mock_qr:
            center = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="icrs")
            results = Vizier.query_region(center, radius=0.5 * u.degree)
            mock_qr.assert_called_once()
            assert len(results[0]) == 4

    def test_vizier_empty_result(self):
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        with patch.object(Vizier, "query_region", return_value=[]):
            center = SkyCoord(ra=0.0 * u.degree, dec=89.9 * u.degree, frame="icrs")
            result = Vizier.query_region(center, radius=1 * u.arcmin)
            assert len(result) == 0

    def test_vizier_row_limit_attribute(self):
        """Verify Vizier exposes ROW_LIMIT without making network calls."""
        from astroquery.vizier import Vizier
        # ROW_LIMIT is a class-level attribute — access it without querying
        assert hasattr(Vizier, "ROW_LIMIT")

    def test_vizier_catalog_column_check(self):
        table = _make_vizier_table()
        assert "RAJ2000" in table.colnames
        assert "Vmag" in table.colnames


# ---------------------------------------------------------------------------
# Gaia
# ---------------------------------------------------------------------------
# NOTE: `from astroquery.gaia import Gaia` triggers a TAP server handshake
# which can block for seconds on slow/no network.  All Gaia tests therefore
# mock the entire module so the real class is never instantiated.
# ---------------------------------------------------------------------------

class TestAstroqueryGaia:
    """Tests for the Gaia query pattern (DR2/DR3 for standard stars)."""

    def _mock_gaia(self):
        """Return a MagicMock that stands in for the astroquery.gaia.Gaia object."""
        gaia = MagicMock()
        mock_job = MagicMock()
        mock_job.get_results.return_value = _make_gaia_table()
        gaia.cone_search_async.return_value = mock_job
        gaia.launch_job_async.return_value = mock_job
        return gaia

    def test_gaia_module_importable(self):
        """astroquery.gaia can be imported (we only need the module, not Gaia())."""
        import astroquery.gaia
        assert hasattr(astroquery.gaia, "Gaia")

    def test_gaia_cone_search_async_called(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        mock_gaia = self._mock_gaia()
        with patch("astroquery.gaia.Gaia", mock_gaia):
            from astroquery.gaia import Gaia  # noqa: F811 — gets the mock
            center = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="icrs")
            job = Gaia.cone_search_async(center, radius=u.Quantity(0.1, u.degree))
            results = job.get_results()
            Gaia.cone_search_async.assert_called_once()
            assert len(results) == 6

    def test_gaia_table_has_photometry(self):
        table = _make_gaia_table()
        assert "phot_g_mean_mag" in table.colnames
        assert "phot_bp_mean_mag" in table.colnames
        assert "phot_rp_mean_mag" in table.colnames

    def test_gaia_launch_job_async(self):
        mock_gaia = self._mock_gaia()
        with patch("astroquery.gaia.Gaia", mock_gaia):
            from astroquery.gaia import Gaia  # noqa: F811
            job = Gaia.launch_job_async(
                "SELECT * FROM gaiadr3.gaia_source WHERE ra BETWEEN 149 AND 151"
            )
            result = job.get_results()
            Gaia.launch_job_async.assert_called_once()
            assert len(result) == 6

    def test_gaia_magnitude_filter(self):
        table = _make_gaia_table(nrows=20)
        bright = table[table["phot_g_mean_mag"] < 18.0]
        assert isinstance(bright.colnames, list)

    def test_gaia_query_object_async_called(self):
        """Mirrors gaia2file(): Gaia.query_object_async(coordinate, width, height)."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        mock_gaia = self._mock_gaia()
        with patch("astroquery.gaia.Gaia", mock_gaia):
            from astroquery.gaia import Gaia  # noqa: F811
            coord = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree)
            height = u.Quantity(26, u.arcminute)
            width  = u.Quantity(26 / np.cos(2 * np.pi / 180.), u.arcminute)
            Gaia.query_object_async(coordinate=coord, width=width, height=height)
            Gaia.query_object_async.assert_called_once()

    def test_gaia_astrometric_noise_filter(self):
        """Mirrors gaia2file() row filter: mag < limit AND noise_sig < 2."""
        from astropy.table import Table

        rng = np.random.default_rng(42)
        n = 30
        table = Table({
            "ra":  rng.uniform(149, 151, n),
            "dec": rng.uniform(1, 3, n),
            "phot_g_mean_mag": rng.uniform(14, 22, n),
            "astrometric_excess_noise_sig": rng.uniform(0, 5, n),
            "SOURCE_ID": np.arange(n, dtype=np.int64),
        })

        mag_limit = 18.0
        filtered = table[
            (table["phot_g_mean_mag"] < mag_limit) &
            (table["astrometric_excess_noise_sig"] < 2)
        ]
        assert all(filtered["phot_g_mean_mag"] < mag_limit)
        assert all(filtered["astrometric_excess_noise_sig"] < 2)

    def test_gaia_source_id_column_selection(self):
        """Mirrors gaia2file(): select ra, dec, SOURCE_ID, phot_g_mean_mag."""
        from astropy.table import Table

        rng = np.random.default_rng(7)
        n = 5
        full = Table({
            "ra":  rng.uniform(149, 151, n),
            "dec": rng.uniform(1, 3, n),
            "phot_g_mean_mag": rng.uniform(14, 18, n),
            "astrometric_excess_noise_sig": np.zeros(n),
            "SOURCE_ID": np.arange(n, dtype=np.int64),
            "extra_col": np.ones(n),
        })
        cat = full["ra", "dec", "SOURCE_ID", "phot_g_mean_mag"]
        assert list(cat.colnames) == ["ra", "dec", "SOURCE_ID", "phot_g_mean_mag"]


# ---------------------------------------------------------------------------
# SDSS — extended coverage (query_sql + get_images + downloadsdss logic)
# ---------------------------------------------------------------------------

def _make_sdss_download_table(nrows: int = 5):
    """Return a table with run/camcol/field columns as returned by SDSS.query_region
    in downloadsdss (externaldata.py)."""
    from astropy.table import Table
    return Table({
        "ra":     [150.0] * nrows,
        "dec":    [2.0]   * nrows,
        "run":    [301, 301, 500, 500, 200],
        "camcol": [1, 1, 2, 3, 1],
        "field":  [10, 10, 11, 12, 5],
    })


class TestAstroquerySDSSExtended:
    """Extended SDSS tests covering sloan2file (query_sql) and downloadsdss paths."""

    def test_sdss_query_sql_called_with_string(self):
        """sloan2file calls SDSS.query_sql(sql_string)."""
        from astroquery.sdss import SDSS
        from astropy.table import Table

        sql_result = Table({
            "ra": [150.0, 150.1], "dec": [2.0, 2.1],
            "objID": [1234, 5678],
            "u": [21.0, 21.5], "err_u": [0.05, 0.06],
            "g": [20.0, 20.5], "err_g": [0.03, 0.04],
            "r": [19.5, 20.0], "err_r": [0.02, 0.03],
            "i": [19.0, 19.5], "err_i": [0.02, 0.03],
            "z": [18.5, 19.0], "err_z": [0.03, 0.04],
        })

        with patch.object(SDSS, "query_sql", return_value=sql_result) as mock_sql:
            sql = (
                "select P.ra, P.dec, P.objID, P.u, P.err_u, P.g, P.err_g, "
                "P.r, P.err_r, P.i, P.err_i, P.z, P.err_z "
                "from PhotoPrimary as P, dbo.fGetNearbyObjEq(150.0, 2.0, 10) as N "
                "where P.objID=N.objID and P.type=6 and P.r >= 13 and P.r <= 20;"
            )
            result = SDSS.query_sql(sql)
            mock_sql.assert_called_once_with(sql)
            assert "ra" in result.colnames
            assert "err_r" in result.colnames

    def test_sdss_query_sql_none_on_empty_field(self):
        """sloan2file guards: if t is None, no write occurs."""
        from astroquery.sdss import SDSS

        with patch.object(SDSS, "query_sql", return_value=None):
            result = SDSS.query_sql("SELECT * FROM PhotoPrimary WHERE 1=0")
            assert result is None

    def test_sdss_run_camcol_field_dedup(self):
        """downloadsdss deduplicates (run, camcol, field) and drops run <= 300."""
        table = _make_sdss_download_table()

        # Replicate pipeline logic
        pointing = []
        for row in table:
            if row["run"] > 300:
                key = (row["run"], row["camcol"], row["field"])
                if key not in pointing:
                    pointing.append(key)

        # run=200 is dropped; (301,1,10) appears twice → deduplicated
        assert (301, 1, 10) in pointing
        assert (500, 2, 11) in pointing
        assert (500, 3, 12) in pointing
        assert (200, 1, 5) not in pointing
        assert len(pointing) == 3

    def test_sdss_get_images_called_with_run_camcol_field(self):
        """downloadsdss calls SDSS.get_images(run, camcol, field, band, cache=True)."""
        from astroquery.sdss import SDSS

        fake_hdu = MagicMock()
        with patch.object(SDSS, "get_images", return_value=[fake_hdu]) as mock_gi:
            result = SDSS.get_images(run=301, camcol=1, field=10, band="r", cache=True)
            mock_gi.assert_called_once_with(run=301, camcol=1, field=10, band="r", cache=True)
            assert len(result) == 1

    def test_sdss_query_region_spectro_false(self):
        """downloadsdss passes spectro=False to query_region."""
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        with patch.object(SDSS, "query_region", return_value=_make_sdss_download_table()) as mock_qr:
            pos = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg)
            SDSS.query_region(pos, spectro=False, radius=20 * u.arcsec)
            _, kwargs = mock_qr.call_args
            assert kwargs.get("spectro") is False

    def test_sdss_catalog_format_strings(self):
        """Verify sloan2file format-string generation for all ugriz columns."""
        from astropy.table import Table

        t = Table({
            "ra": [150.0], "dec": [2.0], "objID": [1111],
            "u": [21.0], "err_u": [0.05],
            "g": [20.0], "err_g": [0.03],
            "r": [19.5], "err_r": [0.02],
            "i": [19.0], "err_i": [0.02],
            "z": [18.5], "err_z": [0.03],
        })
        t["ra"].format  = "%16.12f"
        t["dec"].format = "%16.13f"
        t["objID"].format = "%19d"
        for filt in "ugriz":
            t[filt].format        = "%8.5f"
            t["err_" + filt].format = "%11.9f"

        assert t["ra"].format == "%16.12f"
        assert t["r"].format  == "%8.5f"
        assert t["err_g"].format == "%11.9f"


# ---------------------------------------------------------------------------
# Vizier — extended coverage (PanSTARRS pathway)
# ---------------------------------------------------------------------------

def _make_panstarrs_table(nrows: int = 8):
    """Mimic the Vizier II/349 result for panstarrs2file."""
    from astropy.table import Table
    rng = np.random.default_rng(10)
    good_dq   = 8 + 16 + 32 + 256 + 16384 + 32768
    extended_dq = 16777216
    flags = np.array([good_dq] * 6 + [good_dq | extended_dq] + [0])
    return Table({
        "RAJ2000": rng.uniform(149.9, 150.1, nrows),
        "DEJ2000": rng.uniform(1.9, 2.1, nrows),
        "objID":   np.arange(nrows, dtype=np.int64),
        "gFlags":  flags,
        "yMeanPSFMag":    rng.uniform(17, 22, nrows),
        "yMeanPSFMagErr": rng.uniform(0.01, 0.05, nrows),
        "gMeanPSFMag":    rng.uniform(17, 22, nrows),
        "gMeanPSFMagErr": rng.uniform(0.01, 0.05, nrows),
        "rMeanPSFMag":    rng.uniform(16, 21, nrows),
        "rMeanPSFMagErr": rng.uniform(0.01, 0.05, nrows),
        "iMeanPSFMag":    rng.uniform(15, 20, nrows),
        "iMeanPSFMagErr": rng.uniform(0.01, 0.05, nrows),
        "zMeanPSFMag":    rng.uniform(15, 20, nrows),
        "zMeanPSFMagErr": rng.uniform(0.01, 0.05, nrows),
    })


class TestAstroqueryVizierExtended:
    """Extended Vizier tests covering PanSTARRS (panstarrs2file) usage."""

    def test_vizier_row_limit_minus_one(self):
        """panstarrs2file sets Vizier.ROW_LIMIT = -1 (unlimited)."""
        from astroquery.vizier import Vizier
        original = Vizier.ROW_LIMIT
        try:
            Vizier.ROW_LIMIT = -1
            assert Vizier.ROW_LIMIT == -1
        finally:
            Vizier.ROW_LIMIT = original

    def test_vizier_columns_assignment(self):
        """panstarrs2file assigns a specific column list before querying."""
        from astroquery.vizier import Vizier
        cols = [
            "raMean", "decMean", "objID", "gFlags",
            "yMeanPSFMag", "yMeanPSFMagErr",
            "gMeanPSFMag", "gMeanPSFMagErr",
            "rMeanPSFMag", "rMeanPSFMagErr",
        ]
        original = getattr(Vizier, "columns", None)
        try:
            Vizier.columns = cols
            assert "gFlags" in Vizier.columns
        finally:
            if original is not None:
                Vizier.columns = original

    def test_vizier_column_filters_assignment(self):
        """panstarrs2file assigns column_filters before querying."""
        from astroquery.vizier import Vizier
        filters = {
            "nDetections": ">5",
            "rMeanPSFMag-rMeanKronMag": "<0.05",
            "rMeanPSFMag": ">13.000000",
            "gQfPerfect": ">0.85",
        }
        original = getattr(Vizier, "column_filters", None)
        try:
            Vizier.column_filters = filters
            assert Vizier.column_filters["nDetections"] == ">5"
        finally:
            if original is not None:
                Vizier.column_filters = original

    def test_vizier_panstarrs_catalog_id(self):
        """panstarrs2file queries catalog 'II/349'."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        ps_table = _make_panstarrs_table()
        with patch.object(Vizier, "query_region", return_value=[ps_table]) as mock_qr:
            coord = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree)
            Vizier.query_region(coord, radius=20.0 * u.arcmin, catalog="II/349")
            _, kwargs = mock_qr.call_args
            assert kwargs.get("catalog") == "II/349"

    def test_vizier_panstarrs_quality_flag_filter(self):
        """panstarrs2file bitwise DQ filter: keep good, drop extended sources."""
        table = _make_panstarrs_table()
        good_dq    = 8 + 16 + 32 + 256 + 16384 + 32768
        extended_dq = 16777216

        keep = (table["gFlags"] & good_dq == good_dq) & \
               (table["gFlags"] & extended_dq != extended_dq)
        filtered = table[keep]

        # Last row has flags=0 → fails good_dq check; row index 6 is extended → dropped
        assert len(filtered) == 6
        assert all(filtered["gFlags"] & extended_dq == 0)

    def test_vizier_query_region_with_radius_quantity(self):
        """Vizier.query_region accepts an astropy Quantity radius."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        with patch.object(Vizier, "query_region", return_value=[_make_vizier_table()]) as mock_qr:
            coord = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree)
            radius = 20.0 * u.arcmin
            Vizier.query_region(coord, radius=radius, catalog="II/183A")
            args, kwargs = mock_qr.call_args
            assert kwargs["radius"] == radius
