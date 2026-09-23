"""
Tests for astropy sub-modules used in lcogtsnpipe.

Covers:
- astropy.io.fits       (read/write FITS files everywhere in pipeline)
- astropy.table.Table   (catalog I/O in banzaicat.py, lscastrodef.py)
- astropy.wcs.WCS       (world coordinate transforms in lscastrodef, lscdiff)
- astropy.nddata.Cutout2D (util.imcopy)
- astropy.coordinates   (SkyCoord crossmatch in lscastrodef / myloopdef)
- astropy.units         (angular unit conversions throughout)
"""
import numpy as np
import pytest
import io
import tempfile
import os

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# astropy.io.fits
# ---------------------------------------------------------------------------

class TestAstropyFits:
    """FITS file creation, header access, update, and round-trip."""

    def test_create_primary_hdu(self):
        from astropy.io import fits
        data = np.zeros((10, 10), dtype=np.float32)
        hdu = fits.PrimaryHDU(data)
        assert hdu.data.shape == (10, 10)

    def test_read_write_round_trip(self, tmp_path):
        from astropy.io import fits
        data = np.arange(25, dtype=np.float32).reshape(5, 5)
        path = tmp_path / "test.fits"
        fits.writeto(str(path), data, overwrite=True)
        result = fits.getdata(str(path))
        np.testing.assert_array_equal(result, data)

    def test_header_keyword_access(self, tmp_path):
        from astropy.io import fits
        hdr = fits.Header()
        hdr["EXPTIME"] = 120.0
        hdr["FILTER"] = "r"
        path = tmp_path / "hdr.fits"
        fits.writeto(str(path), np.zeros((4, 4)), header=hdr, overwrite=True)
        hdr2 = fits.getheader(str(path))
        assert hdr2["EXPTIME"] == pytest.approx(120.0)
        assert hdr2["FILTER"] == "r"

    def test_update_header_keyword(self, tmp_path):
        from astropy.io import fits
        path = tmp_path / "upd.fits"
        fits.writeto(str(path), np.zeros((4, 4)), overwrite=True)
        with fits.open(str(path), mode="update") as hdul:
            hdul[0].header["GAIN"] = 2.0
        assert fits.getheader(str(path))["GAIN"] == pytest.approx(2.0)

    def test_image_extension(self, tmp_path):
        from astropy.io import fits
        prim = fits.PrimaryHDU()
        img = fits.ImageHDU(data=np.ones((8, 8)), name="SCI")
        hdul = fits.HDUList([prim, img])
        path = tmp_path / "multi.fits"
        hdul.writeto(str(path))
        with fits.open(str(path)) as f:
            assert f["SCI"].data.shape == (8, 8)

    def test_getval_reads_single_keyword(self, tmp_path):
        from astropy.io import fits
        path = tmp_path / "kw.fits"
        fits.writeto(str(path), np.zeros((3, 3)), overwrite=True)
        with fits.open(str(path), mode="update") as hdul:
            hdul[0].header["RDNOISE"] = 7.5
        val = fits.getval(str(path), "RDNOISE")
        assert val == pytest.approx(7.5)

    def test_binary_table_read(self, tmp_path):
        from astropy.io import fits
        col = fits.Column(name="RA", format="D", array=np.array([150.0, 151.0]))
        tbl = fits.BinTableHDU.from_columns([col])
        path = tmp_path / "table.fits"
        fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(str(path))
        with fits.open(str(path)) as f:
            ra = f[1].data["RA"]
        np.testing.assert_array_equal(ra, [150.0, 151.0])

    def test_missing_keyword_raises(self, tmp_path):
        from astropy.io import fits
        path = tmp_path / "missing.fits"
        fits.writeto(str(path), np.zeros((3, 3)), overwrite=True)
        hdr = fits.getheader(str(path))
        with pytest.raises(KeyError):
            _ = hdr["NOTHERE"]


# ---------------------------------------------------------------------------
# astropy.table.Table
# ---------------------------------------------------------------------------

class TestAstropyTable:
    """Table creation, column access, filtering, and ASCII round-trip."""

    def test_create_from_dict(self):
        from astropy.table import Table
        t = Table({"ra": [150.0, 151.0], "dec": [2.0, 3.0], "mag": [18.0, 19.5]})
        assert len(t) == 2
        assert "ra" in t.colnames

    def test_column_access(self):
        from astropy.table import Table
        t = Table({"x": [1, 2, 3], "y": [4, 5, 6]})
        np.testing.assert_array_equal(t["x"], [1, 2, 3])

    def test_boolean_filter(self):
        from astropy.table import Table
        t = Table({"mag": [15.0, 17.0, 21.0]})
        bright = t[t["mag"] < 18.0]
        assert len(bright) == 2

    def test_add_column(self):
        from astropy.table import Table
        t = Table({"a": [1, 2, 3]})
        t["b"] = [10, 20, 30]
        assert "b" in t.colnames

    def test_read_write_ascii(self, tmp_path):
        from astropy.table import Table
        t = Table({"ra": [10.0, 20.0], "dec": [0.5, -0.5]})
        path = tmp_path / "cat.csv"
        t.write(str(path), format="csv", overwrite=True)
        t2 = Table.read(str(path), format="csv")
        np.testing.assert_array_almost_equal(t2["ra"], [10.0, 20.0])

    def test_read_write_fits(self, tmp_path):
        from astropy.table import Table
        t = Table({"flux": np.array([1.0, 2.0, 3.0])})
        path = tmp_path / "tab.fits"
        t.write(str(path), overwrite=True)
        t2 = Table.read(str(path))
        np.testing.assert_array_almost_equal(t2["flux"], [1.0, 2.0, 3.0])

    def test_stack_tables(self):
        from astropy.table import Table, vstack
        t1 = Table({"a": [1, 2]})
        t2 = Table({"a": [3, 4]})
        combined = vstack([t1, t2])
        assert len(combined) == 4

    def test_masked_table(self):
        from astropy.table import Table
        import numpy.ma as ma
        t = Table({"val": ma.array([1.0, 2.0, 3.0], mask=[False, True, False])})
        assert t["val"][1] is ma.masked


# ---------------------------------------------------------------------------
# astropy.wcs.WCS
# ---------------------------------------------------------------------------

class TestAstropyWCS:
    """WCS pixel <-> sky transforms as used in lscastrodef and lscdiff."""

    def _make_wcs(self):
        from astropy.wcs import WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50.0, 50.0]
        wcs.wcs.cdelt = [-0.000108, 0.000108]   # ~0.389 arcsec/pix in deg
        wcs.wcs.crval = [150.0, 2.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs

    def test_pix2world_crpix_gives_crval(self):
        wcs = self._make_wcs()
        ra, dec = wcs.all_pix2world(50.0, 50.0, 1)
        assert float(ra) == pytest.approx(150.0, abs=1e-6)
        assert float(dec) == pytest.approx(2.0, abs=1e-6)

    def test_world2pix_round_trip(self):
        wcs = self._make_wcs()
        x0, y0 = 35.7, 62.1
        ra, dec = wcs.all_pix2world(x0, y0, 1)
        x1, y1 = wcs.all_world2pix(ra, dec, 1)
        assert float(x1) == pytest.approx(x0, abs=1e-5)
        assert float(y1) == pytest.approx(y0, abs=1e-5)

    def test_wcs_has_two_axes(self):
        wcs = self._make_wcs()
        assert wcs.naxis == 2

    def test_wcs_from_header(self, tmp_path):
        from astropy.io import fits
        from astropy.wcs import WCS
        path = tmp_path / "wcs.fits"
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 100
        hdr["NAXIS2"] = 100
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = 150.0
        hdr["CRVAL2"] = 2.0
        hdr["CRPIX1"] = 50.0
        hdr["CRPIX2"] = 50.0
        hdr["CDELT1"] = -0.000108
        hdr["CDELT2"] = 0.000108
        fits.writeto(str(path), np.zeros((100, 100)), header=hdr, overwrite=True)
        with fits.open(str(path)) as f:
            wcs = WCS(f[0].header)
        assert wcs.naxis == 2

    def test_array_pixel_to_world_batch(self):
        wcs = self._make_wcs()
        xs = np.array([50.0, 51.0, 52.0])
        ys = np.array([50.0, 50.0, 50.0])
        ras, decs = wcs.all_pix2world(xs, ys, 1)
        assert len(ras) == 3
        # moving east → RA increases (since cdelt is negative, RA decreases with x)
        assert ras[0] > ras[1]  # cdelt1 is negative so RA decreases as x increases

    def test_wcs_header_round_trip_preserves_crpix(self):
        from astropy.wcs import WCS

        wcs0 = self._make_wcs()
        hdr = wcs0.to_header()
        wcs1 = WCS(hdr)

        assert float(wcs1.wcs.crpix[0]) == pytest.approx(50.0)
        assert float(wcs1.wcs.crpix[1]) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# astropy.nddata.Cutout2D
# ---------------------------------------------------------------------------

class TestAstropyNddata:
    """Cutout2D as used by util.imcopy."""

    def test_cutout_shape(self):
        from astropy.nddata import Cutout2D
        data = np.arange(100).reshape(10, 10).astype(float)
        cutout = Cutout2D(data, position=(5, 5), size=(4, 4))
        assert cutout.data.shape == (4, 4)

    def test_cutout_center_value(self):
        from astropy.nddata import Cutout2D
        data = np.zeros((20, 20))
        data[10, 10] = 999.0
        cutout = Cutout2D(data, position=(10, 10), size=(6, 6))
        # center of cutout should be 999
        cy, cx = 3, 3
        assert cutout.data[cy, cx] == 999.0

    def test_cutout_wcs_updates(self):
        from astropy.nddata import Cutout2D
        from astropy.wcs import WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50.0, 50.0]
        wcs.wcs.cdelt = [-0.000108, 0.000108]
        wcs.wcs.crval = [150.0, 2.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        data = np.ones((100, 100))
        cutout = Cutout2D(data, position=(50, 50), size=20, wcs=wcs)
        assert cutout.wcs is not None

    def test_cutout_respects_boundary(self):
        from astropy.nddata import Cutout2D
        data = np.ones((10, 10)) * 5.0
        cutout = Cutout2D(data, position=(2, 2), size=4)
        assert cutout.data.shape == (4, 4)

    def test_cutout_position_attribute(self):
        from astropy.nddata import Cutout2D
        data = np.zeros((30, 30))
        cutout = Cutout2D(data, position=(15, 15), size=10)
        # center_cutout is the geometric center (float), position_cutout is int
        cy, cx = cutout.center_cutout
        assert cy == pytest.approx(4.5)
        assert cx == pytest.approx(4.5)

    def test_cutout_partial_mode_uses_fill_value(self):
        from astropy.nddata import Cutout2D

        data = np.ones((5, 5))
        cutout = Cutout2D(
            data,
            position=(0, 0),
            size=(4, 4),
            mode="partial",
            fill_value=-1.0,
        )

        assert cutout.data.shape == (4, 4)
        assert np.any(cutout.data == -1.0)


# ---------------------------------------------------------------------------
# astropy.coordinates
# ---------------------------------------------------------------------------

class TestAstropyCoordinates:
    """SkyCoord creation and matching as used in myloopdef / lscastrodef."""

    def test_skycoord_from_ra_dec(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="icrs")
        assert c.ra.deg == pytest.approx(150.0)
        assert c.dec.deg == pytest.approx(2.0)

    def test_separation_zero_same_coord(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c1 = SkyCoord(ra=100.0 * u.degree, dec=10.0 * u.degree, frame="icrs")
        c2 = SkyCoord(ra=100.0 * u.degree, dec=10.0 * u.degree, frame="icrs")
        sep = c1.separation(c2)
        assert sep.arcsec == pytest.approx(0.0, abs=1e-10)

    def test_separation_one_arcsec(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c1 = SkyCoord(ra=0.0 * u.degree, dec=0.0 * u.degree, frame="icrs")
        c2 = SkyCoord(ra=1.0 / 3600.0 * u.degree, dec=0.0 * u.degree, frame="icrs")
        sep = c1.separation(c2)
        assert sep.arcsec == pytest.approx(1.0, rel=0.001)

    def test_match_catalog_nearest(self):
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        import astropy.units as u
        catalog = SkyCoord(
            ra=[10.0, 20.0, 30.0] * u.degree,
            dec=[0.0, 0.0, 0.0] * u.degree
        )
        target = SkyCoord(ra=20.0001 * u.degree, dec=0.0 * u.degree)
        idx, sep, _ = match_coordinates_sky(target, catalog)
        assert int(idx) == 1  # closest is the second entry

    def test_fk5_to_icrs_transform(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra=150.0 * u.degree, dec=2.0 * u.degree, frame="fk5")
        c_icrs = c.icrs
        # after frame transform, coordinates should be close (small epoch difference)
        assert abs(c_icrs.ra.deg - 150.0) < 0.1

    def test_catalog_match_returns_correct_length(self):
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        import astropy.units as u
        rng = np.random.default_rng(0)
        ref = SkyCoord(ra=rng.uniform(0, 360, 50) * u.degree,
                       dec=rng.uniform(-30, 30, 50) * u.degree)
        src = SkyCoord(ra=rng.uniform(0, 360, 10) * u.degree,
                       dec=rng.uniform(-30, 30, 10) * u.degree)
        idx, sep, _ = match_coordinates_sky(src, ref)
        assert len(idx) == 10

    def test_angle_parsing_from_sexagesimal(self):
        from astropy.coordinates import Angle
        import astropy.units as u

        ra = Angle("12:00:00", unit=u.hourangle)
        dec = Angle("+30:00:00", unit=u.deg)

        assert ra.degree == pytest.approx(180.0)
        assert dec.degree == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# astropy.visualization
# ---------------------------------------------------------------------------

class TestAstropyVisualization:
    """Visualization helpers used in myloopdef image display paths."""

    def test_zscale_interval_limits_are_ordered(self):
        from astropy.visualization import ZScaleInterval

        rng = np.random.default_rng(123)
        arr = rng.normal(1000.0, 20.0, (100, 100))
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(arr)

        assert vmin < vmax

    def test_image_normalize_maps_into_unit_interval(self):
        from astropy.visualization import ImageNormalize, ZScaleInterval

        data = np.linspace(0.0, 100.0, 25).reshape(5, 5)
        norm = ImageNormalize(data, interval=ZScaleInterval())
        mapped = norm(data)

        assert np.nanmin(mapped) >= 0.0
        assert np.nanmax(mapped) <= 1.0

    def test_image_normalize_clips_outliers_when_clip_true(self):
        from astropy.visualization import ImageNormalize, ZScaleInterval

        data = np.array([[0.0, 1.0], [2.0, 1.0e6]])
        norm = ImageNormalize(data, interval=ZScaleInterval(), clip=True)
        mapped = norm(data)

        assert np.nanmax(mapped) <= 1.0
        assert np.nanmin(mapped) >= 0.0


# ---------------------------------------------------------------------------
# astropy.units
# ---------------------------------------------------------------------------

class TestAstropyUnits:
    """Unit conversions used throughout the pipeline."""

    def test_deg_to_arcsec(self):
        import astropy.units as u
        val = 1.0 * u.degree
        assert val.to(u.arcsec).value == pytest.approx(3600.0)

    def test_arcsec_to_rad(self):
        import astropy.units as u
        import math
        val = (180.0 * 3600.0) * u.arcsec
        assert val.to(u.rad).value == pytest.approx(math.pi)

    def test_pixel_scale_conversion(self):
        import astropy.units as u
        # 0.389 arcsec/pix is typical for LCO 1-m
        scale = 0.389 * u.arcsec / u.pixel
        assert scale.value == pytest.approx(0.389)

    def test_magnitude_quantity(self):
        import astropy.units as u
        mag = 18.5 * u.mag
        assert mag.unit == u.mag

    def test_unit_equivalence(self):
        import astropy.units as u
        assert u.degree.is_equivalent(u.arcsec)

    def test_to_string(self):
        import astropy.units as u
        assert str(u.degree) == "deg"

    def test_quantity_arithmetic(self):
        import astropy.units as u
        a = 2.0 * u.hour
        b = 30.0 * u.minute
        total = (a + b).to(u.hour)
        assert total.value == pytest.approx(2.5)

    def test_compound_unit_conversion(self):
        import astropy.units as u

        speed = 10.0 * u.m / u.s
        speed_kmh = speed.to(u.km / u.hour)

        assert speed_kmh.value == pytest.approx(36.0)
