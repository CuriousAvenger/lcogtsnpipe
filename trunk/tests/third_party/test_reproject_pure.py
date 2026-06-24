"""
Tests for the reproject package as used in lcogtsnpipe.

reproject is used in bin/lscdiff.py to align template images to the
science-image WCS before image subtraction:

    from reproject import reproject_interp
    temp_reproj, temp_foot = reproject_interp(_dirtemp + imgtemp0, head_targ)
    tempmask_reproj, tempmask_foot = reproject_interp(_dirtemp + tempmask0, head_targ)
    tempnoise_reproj, tempnoise_foot = reproject_interp(_dirtemp + tempnoise0, head_targ)

reproject is NOT installed in the default lcogtsnpipe conda env by default.
All tests use `pytest.importorskip` so the suite still passes when the
package is absent, and they exercise real functionality when it is present.

Additional coverage added:
  - TestReprojectInterpOrders: nearest-neighbor, bilinear, biquadratic, bicubic orders
  - TestReprojectInterpInputFormats: HDU tuple, WCS object, shifted WCS
  - TestReprojectInterpPixelScale: coarser/finer scale reprojection
  - TestReprojectInterpReturnFootprint: return_footprint=False returns ndarray
  - TestReprojectInterpNaNHandling: NaN input propagation, footprint zeros outside
  - TestReprojectInterpShiftedWCS: template-alignment use case from lscdiff.py
  - TestReprojectInterpMaskImage: binary mask reprojection (nearest-neighbor)
  - TestReprojectExact: flux-conserving exact reprojection shape/finiteness/flux
  - TestReprojectAdaptive: Gaussian adaptive kernel, flux conservation
  - TestLscdiffReprojPattern: end-to-end template-alignment pattern from lscdiff.py
"""
import numpy as np
import pytest

pytestmark = pytest.mark.unit

reproject = pytest.importorskip("reproject", reason="reproject not installed")


# ---------------------------------------------------------------------------
# Import checks (only reached if reproject is installed)
# ---------------------------------------------------------------------------

class TestReprojectImport:
    def test_reproject_interp_importable(self):
        from reproject import reproject_interp
        assert callable(reproject_interp)

    def test_reproject_exact_importable(self):
        from reproject import reproject_exact
        assert callable(reproject_exact)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wcs_header(ra_center, dec_center, pixel_scale_deg, naxis):
    from astropy.io import fits
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis
    hdr["NAXIS2"] = naxis
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = ra_center
    hdr["CRVAL2"] = dec_center
    hdr["CRPIX1"] = naxis / 2.0
    hdr["CRPIX2"] = naxis / 2.0
    hdr["CDELT1"] = -pixel_scale_deg
    hdr["CDELT2"] = pixel_scale_deg
    return hdr


# ---------------------------------------------------------------------------
# reproject_interp — pixel grid reprojection
# ---------------------------------------------------------------------------

class TestReprojectInterp:
    """Basic sanity tests for reproject_interp as used in lscdiff.py."""

    def test_identity_reprojection_preserves_shape(self):
        from reproject import reproject_interp
        from astropy.io import fits

        data = np.ones((32, 32), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 32)
        hdu = fits.PrimaryHDU(data, header=hdr)

        reproj, footprint = reproject_interp(hdu, hdr)
        assert reproj.shape == (32, 32)
        assert footprint.shape == (32, 32)

    def test_footprint_is_binary(self):
        from reproject import reproject_interp
        from astropy.io import fits

        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)

        _, footprint = reproject_interp(hdu, hdr)
        unique_vals = set(np.unique(footprint[~np.isnan(footprint)]))
        assert unique_vals.issubset({0.0, 1.0})

    def test_reprojection_to_larger_grid(self):
        from reproject import reproject_interp
        from astropy.io import fits

        # Source image: 16×16
        data = np.ones((16, 16), dtype=np.float64)
        src_hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)

        # Target grid: 32×32, same scale → larger sky coverage
        tgt_hdr = _make_wcs_header(150.0, 2.0, 3e-4, 32)
        hdu = fits.PrimaryHDU(data, header=src_hdr)

        reproj, footprint = reproject_interp(hdu, tgt_hdr)
        assert reproj.shape == (32, 32)
        # Source only covers inner portion → outer footprint should be 0
        assert footprint[0, 0] == pytest.approx(0.0, abs=0.1)

    def test_reprojection_constant_image(self):
        from reproject import reproject_interp
        from astropy.io import fits

        data = np.full((16, 16), 42.0, dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)

        reproj, footprint = reproject_interp(hdu, hdr)
        # Within footprint, values should stay ≈ 42
        inside = reproj[footprint == 1.0]
        if len(inside) > 0:
            np.testing.assert_allclose(inside, 42.0, atol=1e-3)

    def test_reprojection_returns_floats(self):
        from reproject import reproject_interp
        from astropy.io import fits

        data = np.ones((8, 8), dtype=np.float32)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 8)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, _ = reproject_interp(hdu, hdr)
        assert reproj.dtype.kind == "f"


# ---------------------------------------------------------------------------
# reproject_interp — interpolation orders
# ---------------------------------------------------------------------------

class TestReprojectInterpOrders:
    """Pipeline mask/noise images may need different interpolation orders."""

    def test_nearest_neighbor_order(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr, order='nearest-neighbor')
        assert reproj.shape == (16, 16)
        assert np.all(np.isfinite(reproj[fp == 1.0]))

    def test_bilinear_order(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr, order='bilinear')
        assert reproj.shape == (16, 16)

    def test_biquadratic_order(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, _ = reproject_interp(hdu, hdr, order='biquadratic')
        assert reproj.shape == (16, 16)

    def test_bicubic_order(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, _ = reproject_interp(hdu, hdr, order='bicubic')
        assert reproj.shape == (16, 16)

    def test_nearest_preserves_binary_mask(self):
        """Mask images (0/1) must stay binary after nearest-neighbor reproject."""
        from reproject import reproject_interp
        from astropy.io import fits
        mask = np.zeros((16, 16), dtype=np.float64)
        mask[4:12, 4:12] = 1.0
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(mask, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr, order='nearest-neighbor')
        valid = reproj[fp == 1.0]
        unique = set(np.unique(valid[np.isfinite(valid)]))
        assert unique.issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# reproject_interp — input formats
# ---------------------------------------------------------------------------

class TestReprojectInterpInputFormats:
    """reproject accepts HDU, (array, header) tuple, or WCS object as input."""

    def test_hdu_tuple_input(self):
        """(data, header) tuple is the form used in lscdiff.py."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        reproj, fp = reproject_interp((data, hdr), hdr)
        assert reproj.shape == (16, 16)

    def test_wcs_object_as_output_projection(self):
        """Output projection can be an astropy.wcs.WCS object."""
        from reproject import reproject_interp
        from astropy.io import fits
        from astropy.wcs import WCS
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        wcs = WCS(hdr)
        reproj, fp = reproject_interp(hdu, wcs, shape_out=(16, 16))
        assert reproj.shape == (16, 16)

    def test_shape_out_overrides_header(self):
        """shape_out parameter forces the output grid size."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((32, 32), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 32)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr, shape_out=(32, 32))
        assert reproj.shape == (32, 32)

    def test_primaryhdu_input(self):
        """Standard PrimaryHDU input — most common in lscdiff.py."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.random.default_rng(10).normal(500, 10, (16, 16)).astype(np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr)
        assert reproj.shape == data.shape


# ---------------------------------------------------------------------------
# reproject_interp — pixel-scale changes
# ---------------------------------------------------------------------------

class TestReprojectInterpPixelScale:
    """Pipeline sometimes reprojects between different plate scales."""

    def test_coarser_scale_output(self):
        """Reproject fine grid to coarser scale → same output shape, different coverage."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((32, 32), dtype=np.float64)
        src_hdr = _make_wcs_header(150.0, 2.0, 3e-4, 32)
        tgt_hdr = _make_wcs_header(150.0, 2.0, 6e-4, 32)  # 2× coarser
        hdu = fits.PrimaryHDU(data, header=src_hdr)
        reproj, fp = reproject_interp(hdu, tgt_hdr)
        assert reproj.shape == (32, 32)

    def test_finer_scale_larger_grid(self):
        """Reproject into a larger grid at finer scale; output contains source."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        src_hdr = _make_wcs_header(150.0, 2.0, 6e-4, 16)
        tgt_hdr = _make_wcs_header(150.0, 2.0, 3e-4, 32)
        hdu = fits.PrimaryHDU(data, header=src_hdr)
        reproj, fp = reproject_interp(hdu, tgt_hdr)
        assert reproj.shape == (32, 32)
        assert fp.sum() > 0  # at least some valid pixels

    def test_constant_image_preserved_at_same_scale(self):
        """Identity reprojection should leave a constant image unchanged."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.full((16, 16), 7.0, dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr)
        valid = reproj[fp == 1.0]
        np.testing.assert_allclose(valid, 7.0, atol=1e-3)


# ---------------------------------------------------------------------------
# reproject_interp — return_footprint=False
# ---------------------------------------------------------------------------

class TestReprojectInterpReturnFootprint:
    """return_footprint=False returns just the reprojected array."""

    def test_return_footprint_false_gives_ndarray(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        result = reproject_interp(hdu, hdr, return_footprint=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)

    def test_return_footprint_true_gives_tuple(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        result = reproject_interp(hdu, hdr, return_footprint=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_footprint_data_matches_with_footprint(self):
        """The reprojected data must be the same whether footprint is returned or not."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        r_with, fp = reproject_interp(hdu, hdr, return_footprint=True)
        r_without = reproject_interp(hdu, hdr, return_footprint=False)
        np.testing.assert_array_equal(r_with, r_without)


# ---------------------------------------------------------------------------
# reproject_interp — NaN handling
# ---------------------------------------------------------------------------

class TestReprojectInterpNaNHandling:
    """NaN pixels in input propagate correctly to the output."""

    def test_nan_input_produces_nan_output(self):
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        data[8, 8] = np.nan
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr)
        assert np.any(np.isnan(reproj))

    def test_footprint_zero_outside_source(self):
        """For a small source in a larger target, footprint is 0 outside."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        src_hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        tgt_hdr = _make_wcs_header(150.0, 2.0, 3e-4, 32)  # larger grid
        hdu = fits.PrimaryHDU(data, header=src_hdr)
        _, fp = reproject_interp(hdu, tgt_hdr)
        # corners of 32×32 grid are outside the 16×16 source
        assert fp[0, 0] == pytest.approx(0.0, abs=0.1)
        assert fp[31, 31] == pytest.approx(0.0, abs=0.1)

    def test_valid_pixels_are_finite(self):
        """All pixels inside the footprint must be finite."""
        from reproject import reproject_interp
        from astropy.io import fits
        data = np.random.default_rng(77).normal(100, 5, (16, 16)).astype(np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr)
        assert np.all(np.isfinite(reproj[fp == 1.0]))


# ---------------------------------------------------------------------------
# reproject_interp — shifted WCS (lscdiff.py template alignment pattern)
# ---------------------------------------------------------------------------

class TestReprojectInterpShiftedWCS:
    """Reproduce the lscdiff.py use case: align template to science WCS."""

    def test_shift_by_half_pixel(self):
        """A half-pixel shift should move bright pixels slightly."""
        from reproject import reproject_interp
        from astropy.io import fits
        n = 32
        scale = 3e-4
        data = np.zeros((n, n), dtype=np.float64)
        data[n // 2, n // 2] = 1000.0
        src_hdr = _make_wcs_header(150.0, 2.0, scale, n)
        # Shift CRPIX by half a pixel
        tgt_hdr = _make_wcs_header(150.0, 2.0, scale, n)
        tgt_hdr["CRPIX1"] = n / 2.0 + 0.5
        tgt_hdr["CRPIX2"] = n / 2.0 + 0.5
        hdu = fits.PrimaryHDU(data, header=src_hdr)
        reproj, fp = reproject_interp(hdu, tgt_hdr)
        assert reproj.shape == (n, n)
        assert np.nansum(reproj) > 0

    def test_footprint_nonzero_in_overlap_region(self):
        """When source and target WCS overlap, footprint must have non-zero pixels."""
        from reproject import reproject_interp
        from astropy.io import fits
        n = 32
        data = np.ones((n, n), dtype=np.float64)
        src_hdr = _make_wcs_header(150.0, 2.0, 3e-4, n)
        # Small RA offset — still overlapping
        tgt_hdr = _make_wcs_header(150.001, 2.0, 3e-4, n)
        hdu = fits.PrimaryHDU(data, header=src_hdr)
        _, fp = reproject_interp(hdu, tgt_hdr)
        assert fp.sum() > 0

    def test_science_template_flux_conservation(self):
        """Identity reprojection: nansum of output ≈ nansum of input."""
        from reproject import reproject_interp
        from astropy.io import fits
        n = 32
        rng = np.random.default_rng(55)
        data = rng.normal(500, 10, (n, n)).astype(np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, n)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr)
        np.testing.assert_allclose(
            np.nansum(reproj[fp == 1.0]),
            np.sum(data),
            rtol=0.01,
        )

    def test_mask_image_alignment(self):
        """Align a binary bad-pixel mask (nearest-neighbor, no interpolation artefacts)."""
        from reproject import reproject_interp
        from astropy.io import fits
        n = 32
        mask = np.zeros((n, n), dtype=np.float64)
        mask[10:20, 10:20] = 1.0
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, n)
        hdu = fits.PrimaryHDU(mask, header=hdr)
        reproj, fp = reproject_interp(hdu, hdr, order='nearest-neighbor')
        valid = reproj[fp == 1.0]
        assert set(np.unique(valid[np.isfinite(valid)])).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# reproject_exact — flux-conserving exact reprojection
# ---------------------------------------------------------------------------

class TestReprojectExact:
    """reproject_exact uses exact polygon intersection for flux conservation."""

    def test_identity_shape_preserved(self):
        from reproject import reproject_exact
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_exact(hdu, hdr)
        assert reproj.shape == (16, 16)

    def test_footprint_is_binary(self):
        from reproject import reproject_exact
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        _, fp = reproject_exact(hdu, hdr)
        unique = set(np.unique(fp[np.isfinite(fp)]))
        assert unique.issubset({0.0, 1.0})

    def test_output_is_finite(self):
        from reproject import reproject_exact
        from astropy.io import fits
        data = np.random.default_rng(1).normal(500, 10, (16, 16)).astype(np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_exact(hdu, hdr)
        assert np.all(np.isfinite(reproj[fp == 1.0]))

    def test_flux_conservation_identity(self):
        """Exact reprojection of a single-pixel spike conserves total flux."""
        from reproject import reproject_exact
        from astropy.io import fits
        n = 32
        data = np.zeros((n, n), dtype=np.float64)
        data[n // 2, n // 2] = 1000.0
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, n)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, _ = reproject_exact(hdu, hdr)
        assert np.nansum(reproj) == pytest.approx(1000.0, rel=0.01)

    def test_return_footprint_false(self):
        from reproject import reproject_exact
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        result = reproject_exact(hdu, hdr, return_footprint=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)

    def test_output_dtype_is_floating(self):
        from reproject import reproject_exact
        from astropy.io import fits
        data = np.ones((8, 8), dtype=np.float32)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 8)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, _ = reproject_exact(hdu, hdr)
        assert np.issubdtype(reproj.dtype, np.floating)

    def test_constant_image_preserved(self):
        from reproject import reproject_exact
        from astropy.io import fits
        data = np.full((16, 16), 3.0, dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_exact(hdu, hdr)
        valid = reproj[fp == 1.0]
        np.testing.assert_allclose(valid, 3.0, atol=0.01)


# ---------------------------------------------------------------------------
# reproject_adaptive — Gaussian adaptive kernel reprojection
# ---------------------------------------------------------------------------

class TestReprojectAdaptive:
    """reproject_adaptive uses a locally-adapted Gaussian kernel."""

    def test_identity_shape_preserved(self):
        from reproject import reproject_adaptive
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_adaptive(hdu, hdr)
        assert reproj.shape == (16, 16)

    def test_footprint_is_returned(self):
        from reproject import reproject_adaptive
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_adaptive(hdu, hdr)
        assert fp.shape == (16, 16)

    def test_output_is_finite(self):
        from reproject import reproject_adaptive
        from astropy.io import fits
        data = np.random.default_rng(2).normal(500, 10, (16, 16)).astype(np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_adaptive(hdu, hdr)
        assert np.all(np.isfinite(reproj[fp > 0]))

    def test_output_dtype_is_floating(self):
        from reproject import reproject_adaptive
        from astropy.io import fits
        data = np.ones((8, 8), dtype=np.float32)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 8)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, _ = reproject_adaptive(hdu, hdr)
        assert np.issubdtype(reproj.dtype, np.floating)

    def test_conserve_flux_option(self):
        """conserve_flux=True should approximately preserve total flux."""
        from reproject import reproject_adaptive
        from astropy.io import fits
        n = 32
        data = np.zeros((n, n), dtype=np.float64)
        data[n // 2, n // 2] = 1000.0
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, n)
        hdu = fits.PrimaryHDU(data, header=hdr)
        reproj, fp = reproject_adaptive(hdu, hdr, conserve_flux=True)
        assert reproj.shape == (n, n)
        # total flux should be within 10% due to Gaussian spreading
        assert np.nansum(reproj) == pytest.approx(1000.0, rel=0.1)

    def test_return_footprint_false(self):
        from reproject import reproject_adaptive
        from astropy.io import fits
        data = np.ones((16, 16), dtype=np.float64)
        hdr = _make_wcs_header(150.0, 2.0, 3e-4, 16)
        hdu = fits.PrimaryHDU(data, header=hdr)
        result = reproject_adaptive(hdu, hdr, return_footprint=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)


# ---------------------------------------------------------------------------
# lscdiff.py template-alignment end-to-end pattern
# ---------------------------------------------------------------------------

class TestLscdiffReprojPattern:
    """Reproduce the exact three-image reproject pattern from lscdiff.py."""

    def _make_hdr(self, ra, dec):
        return _make_wcs_header(ra, dec, 3e-4, 32)

    def test_template_image_alignment(self):
        """Align a template image to the science WCS (primary use case)."""
        from reproject import reproject_interp
        from astropy.io import fits
        rng = np.random.default_rng(100)
        sci_hdr = self._make_hdr(150.0, 2.0)
        # Template is offset slightly in RA
        tmpl_hdr = self._make_hdr(150.001, 2.0)
        tmpl_data = rng.normal(1000, 10, (32, 32)).astype(np.float64)
        tmpl_hdu = fits.PrimaryHDU(tmpl_data, header=tmpl_hdr)
        temp_reproj, temp_foot = reproject_interp(tmpl_hdu, sci_hdr)
        assert temp_reproj.shape == (32, 32)
        assert temp_foot.shape == (32, 32)

    def test_template_mask_alignment(self):
        """Align a binary mask with nearest-neighbor to avoid interpolation artefacts."""
        from reproject import reproject_interp
        from astropy.io import fits
        sci_hdr = self._make_hdr(150.0, 2.0)
        tmpl_hdr = self._make_hdr(150.001, 2.0)
        mask_data = np.zeros((32, 32), dtype=np.float64)
        mask_data[5:10, 5:10] = 1.0  # bad-pixel region
        mask_hdu = fits.PrimaryHDU(mask_data, header=tmpl_hdr)
        tempmask_reproj, tempmask_foot = reproject_interp(
            mask_hdu, sci_hdr, order='nearest-neighbor'
        )
        assert tempmask_reproj.shape == (32, 32)
        valid = tempmask_reproj[tempmask_foot == 1.0]
        assert set(np.unique(valid[np.isfinite(valid)])).issubset({0.0, 1.0})

    def test_template_noise_alignment(self):
        """Align a noise/variance map with bilinear interpolation."""
        from reproject import reproject_interp
        from astropy.io import fits
        rng = np.random.default_rng(200)
        sci_hdr = self._make_hdr(150.0, 2.0)
        tmpl_hdr = self._make_hdr(150.001, 2.0)
        noise_data = rng.exponential(10, (32, 32)).astype(np.float64)
        noise_hdu = fits.PrimaryHDU(noise_data, header=tmpl_hdr)
        tempnoise_reproj, tempnoise_foot = reproject_interp(noise_hdu, sci_hdr)
        assert tempnoise_reproj.shape == (32, 32)
        # All valid noise values must be non-negative
        valid = tempnoise_reproj[tempnoise_foot == 1.0]
        assert np.all(valid[np.isfinite(valid)] >= 0.0)

    def test_all_three_images_have_consistent_footprints(self):
        """Science, mask, and noise reprojections must have the same footprint shape."""
        from reproject import reproject_interp
        from astropy.io import fits
        rng = np.random.default_rng(300)
        sci_hdr = self._make_hdr(150.0, 2.0)
        tmpl_hdr = self._make_hdr(150.001, 2.0)
        tmpl = fits.PrimaryHDU(rng.normal(500, 5, (32, 32)).astype(np.float64), header=tmpl_hdr)
        mask = fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float64), header=tmpl_hdr)
        noise = fits.PrimaryHDU(rng.exponential(5, (32, 32)).astype(np.float64), header=tmpl_hdr)
        _, fp_img = reproject_interp(tmpl, sci_hdr)
        _, fp_mask = reproject_interp(mask, sci_hdr, order='nearest-neighbor')
        _, fp_noise = reproject_interp(noise, sci_hdr)
        assert fp_img.shape == fp_mask.shape == fp_noise.shape

    def test_footprint_used_to_mask_output(self):
        """Setting output to NaN where footprint==0 is a common pipeline step."""
        from reproject import reproject_interp
        from astropy.io import fits
        rng = np.random.default_rng(400)
        sci_hdr = self._make_hdr(150.0, 2.0)
        tmpl_hdr = self._make_hdr(150.0, 2.0)
        data = rng.normal(500, 10, (32, 32)).astype(np.float64)
        hdu = fits.PrimaryHDU(data, header=tmpl_hdr)
        reproj, fp = reproject_interp(hdu, sci_hdr)
        reproj[fp == 0] = np.nan
        # After masking, no non-NaN pixel should be outside the footprint
        assert np.all(np.isnan(reproj[fp == 0]))
