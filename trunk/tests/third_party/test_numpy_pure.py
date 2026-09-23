"""
Tests for numpy operations used throughout lcogtsnpipe.

Covers the patterns actually used in the pipeline:
- array creation, dtype casting (lscabsphotdef, cosmics, lscastrodef)
- masked arrays (PyZOGY, util)
- statistics: nanpercentile, nanmean, median, std (lscabsphotdef, PyZOGY)
- linear algebra / FFT (cosmics, PyZOGY)
- random number generation (synthetic data in tests)
- structured array / record access (sep output, standard catalogs)

Coverage additions:
- np.compress with boolean condition       (lscabsphotdef, lscpsfdef, lscastrodef)
- np.where string-sentinel replacement     (lscpsfdef INDEF handling)
- np.tile for placeholder arrays           (lscabsphotdef fileph)
- np.isfinite / np.isnan guards            (lscabsphotdef, util)
- np.polyfit / np.polyval regression       (fitcol / fitcol2)
- np.average with weights + returned=True  (myloopdef weighted_avg, calcZC)
- np.argsort / np.sort                     (externaldata, lscastrodef)
- np.delete element removal                (lscpsfdef nearest-neighbour)
- np.deg2rad / trig for separation         (lscastrodef crossmatch)
- np.median absolute deviation (MAD)       (lscabsphotdef, util sky noise)
- np.testing.assert_allclose               (pipeline test infrastructure)
- np.unravel_index                         (peak finding)
- np.empty_like                            (externaldata calib image)
- np.abs element-wise                      (lscpsfdef, lscabsphotdef)
"""
import numpy as np
import numpy.ma as ma
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Array creation and dtype handling
# ---------------------------------------------------------------------------

class TestArrayCreation:
    def test_zeros_shape(self):
        a = np.zeros((3, 4))
        assert a.shape == (3, 4)
        assert np.all(a == 0)

    def test_full_float32(self):
        a = np.full((5, 5), 1000.0, dtype=np.float32)
        assert a.dtype == np.float32
        assert a[2, 2] == pytest.approx(1000.0)

    def test_astype_float64(self):
        a = np.array([1, 2, 3], dtype=np.int16)
        b = a.astype(np.float64)
        assert b.dtype == np.float64

    def test_arange_length(self):
        a = np.arange(0, 10, 2)
        assert len(a) == 5
        np.testing.assert_array_equal(a, [0, 2, 4, 6, 8])

    def test_linspace_endpoints(self):
        a = np.linspace(0, 1, 11)
        assert a[0] == pytest.approx(0.0)
        assert a[-1] == pytest.approx(1.0)
        assert len(a) == 11

    def test_eye_identity(self):
        I = np.eye(3)
        assert I[0, 0] == 1.0
        assert I[0, 1] == 0.0

    def test_stack_axis0(self):
        a = np.ones((2, 3))
        b = np.ones((2, 3)) * 2
        s = np.stack([a, b], axis=0)
        assert s.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# Statistics — pipeline uses these extensively for sigma-clipping/zeropoints
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_nanmean_ignores_nan(self):
        a = np.array([1.0, 2.0, np.nan, 4.0])
        assert np.nanmean(a) == pytest.approx(7.0 / 3.0)

    def test_nanmedian(self):
        a = np.array([1.0, 3.0, np.nan, 5.0])
        assert np.nanmedian(a) == pytest.approx(3.0)

    def test_nanstd(self):
        a = np.array([0.0, 1.0, 2.0, np.nan])
        assert np.nanstd(a) == pytest.approx(np.std([0.0, 1.0, 2.0]))

    def test_nanpercentile_quartiles(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
        q25, q75 = np.nanpercentile(a, [25, 75])
        assert q25 < q75

    def test_histogram_bins(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        counts, edges = np.histogram(data, bins=20)
        assert counts.sum() == 1000
        assert len(edges) == 21

    def test_argmax_2d(self):
        a = np.zeros((4, 4))
        a[2, 3] = 1.0
        idx = np.unravel_index(a.argmax(), a.shape)
        assert idx == (2, 3)

    def test_clip_values(self):
        a = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
        c = np.clip(a, 0.0, 10.0)
        assert c[0] == 0.0
        assert c[-1] == 10.0


# ---------------------------------------------------------------------------
# Masked arrays — used in PyZOGY and cosmic-ray rejection
# ---------------------------------------------------------------------------

class TestMaskedArrays:
    def test_create_masked_array(self):
        data = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, True, False])
        m = ma.array(data, mask=mask)
        assert m[1] is ma.masked

    def test_compressed_excludes_masked(self):
        data = np.arange(5, dtype=float)
        mask = np.array([False, True, False, False, True])
        m = ma.array(data, mask=mask)
        assert list(m.compressed()) == [0.0, 2.0, 3.0]

    def test_masked_mean(self):
        data = np.array([1.0, 100.0, 3.0])
        mask = np.array([False, True, False])
        m = ma.array(data, mask=mask)
        assert float(m.mean()) == pytest.approx(2.0)

    def test_mask_propagation_on_operations(self):
        # Use explicit data + mask to avoid the UserWarning that arises when
        # numpy converts the ma.masked sentinel to NaN in a list literal.
        data = np.array([1.0, 0.0, 3.0])
        mask = np.array([False, True, False])
        a = ma.array(data, mask=mask)
        b = a * 2
        assert b[1] is ma.masked
        assert float(b[0]) == 2.0

    def test_fill_value(self):
        data = np.array([1.0, 0.0, 3.0])
        mask = np.array([False, True, False])
        a = ma.array(data, mask=mask, fill_value=-999.0)
        filled = a.filled()
        assert filled[1] == -999.0


# ---------------------------------------------------------------------------
# Linear algebra and FFT — used in cosmics.py and PyZOGY
# ---------------------------------------------------------------------------

class TestLinalgAndFFT:
    def test_dot_product(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert np.dot(a, b) == pytest.approx(32.0)

    def test_matmul(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.eye(2)
        np.testing.assert_array_almost_equal(A @ B, A)

    def test_fft_round_trip(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 64)
        freq = np.fft.fft(data)
        back = np.fft.ifft(freq).real
        np.testing.assert_allclose(back, data, atol=1e-10)

    def test_fft2_shape(self):
        data = np.ones((8, 8))
        freq = np.fft.fft2(data)
        assert freq.shape == (8, 8)

    def test_fftshift_reverses_with_ifftshift(self):
        data = np.arange(8, dtype=float)
        shifted = np.fft.fftshift(data)
        back = np.fft.ifftshift(shifted)
        np.testing.assert_array_equal(back, data)


# ---------------------------------------------------------------------------
# Structured / record array — mirrors sep extraction output used in PyZOGY
# ---------------------------------------------------------------------------

class TestStructuredArrays:
    def test_create_and_access_fields(self):
        dt = np.dtype([("x", np.float64), ("y", np.float64), ("flux", np.float64)])
        arr = np.array([(1.0, 2.0, 100.0), (3.0, 4.0, 200.0)], dtype=dt)
        np.testing.assert_array_equal(arr["x"], [1.0, 3.0])
        assert arr["flux"][1] == 200.0

    def test_boolean_indexing(self):
        dt = np.dtype([("mag", np.float64)])
        arr = np.array([(15.0,), (18.0,), (22.0,)], dtype=dt)
        bright = arr[arr["mag"] < 17.0]
        assert len(bright) == 1
        assert bright["mag"][0] == 15.0


# ---------------------------------------------------------------------------
# Coordinate / angle arithmetic — mirrors lscastrodef crossmatch
# ---------------------------------------------------------------------------

class TestCoordinateArithmetic:
    def test_angular_separation_small(self):
        # haversine-style check: small sep in arcsec
        ra1, dec1 = np.radians(150.0), np.radians(2.0)
        ra2, dec2 = np.radians(150.0 + 1.0 / 3600.0), np.radians(2.0)
        dra = ra2 - ra1
        sep = np.sqrt((np.cos(dec1) * dra) ** 2) * (180 / np.pi) * 3600
        assert sep == pytest.approx(1.0, rel=0.01)

    def test_roll_wraps_correctly(self):
        a = np.array([1, 2, 3, 4, 5])
        rolled = np.roll(a, -2)
        np.testing.assert_array_equal(rolled, [3, 4, 5, 1, 2])


# ---------------------------------------------------------------------------
# np.compress — used heavily in lscabsphotdef, lscpsfdef, lscastrodef
# ---------------------------------------------------------------------------

class TestCompress:
    """np.compress(condition, a) — 1-D boolean mask selection."""

    def test_compress_magnitude_cut(self):
        """lscabsphotdef: magsex = np.compress(np.array(magsex) <= cutmag, magsex)"""
        magsex = np.array([14.5, 18.0, 19.5, 22.0, 25.0])
        cutmag = 20.0
        result = np.compress(magsex <= cutmag, magsex)
        assert list(result) == pytest.approx([14.5, 18.0, 19.5])

    def test_compress_sigma_clip(self):
        """lscabsphotdef/lscpsfdef: sigma-clip with compress + abs."""
        data = np.array([23.0, 23.1, 23.05, 30.0, 22.9])
        med = np.median(data)
        std = np.std(data)
        clipped = np.compress(np.abs(data - med) < 2 * std, data)
        assert 30.0 not in clipped
        assert len(clipped) == 4

    def test_compress_two_arrays_same_mask(self):
        """lscabsphotdef: rasex = np.compress(mask, rasex); decsex = np.compress(mask, decsex)"""
        mags = np.array([15.0, 19.0, 21.0, 23.0])
        ras = np.array([150.0, 150.1, 150.2, 150.3])
        decs = np.array([2.0, 2.1, 2.2, 2.3])
        mask = mags < 22.0
        ras_cut = np.compress(mask, ras)
        decs_cut = np.compress(mask, decs)
        assert len(ras_cut) == 3
        assert len(decs_cut) == 3

    def test_compress_abs_condition(self):
        """lscabsphotdef: good = abs(zero) < 50"""
        zero = np.array([-100.0, 1.0, -0.5, 60.0, 0.2])
        good = np.abs(zero) < 50
        result = np.compress(good, zero)
        np.testing.assert_array_equal(result, [1.0, -0.5, 0.2])


# ---------------------------------------------------------------------------
# np.where — used in lscpsfdef for INDEF → 9999 substitution
# ---------------------------------------------------------------------------

class TestWhere:
    """np.where(condition, x, y) — element-wise conditional selection."""

    def test_where_indef_replacement(self):
        """lscpsfdef: np.where(arr != 'INDEF', arr, 9999)"""
        raw = np.array(["15.34", "INDEF", "18.22", "INDEF"], dtype=object)
        replaced = np.where(raw != "INDEF", raw, "9999").astype(float)
        assert replaced[1] == 9999.0
        assert replaced[3] == 9999.0
        assert replaced[0] == pytest.approx(15.34)

    def test_where_zero_mask(self):
        """externaldata: ar = np.where(ar2 == 0, _saturate, ar)"""
        ar2 = np.array([1.0, 0.0, 3.0, 0.0])
        ar = np.array([10.0, 20.0, 30.0, 40.0])
        saturate = 65535.0
        result = np.where(ar2 == 0, saturate, ar)
        assert result[1] == saturate
        assert result[3] == saturate
        assert result[0] == 10.0

    def test_where_boolean_three_outcomes(self):
        values = np.array([-2, -1, 0, 1, 2])
        result = np.where(values >= 0, values, 0)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 2])


# ---------------------------------------------------------------------------
# np.tile — used in lscabsphotdef to initialise placeholder fileph arrays
# ---------------------------------------------------------------------------

class TestTile:
    """np.tile(A, reps) — repeat array to fill placeholder column."""

    def test_tile_scalar_to_array(self):
        """lscabsphotdef: fileph['mU'] = np.tile(999, len(rastd0))"""
        n = 10
        result = np.tile(999, n)
        assert result.shape == (n,)
        assert np.all(result == 999)

    def test_tile_along_axis(self):
        a = np.array([1, 2, 3])
        tiled = np.tile(a, 3)
        np.testing.assert_array_equal(tiled, [1, 2, 3, 1, 2, 3, 1, 2, 3])

    def test_tile_2d(self):
        a = np.array([[1, 2]])
        tiled = np.tile(a, (3, 1))
        assert tiled.shape == (3, 2)


# ---------------------------------------------------------------------------
# np.isfinite / np.isnan — guards in lscabsphotdef and util
# ---------------------------------------------------------------------------

class TestFiniteChecks:
    """np.isfinite and np.isnan used as result guards before DB writes."""

    def test_isfinite_rejects_inf(self):
        """lscabsphotdef: if not np.isfinite(result[ll][kk]): result = 0.0"""
        values = np.array([1.0, np.inf, -np.inf, np.nan, 23.5])
        finite_mask = np.isfinite(values)
        assert list(finite_mask) == [True, False, False, False, True]

    def test_isnan_guard(self):
        """util / lscabsphotdef: if np.isnan(z2): z2, std2 = 9999, 9999"""
        z2 = float("nan")
        if np.isnan(z2):
            z2 = 9999
        assert z2 == 9999

    def test_replace_nonfinite_with_zero(self):
        """Mirrors the DB-safe replacement loop in lscabsphotdef."""
        results = [1.5, np.inf, np.nan, -np.inf, 0.05]
        clean = [0.0 if not np.isfinite(v) else v for v in results]
        assert clean[1] == 0.0
        assert clean[2] == 0.0
        assert clean[3] == 0.0
        assert clean[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# np.polyfit / np.polyval — used in fitcol, fitcol2, onkeypress, onclick
# ---------------------------------------------------------------------------

class TestPolyfitPolyval:
    """Linear regression via np.polyfit; evaluation via np.polyval."""

    def test_polyfit_degree1_perfect_line(self):
        """fitcol: pol = np.polyfit(_col, _dmag, 1, full=True)"""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 0.5 * x + 23.0
        coeffs = np.polyfit(x, y, 1)
        assert coeffs[0] == pytest.approx(0.5, abs=1e-10)
        assert coeffs[1] == pytest.approx(23.0, abs=1e-10)

    def test_polyfit_full_returns_residuals(self):
        """fitcol: pol = np.polyfit(…, full=True); sigmae = sqrt(pol[1][0] / (N-2))"""
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 20)
        y = 0.1 * x + 23.5 + rng.normal(0, 0.01, 20)
        pol = np.polyfit(x, y, 1, full=True)
        coeffs = pol[0]
        residuals = pol[1]
        assert len(residuals) > 0
        assert residuals[0] > 0

    def test_polyval_evaluates_line(self):
        """fitcol: yy = np.polyval([bb, aa], xx)"""
        coeffs = [0.05, 23.5]   # slope, intercept
        xx = np.array([-0.5, 0.0, 1.0, 1.5])
        yy = np.polyval(coeffs, xx)
        expected = 0.05 * xx + 23.5
        np.testing.assert_allclose(yy, expected, atol=1e-12)

    def test_polyval_regression_line_endpoints(self):
        """Mirrors xx = [min(_col)-.1, max(_col)+.1]; yy = polyval([bb,aa], xx)"""
        col = np.array([0.1, 0.5, 0.8, 1.2])
        dmag = np.array([23.4, 23.45, 23.48, 23.52])
        bb, aa = np.polyfit(col, dmag, 1)
        xx = np.array([col.min() - 0.1, col.max() + 0.1])
        yy = np.polyval([bb, aa], xx)
        # slope exists so endpoints are not equal
        assert yy[0] != pytest.approx(yy[1])


# ---------------------------------------------------------------------------
# np.average with weights — mirrors myloopdef.weighted_avg and calcZC
# ---------------------------------------------------------------------------

class TestWeightedAverage:
    """np.average(values, weights=w) and returned=True variant."""

    def test_weighted_average_simple(self):
        """myloopdef.weighted_avg: average = np.average(values, weights=weights)"""
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 2.0, 1.0])
        avg = np.average(values, weights=weights)
        assert avg == pytest.approx(20.0)

    def test_weighted_average_returned_sum_of_weights(self):
        """calcZC: Z, sum_of_weights = np.average(…, weights=…, returned=True)"""
        values = np.array([23.4, 23.5, 23.6])
        weights = np.array([100.0, 25.0, 100.0])
        avg, sum_w = np.average(values, weights=weights, returned=True)
        assert sum_w == pytest.approx(225.0)
        assert avg == pytest.approx(np.sum(values * weights) / sum_w)

    def test_weighted_std_from_average(self):
        """myloopdef: variance = np.average((values-average)**2, weights=weights)"""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        avg = np.average(values, weights=weights)
        variance = np.average((values - avg) ** 2, weights=weights)
        assert np.sqrt(variance) == pytest.approx(np.std(values))


# ---------------------------------------------------------------------------
# np.argsort / np.sort — used in externaldata and lscastrodef
# ---------------------------------------------------------------------------

class TestSortAndArgsort:
    """Sorting utilities used to order catalog rows and match lists."""

    def test_argsort_ascending(self):
        """externaldata: table = table[np.argsort(flist)]"""
        flist = np.array([3.0, 1.0, 4.0, 1.5, 2.0])
        idx = np.argsort(flist)
        sorted_vals = flist[idx]
        assert list(sorted_vals) == sorted(flist.tolist())

    def test_sort_2d_by_column(self):
        data = np.array([[3, 10], [1, 20], [2, 15]])
        order = np.argsort(data[:, 0])
        sorted_data = data[order]
        np.testing.assert_array_equal(sorted_data[:, 0], [1, 2, 3])

    def test_argsort_with_lscastrodef_distance_argmin(self):
        """lscastrodef crossmatch: np.argmin(distance)"""
        ref_x = np.array([0.0, 5.0, 10.0])
        query_x = 4.8
        distances = np.sqrt((ref_x - query_x) ** 2)
        nearest = np.argmin(distances)
        assert nearest == 1  # index of 5.0


# ---------------------------------------------------------------------------
# np.delete — lscpsfdef removes one star at a time for nearest-neighbour
# ---------------------------------------------------------------------------

class TestDelete:
    """np.delete(arr, idx) used in lscpsfdef to exclude self from distance calc."""

    def test_delete_single_index(self):
        """lscpsfdef: _xs = np.delete(xs, i)"""
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _xs = np.delete(xs, 2)
        np.testing.assert_array_equal(_xs, [1.0, 2.0, 4.0, 5.0])

    def test_delete_then_distance(self):
        """lscpsfdef: dist2 = np.sqrt((_xs - xs[i])**2 + (_ys - ys[i])**2)"""
        xs = np.array([0.0, 3.0, 6.0, 9.0])
        ys = np.array([0.0, 4.0, 0.0, 4.0])
        i = 1
        _xs = np.delete(xs, i)
        _ys = np.delete(ys, i)
        dist2 = np.sqrt((_xs - xs[i]) ** 2 + (_ys - ys[i]) ** 2)
        assert dist2[0] == pytest.approx(5.0)  # (3,4) → (0,0): dist=5

    def test_delete_multiple_indices(self):
        a = np.arange(10)
        result = np.delete(a, [2, 5, 8])
        assert 2 not in result
        assert 5 not in result
        assert 8 not in result
        assert len(result) == 7


# ---------------------------------------------------------------------------
# Trigonometry (deg2rad, cos, sin, arccos, pi) — lscastrodef, externaldata
# ---------------------------------------------------------------------------

class TestTrigonometry:
    """np.cos/sin/arccos/deg2rad used in coordinate conversions."""

    def test_deg2rad_roundtrip(self):
        """lscastrodef: from numpy import sin, cos, deg2rad"""
        angles_deg = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
        radians = np.deg2rad(angles_deg)
        back = np.rad2deg(radians)
        np.testing.assert_allclose(back, angles_deg, atol=1e-10)

    def test_cos_dec_correction(self):
        """externaldata: DR = delta / np.cos(_dec * np.pi / 180)"""
        _dec = 30.0  # degrees
        delta = 1.0  # arcsec
        DR = delta / np.cos(_dec * np.pi / 180.0)
        assert DR == pytest.approx(1.0 / np.cos(np.radians(30.0)))

    def test_crossmatch_distance_formula(self):
        """lscastrodef.crossmatch: distance = np.sqrt(dx**2 + dy**2)"""
        x0 = np.array([0.0, 3.0])
        y0 = np.array([0.0, 4.0])
        x1, y1 = 3.0, 4.0
        dist = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        assert dist[0] == pytest.approx(5.0)
        assert dist[1] == pytest.approx(0.0)

    def test_arccos_within_domain(self):
        """util.py: from numpy import arccos — used for great-circle sep."""
        val = np.clip(0.9999, -1.0, 1.0)
        sep_rad = np.arccos(val)
        assert 0.0 <= sep_rad <= np.pi


# ---------------------------------------------------------------------------
# Median Absolute Deviation (MAD) — util.py and lscabsphotdef sky noise
# ---------------------------------------------------------------------------

class TestMAD:
    """1.4826 * median(|x - median(x)|) — robust sigma estimator in pipeline."""

    def test_mad_gaussian_approx_sigma(self):
        """util.py: noise = 1.4826 * np.median(np.abs(ar - med))"""
        rng = np.random.default_rng(7)
        ar = rng.normal(0, 1, 10000)
        med = np.median(ar)
        mad = 1.4826 * np.median(np.abs(ar - med))
        assert mad == pytest.approx(1.0, abs=0.05)

    def test_mad_robust_to_outliers(self):
        """MAD ignores extreme outliers; regular std does not."""
        clean = np.ones(99) * 5.0
        data = np.concatenate([clean, [1000.0]])
        mad = 1.4826 * np.median(np.abs(data - np.median(data)))
        sigma = np.std(data)
        assert mad < 1.0          # MAD ~ 0 for near-constant clean data
        assert sigma > 99.0       # std blown up by outlier

    def test_variance_rescaling(self):
        """externaldata: variance *= (MAD * 1.48)**2 / np.median(variance)"""
        rng = np.random.default_rng(8)
        ar = rng.normal(100.0, 5.0, 200)
        variance = np.ones_like(ar) * 10.0
        scale = (np.median(np.abs(ar - np.median(ar))) * 1.48) ** 2 / np.median(variance)
        variance_scaled = variance * scale
        assert np.all(np.isfinite(variance_scaled))
        assert np.median(variance_scaled) == pytest.approx(scale * 10.0)


# ---------------------------------------------------------------------------
# np.empty_like / np.sqrt element-wise — externaldata calibration image
# ---------------------------------------------------------------------------

class TestArrayAllocation:
    """np.empty_like and element-wise np.sqrt used in externaldata."""

    def test_empty_like_preserves_shape_and_dtype(self):
        """externaldata: calib_image = np.empty_like(frame_image)"""
        frame = np.ones((128, 128), dtype=np.float32) * 1000.0
        calib = np.empty_like(frame)
        assert calib.shape == frame.shape
        assert calib.dtype == frame.dtype

    def test_sqrt_noise_model(self):
        """externaldata: dn_err = np.sqrt((dn + sky) / gain + dark_var)"""
        dn_image = np.ones((10, 10)) * 500.0
        sky_image = np.ones((10, 10)) * 50.0
        gain = 2.0
        dark_var = 5.0
        dn_err = np.sqrt((dn_image + sky_image) / gain + dark_var)
        expected = np.sqrt(275.0 + 5.0)
        np.testing.assert_allclose(dn_err, expected, rtol=1e-6)

    def test_sqrt_quadrature_error(self):
        """lscpsfdef: np.sqrt(smagerrf**2 + aperture_correction_err**2)"""
        smagerrf = 0.03
        aperture_correction_err = 0.04
        combined = np.sqrt(smagerrf ** 2 + aperture_correction_err ** 2)
        assert combined == pytest.approx(0.05, abs=1e-10)


# ---------------------------------------------------------------------------
# np.testing utilities — used in test infrastructure itself
# ---------------------------------------------------------------------------

class TestNumpyTesting:
    """np.testing functions used throughout test suite validation."""

    def test_assert_allclose(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0 + 1e-12, 2.0, 3.0 - 1e-12])
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_assert_array_equal(self):
        a = np.arange(5)
        b = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(a, b)

    def test_assert_array_almost_equal(self):
        a = np.array([1.0, 2.0])
        b = a + 1e-7
        np.testing.assert_array_almost_equal(a, b, decimal=5)

    def test_assert_raises_on_shape_mismatch(self):
        a = np.zeros(3)
        b = np.zeros(4)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(a, b)
