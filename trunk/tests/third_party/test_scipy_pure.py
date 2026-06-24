"""
Tests for scipy functions used in lcogtsnpipe.

Covers the modules actually imported by the pipeline:
- scipy.optimize.fsolve  (lscabsphotdef.py — limiting magnitude SNR inversion)
- scipy.signal           (cosmics.py — median filter, convolve2d with boundary='symm')
- scipy.ndimage          (cosmics.py — label, median_filter, binary_dilation, find_objects, sum)
- scipy.stats.sigmaclip  (banzaicat.py — outlier rejection)
- scipy.stats.theilslopes (lscabsphotdef.py — robust color-term slope estimation)
- scipy.odr              (lscabsphotdef.py — orthogonal distance regression for color terms)
- scipy.interpolate      (externaldata.py — interp2d sky background model)

Additional coverage added:
  - TestFsolveSNRPattern: snr_equation / limmag pattern from lscabsphotdef.py
  - TestScipyStatsTheilslopes: robust Theil-Sen slope estimation for color calibration
  - TestScipyOdr: ODR Model/Data/ODR.run() for color-term fitting with uncertainties
  - TestScipyInterpolate: interp1d and RectBivariateSpline (interp2d replacement) for sky model
  - TestNdimageMedianFilter: median_filter size=5 mode='mirror' as used in cosmics.py
  - TestNdimageBinaryDilation: binary_dilation for cosmic-ray mask growing in cosmics.py
  - TestNdimageFindObjects: find_objects for bounding-box slices in cosmics.py
  - TestNdimageSum: ndimage.sum for blob-size measurements in cosmics.py
  - TestSignalConvolve2dSymm: boundary='symm' convolution as used in cosmics.py
"""
import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# scipy.optimize.fsolve — used in lscabsphotdef.limmag to find limiting mag
# ---------------------------------------------------------------------------

class TestFsolve:
    def test_finds_root_of_linear(self):
        from scipy.optimize import fsolve
        # f(x) = 2x - 6 → root at x=3
        root = fsolve(lambda x: 2 * x - 6, x0=0.0)
        assert root[0] == pytest.approx(3.0)

    def test_finds_root_of_quadratic(self):
        from scipy.optimize import fsolve
        # x^2 - 4 = 0, starting near x=2 → root at +2
        root = fsolve(lambda x: x ** 2 - 4.0, x0=1.5)
        assert abs(root[0]) == pytest.approx(2.0, rel=1e-5)

    def test_root_of_exponential(self):
        from scipy.optimize import fsolve
        import math
        # exp(x) - 2 = 0 → x = ln(2)
        root = fsolve(lambda x: np.exp(x) - 2.0, x0=0.5)
        assert root[0] == pytest.approx(math.log(2), rel=1e-5)

    def test_returns_array(self):
        from scipy.optimize import fsolve
        result = fsolve(lambda x: x - 7.0, x0=0.0)
        assert isinstance(result, np.ndarray)

    def test_vector_system(self):
        from scipy.optimize import fsolve
        # system: x+y=3, x-y=1 → x=2, y=1
        def system(v):
            x, y = v
            return [x + y - 3, x - y - 1]
        root = fsolve(system, [0.0, 0.0])
        assert root[0] == pytest.approx(2.0)
        assert root[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# scipy.signal — used in cosmics.py for median filtering and convolution
# ---------------------------------------------------------------------------

class TestScipySignal:
    def test_medfilt2d_uniform(self):
        from scipy.signal import medfilt2d
        data = np.ones((9, 9)) * 5.0
        result = medfilt2d(data, kernel_size=3)
        assert result[4, 4] == pytest.approx(5.0)

    def test_medfilt2d_spike_removed(self):
        from scipy.signal import medfilt2d
        data = np.ones((9, 9)) * 10.0
        data[4, 4] = 9999.0  # single spike
        result = medfilt2d(data, kernel_size=3)
        assert result[4, 4] == pytest.approx(10.0)

    def test_convolve2d_identity_kernel(self):
        from scipy.signal import convolve2d
        data = np.random.default_rng(5).normal(0, 1, (8, 8))
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        result = convolve2d(data, kernel, mode='same')
        np.testing.assert_allclose(result, data, atol=1e-10)

    def test_convolve2d_smoothing_kernel(self):
        from scipy.signal import convolve2d
        data = np.zeros((9, 9))
        data[4, 4] = 1.0
        kernel = np.ones((3, 3)) / 9.0
        result = convolve2d(data, kernel, mode='same')
        # energy conserved
        assert np.sum(result) == pytest.approx(1.0, abs=1e-10)

    def test_output_shape_same_mode(self):
        from scipy.signal import convolve2d
        data = np.ones((10, 10))
        kernel = np.ones((3, 3))
        result = convolve2d(data, kernel, mode='same')
        assert result.shape == data.shape


# ---------------------------------------------------------------------------
# scipy.ndimage — used in cosmics.py (label, uniform_filter) and PyZOGY.util (zoom)
# ---------------------------------------------------------------------------

class TestScipyNdimage:
    def test_label_finds_connected_components(self):
        from scipy.ndimage import label
        # Default 4-connectivity: top-left cluster (3 cells) and bottom-right
        # cluster (3 cells) are NOT connected via the diagonal — 2 blobs.
        data = np.array([[1, 1, 0, 0],
                         [1, 0, 0, 1],
                         [0, 0, 1, 1]])
        labeled, n = label(data)
        assert n == 2  # two blobs under default 4-connectivity

    def test_label_single_blob(self):
        from scipy.ndimage import label
        data = np.ones((4, 4), dtype=int)
        _, n = label(data)
        assert n == 1

    def test_uniform_filter_uniform_input(self):
        from scipy.ndimage import uniform_filter
        data = np.ones((8, 8)) * 7.0
        result = uniform_filter(data, size=3)
        assert result[4, 4] == pytest.approx(7.0)

    def test_zoom_doubles_size(self):
        from scipy.ndimage import zoom
        data = np.ones((4, 4))
        result = zoom(data, 2)
        assert result.shape == (8, 8)

    def test_zoom_halves_size(self):
        from scipy.ndimage import zoom
        data = np.ones((8, 8))
        result = zoom(data, 0.5)
        assert result.shape == (4, 4)

    def test_gaussian_filter_smooths(self):
        from scipy.ndimage import gaussian_filter
        data = np.zeros((11, 11))
        data[5, 5] = 100.0
        result = gaussian_filter(data, sigma=1.5)
        # peak spreads — original pixel value should decrease
        assert result[5, 5] < 100.0
        # energy approximately conserved
        assert np.sum(result) == pytest.approx(100.0, rel=0.01)


# ---------------------------------------------------------------------------
# scipy.stats.sigmaclip — used in banzaicat.py for star catalog rejection
# ---------------------------------------------------------------------------

class TestSigmaclip:
    def test_clips_outliers(self):
        from scipy.stats import sigmaclip
        # sigmaclip is iterative; with only 6 points the first iteration may
        # not remove an outlier if the std is dominated by it.  Use a larger
        # clean sample so the std is small enough for 1000.0 to be rejected.
        rng = np.random.default_rng(99)
        data = np.concatenate([rng.normal(5.0, 0.5, 50), [1000.0]])
        clipped, lo, hi = sigmaclip(data, low=3.0, high=3.0)
        assert 1000.0 not in clipped

    def test_no_clipping_on_clean_data(self):
        from scipy.stats import sigmaclip
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 100)
        clipped, _, _ = sigmaclip(data, low=5.0, high=5.0)
        # with 5-sigma clip on normal data almost nothing should be removed
        assert len(clipped) > 90

    def test_returns_three_values(self):
        from scipy.stats import sigmaclip
        result = sigmaclip([1.0, 2.0, 3.0])
        assert len(result) == 3

    def test_lower_bound_correct(self):
        from scipy.stats import sigmaclip
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        _, lo, hi = sigmaclip(data)
        assert lo < hi

    def test_symmetric_clip(self):
        from scipy.stats import sigmaclip
        data = np.concatenate([np.ones(50) * 5.0, [-1000.0, 1000.0]])
        clipped, _, _ = sigmaclip(data, low=3.0, high=3.0)
        assert -1000.0 not in clipped
        assert 1000.0 not in clipped


# ---------------------------------------------------------------------------
# fsolve — SNR inversion pattern from lscabsphotdef.py limmag()
# ---------------------------------------------------------------------------

class TestFsolveSNRPattern:
    """Mirror the snr_equation / fsolve pattern used in lscabsphotdef.limmag()."""

    @staticmethod
    def _snr_equation(counts, Nsigma_limit, rdnoise, gain, skynoise, radius):
        """Replica of lscabsphotdef.snr_equation."""
        area = np.pi * radius ** 2
        snr = (counts * gain
               - Nsigma_limit
               * (counts * gain + (skynoise * gain) ** 2 * area + rdnoise ** 2 * area) ** 0.5)
        return snr

    def test_snr_equation_positive_for_bright_source(self):
        """A very bright source should give positive SNR above the limit."""
        snr = self._snr_equation(1e6, 5, 10, 2.0, 20, 5)
        assert snr > 0

    def test_snr_equation_negative_for_faint_source(self):
        """A very faint source should fail the SNR threshold."""
        snr = self._snr_equation(1.0, 5, 10, 2.0, 20, 5)
        assert snr < 0

    def test_fsolve_finds_limit_counts(self):
        """fsolve should find the count level where SNR = Nsigma_limit."""
        from scipy.optimize import fsolve
        rdnoise, gain, skynoise, radius, Nsigma = 10.0, 2.0, 20.0, 5.0, 3.0

        def helper(counts, extra):
            return self._snr_equation(counts, *extra)

        limit_counts = fsolve(helper, 100.0, args=([Nsigma, rdnoise, gain, skynoise, radius],))[0]
        # Verify: SNR at the limit should be ~0
        snr_at_limit = self._snr_equation(limit_counts, Nsigma, rdnoise, gain, skynoise, radius)
        assert snr_at_limit == pytest.approx(0.0, abs=1e-6)

    def test_limit_counts_positive(self):
        """Limiting counts must be physically positive."""
        from scipy.optimize import fsolve

        def helper(counts, extra):
            return self._snr_equation(counts, *extra)

        limit_counts = fsolve(helper, 500.0, args=([5.0, 10.0, 2.0, 20.0, 5.0],))[0]
        assert limit_counts > 0

    def test_higher_sigma_gives_higher_limit(self):
        """A 5-sigma limit requires more counts than a 3-sigma limit."""
        from scipy.optimize import fsolve

        def helper(counts, extra):
            return self._snr_equation(counts, *extra)

        params = [10.0, 2.0, 20.0, 5.0]  # rdnoise, gain, skynoise, radius
        lim3 = fsolve(helper, 100.0, args=([3.0] + params,))[0]
        lim5 = fsolve(helper, 100.0, args=([5.0] + params,))[0]
        assert lim5 > lim3


# ---------------------------------------------------------------------------
# scipy.stats.theilslopes — robust slope estimation (lscabsphotdef.py fitcol3)
# ---------------------------------------------------------------------------

class TestScipyStatsTheilslopes:
    """theilslopes gives a robust (Theil-Sen) linear fit for color calibration."""

    def test_perfect_line_recovered(self):
        from scipy.stats import theilslopes
        x = np.linspace(0, 1, 20)
        y = 2.5 * x + 0.3
        result = theilslopes(y, x)
        assert result.slope == pytest.approx(2.5, rel=1e-4)
        assert result.intercept == pytest.approx(0.3, abs=1e-4)

    def test_robust_to_outlier(self):
        """Unlike OLS, Theil-Sen slope is robust to a single extreme outlier."""
        from scipy.stats import theilslopes
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 50)
        y = 1.0 * x + 0.0 + rng.normal(0, 0.05, 50)
        y[0] = 999.0  # severe outlier
        result = theilslopes(y, x)
        assert result.slope == pytest.approx(1.0, abs=0.15)

    def test_returns_named_result(self):
        """theilslopes returns a result with slope, intercept, low_slope, high_slope."""
        from scipy.stats import theilslopes
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.1, 1.0, 2.1, 2.9])
        result = theilslopes(y, x)
        assert hasattr(result, "slope")
        assert hasattr(result, "intercept")

    def test_slope_sign_preserved(self):
        from scipy.stats import theilslopes
        x = np.linspace(0, 1, 10)
        y = -3.0 * x + 5.0
        result = theilslopes(y, x)
        assert result.slope < 0

    def test_zero_slope_flat_data(self):
        from scipy.stats import theilslopes
        x = np.linspace(0, 1, 20)
        y = np.ones(20) * 4.5
        result = theilslopes(y, x)
        assert result.slope == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# scipy.odr — orthogonal distance regression (lscabsphotdef.py fitcol3)
# ---------------------------------------------------------------------------

class TestScipyOdr:
    """ODR fits a linear model accounting for uncertainties in both x and y."""

    def _run_linear_odr(self, x, y, sx, sy, beta0=(1.0, 0.0)):
        from scipy import odr
        linear = odr.Model(lambda B, x: B[0] * x + B[1])
        data = odr.Data(x, y, wd=sx ** -2, we=sy ** -2)
        myodr = odr.ODR(data, linear, beta0=list(beta0))
        return myodr.run()

    def test_perfect_line(self):
        x = np.linspace(-1, 1, 20)
        y = 0.5 * x + 2.0
        sx = np.ones(20) * 0.01
        sy = np.ones(20) * 0.01
        out = self._run_linear_odr(x, y, sx, sy, beta0=(0.4, 1.9))
        assert out.beta[0] == pytest.approx(0.5, abs=0.01)
        assert out.beta[1] == pytest.approx(2.0, abs=0.01)

    def test_odr_output_has_beta(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.1, 3.0, 4.05])
        sx = np.ones(4) * 0.1
        sy = np.ones(4) * 0.1
        out = self._run_linear_odr(x, y, sx, sy)
        assert hasattr(out, "beta")
        assert len(out.beta) == 2

    def test_odr_output_has_sd_beta(self):
        x = np.linspace(0, 1, 10)
        y = 2.0 * x + 1.0 + np.random.default_rng(7).normal(0, 0.02, 10)
        sx = np.ones(10) * 0.05
        sy = np.ones(10) * 0.05
        out = self._run_linear_odr(x, y, sx, sy)
        assert hasattr(out, "sd_beta")
        assert len(out.sd_beta) == 2

    def test_model_callable(self):
        from scipy import odr
        linear = odr.Model(lambda B, x: B[0] * x + B[1])
        assert callable(linear.fcn)

    def test_data_with_weights(self):
        from scipy import odr
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        data = odr.Data(x, y, wd=np.array([1.0, 1.0, 1.0]), we=np.array([1.0, 1.0, 1.0]))
        assert data.x is not None
        assert data.y is not None

    def test_slope_matches_linear_input(self):
        """ODR of noise-free linear data should recover the true slope."""
        x = np.linspace(0.5, 2.0, 20)
        y = 0.109 * x + 0.3  # ggr color term from the pipeline
        sx = np.ones(20) * 0.01
        sy = np.ones(20) * 0.01
        out = self._run_linear_odr(x, y, sx, sy, beta0=(0.1, 0.25))
        assert out.beta[0] == pytest.approx(0.109, abs=0.005)


# ---------------------------------------------------------------------------
# scipy.interpolate — sky background model (externaldata.py)
# ---------------------------------------------------------------------------

class TestScipyInterpolate:
    """interp2d / RectBivariateSpline used in externaldata.py for SDSS sky model."""

    def test_interp1d_linear(self):
        from scipy.interpolate import interp1d
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        f = interp1d(x, y, kind='linear')
        assert f(1.5) == pytest.approx(3.0)

    def test_interp1d_cubic(self):
        from scipy.interpolate import interp1d
        x = np.linspace(0, np.pi, 10)
        y = np.sin(x)
        f = interp1d(x, y, kind='cubic')
        assert f(np.pi / 2) == pytest.approx(1.0, abs=0.01)

    def test_interp1d_returns_array(self):
        from scipy.interpolate import interp1d
        x = np.linspace(0, 1, 5)
        y = x ** 2
        f = interp1d(x, y)
        result = f(np.linspace(0.1, 0.9, 10))
        assert isinstance(result, np.ndarray)

    def test_rectbivariate_identity_surface(self):
        """RectBivariateSpline: flat surface should interpolate as flat."""
        from scipy.interpolate import RectBivariateSpline
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        z = np.ones((10, 10)) * 42.0
        f = RectBivariateSpline(x, y, z)
        assert f(5.0, 5.0) == pytest.approx(42.0, abs=0.01)

    def test_rectbivariate_grid_eval(self):
        """RectBivariateSpline grid evaluation returns 2-D array."""
        from scipy.interpolate import RectBivariateSpline
        x = np.arange(8, dtype=float)
        y = np.arange(8, dtype=float)
        z = np.outer(x, y)
        f = RectBivariateSpline(x, y, z)
        xi = np.linspace(1, 6, 20)
        yi = np.linspace(1, 6, 20)
        result = f(xi, yi)
        assert result.shape == (20, 20)

    def test_interp2d_linear_sky_model(self):
        """RegularGridInterpolator is the modern replacement for interp2d used in externaldata.py."""
        from scipy.interpolate import RegularGridInterpolator
        allsky = np.ones((5, 5)) * 100.0
        x = np.arange(5, dtype=float)
        y = np.arange(5, dtype=float)
        f = RegularGridInterpolator((x, y), allsky, method='linear')
        pts = np.array([[2.0, 2.0], [1.5, 3.5]])
        result = f(pts)
        np.testing.assert_allclose(result, 100.0, atol=1e-6)


# ---------------------------------------------------------------------------
# scipy.ndimage.median_filter — cosmics.py background estimation
# ---------------------------------------------------------------------------

class TestNdimageMedianFilter:
    """median_filter with size=5 mode='mirror' is the primary use in cosmics.py."""

    def test_uniform_unchanged(self):
        from scipy.ndimage import median_filter
        data = np.ones((11, 11)) * 42.0
        result = median_filter(data, size=5, mode='mirror')
        np.testing.assert_allclose(result, 42.0)

    def test_spike_removed(self):
        """A single spike pixel should be replaced by the local median."""
        from scipy.ndimage import median_filter
        data = np.ones((11, 11)) * 10.0
        data[5, 5] = 9999.0
        result = median_filter(data, size=5, mode='mirror')
        assert result[5, 5] == pytest.approx(10.0)

    def test_output_shape_preserved(self):
        from scipy.ndimage import median_filter
        data = np.random.default_rng(3).normal(100, 5, (20, 20))
        result = median_filter(data, size=5, mode='mirror')
        assert result.shape == data.shape

    def test_mirror_mode_no_edge_artefacts(self):
        """mode='mirror' should not introduce NaNs or extreme values at edges."""
        from scipy.ndimage import median_filter
        data = np.ones((10, 10)) * 5.0
        result = median_filter(data, size=3, mode='mirror')
        assert np.all(np.isfinite(result))

    def test_size3_vs_size5_difference(self):
        """size=3 and size=5 filters should produce different results on noisy data."""
        from scipy.ndimage import median_filter
        rng = np.random.default_rng(10)
        data = rng.normal(0, 1, (20, 20))
        r3 = median_filter(data, size=3, mode='mirror')
        r5 = median_filter(data, size=5, mode='mirror')
        assert not np.allclose(r3, r5)

    def test_cosmics_double_filter_pattern(self):
        """cosmics.py applies m3 = median(3), m37 = median(m3, 7); both shapes match."""
        from scipy.ndimage import median_filter
        data = np.random.default_rng(20).normal(500, 10, (32, 32))
        m3 = median_filter(data, size=3, mode='mirror')
        m37 = median_filter(m3, size=7, mode='mirror')
        assert m3.shape == data.shape
        assert m37.shape == data.shape


# ---------------------------------------------------------------------------
# scipy.ndimage.binary_dilation — cosmics.py cosmic-ray mask growing
# ---------------------------------------------------------------------------

class TestNdimageBinaryDilation:
    """binary_dilation grows the cosmic-ray mask in cosmics.py."""

    def test_single_pixel_dilates_to_cross(self):
        """Default structure dilates a single pixel into a + cross."""
        from scipy.ndimage import binary_dilation
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        dilated = binary_dilation(mask)
        # centre + 4-connected neighbours = 5 pixels
        assert dilated.sum() == 5

    def test_custom_structure(self):
        """3×3 all-ones structure dilates to a 3×3 square."""
        from scipy.ndimage import binary_dilation
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        structure = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(mask, structure=structure, iterations=1)
        assert dilated.sum() == 9  # 3×3 square

    def test_output_is_bool(self):
        from scipy.ndimage import binary_dilation
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = binary_dilation(mask)
        assert result.dtype == bool

    def test_dilation_grows_mask(self):
        from scipy.ndimage import binary_dilation
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 5] = True
        d1 = binary_dilation(mask, iterations=1)
        d2 = binary_dilation(mask, iterations=2)
        assert d2.sum() > d1.sum()

    def test_iterations_two_used_in_cosmics(self):
        """cosmics.py calls binary_dilation with iterations=2 for saturated pixels."""
        from scipy.ndimage import binary_dilation
        mask = np.zeros((15, 15), dtype=bool)
        mask[7, 7] = True
        result = binary_dilation(mask, iterations=2)
        # Should reach at least 13 pixels
        assert result.sum() >= 13


# ---------------------------------------------------------------------------
# scipy.ndimage.find_objects — cosmics.py bounding-box slices for blobs
# ---------------------------------------------------------------------------

class TestNdimageFindObjects:
    """find_objects returns bounding-box slices for each labelled blob."""

    def test_single_blob_returns_one_slice(self):
        from scipy.ndimage import label, find_objects
        mask = np.zeros((10, 10), dtype=int)
        mask[3:6, 3:6] = 1
        labeled, n = label(mask)
        objects = find_objects(labeled)
        assert len(objects) == 1

    def test_two_blobs_return_two_slices(self):
        from scipy.ndimage import label, find_objects
        mask = np.zeros((10, 10), dtype=int)
        mask[1:3, 1:3] = 1
        mask[7:9, 7:9] = 1
        labeled, n = label(mask)
        objects = find_objects(labeled)
        assert len(objects) == 2

    def test_slice_covers_blob(self):
        from scipy.ndimage import label, find_objects
        mask = np.zeros((12, 12), dtype=int)
        mask[4:7, 4:7] = 1  # 3×3 blob
        labeled, _ = label(mask)
        obj = find_objects(labeled)[0]
        region = labeled[obj]
        assert (region > 0).any()

    def test_empty_mask_returns_empty_list(self):
        from scipy.ndimage import find_objects
        labeled = np.zeros((5, 5), dtype=int)
        objects = find_objects(labeled)
        assert objects == []


# ---------------------------------------------------------------------------
# scipy.ndimage.sum — blob-size measurement in cosmics.py
# ---------------------------------------------------------------------------

class TestNdimageSum:
    """ndimage.sum measures the size of each labelled cosmic-ray blob."""

    def test_sum_single_blob(self):
        from scipy.ndimage import label
        import scipy.ndimage as ndimage
        mask = np.zeros((8, 8), dtype=int)
        mask[2:5, 2:5] = 1  # 9 pixels
        labeled, n = label(mask)
        sizes = ndimage.sum(mask.ravel(), labeled.ravel(), np.arange(1, n + 1))
        assert sizes[0] == pytest.approx(9.0)

    def test_sum_two_blobs_different_sizes(self):
        from scipy.ndimage import label
        import scipy.ndimage as ndimage
        mask = np.zeros((10, 10), dtype=int)
        mask[1:3, 1:3] = 1   # 4 pixels
        mask[6:9, 6:9] = 1   # 9 pixels
        labeled, n = label(mask)
        sizes = ndimage.sum(mask.ravel(), labeled.ravel(), np.arange(1, n + 1))
        assert sorted(sizes) == pytest.approx([4.0, 9.0])

    def test_sum_returns_array(self):
        from scipy.ndimage import label
        import scipy.ndimage as ndimage
        mask = np.ones((4, 4), dtype=int)
        labeled, n = label(mask)
        sizes = ndimage.sum(mask.ravel(), labeled.ravel(), np.arange(1, n + 1))
        assert isinstance(sizes, (list, np.ndarray))


# ---------------------------------------------------------------------------
# scipy.signal.convolve2d with boundary='symm' — cosmics.py Laplacian detection
# ---------------------------------------------------------------------------

class TestSignalConvolve2dSymm:
    """convolve2d with boundary='symm' is used in cosmics.py for Laplacian filtering."""

    def test_symm_boundary_no_edge_zeros(self):
        """boundary='symm' should not produce zero-padded edges."""
        from scipy.signal import convolve2d
        data = np.ones((9, 9))
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Laplacian
        result = convolve2d(data, kernel, mode='same', boundary='symm')
        # Uniform input → Laplacian should be zero everywhere
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_laplacian_detects_spike(self):
        """Laplacian of a spike image should have a strong response at the spike."""
        from scipy.signal import convolve2d
        data = np.zeros((11, 11))
        data[5, 5] = 100.0
        laplkernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        result = convolve2d(data, laplkernel, mode='same', boundary='symm')
        assert result[5, 5] != 0.0
        assert abs(result[5, 5]) > abs(result[0, 0])

    def test_output_shape_matches_input(self):
        from scipy.signal import convolve2d
        data = np.random.default_rng(99).normal(0, 1, (15, 15))
        kernel = np.ones((3, 3)) / 9.0
        result = convolve2d(data, kernel, mode='same', boundary='symm')
        assert result.shape == data.shape

    def test_bool_mask_convolution_pattern(self):
        """cosmics.py converts bool mask to float32 before convolving."""
        from scipy.signal import convolve2d
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 5] = True
        growkernel = np.ones((3, 3), dtype=np.float32)
        result = np.asarray(
            convolve2d(np.asarray(mask, dtype=np.float32), growkernel,
                       mode='same', boundary='symm'),
            dtype=bool,
        )
        # After dilation with all-ones 3×3, 9 pixels should be True
        assert result.sum() == 9

    def test_negative_convolve_preserves_shape(self):
        """cosmics.py applies -signal.convolve2d(...) for fine-structure detection."""
        from scipy.signal import convolve2d
        data = np.random.default_rng(55).normal(500, 10, (16, 16))
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        result = -convolve2d(data, kernel, mode='same', boundary='symm')
        assert result.shape == data.shape
