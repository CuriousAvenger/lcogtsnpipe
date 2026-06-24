"""
Tests for matplotlib usage in lcogtsnpipe.

The pipeline uses matplotlib in myloopdef.py to display astrometric
residual plots interactively, and in lscabsphotdef.py for zeropoint/
colour-term calibration plots (fitcol, fitcol2, fitcol3, calcZC,
absphot).  Tests verify the non-interactive (Agg backend) behaviour
that would run during automated pipeline operation.

All tests run without a display using the Agg backend.

Coverage additions (vs. original):
- plt.cla() / plt.clf()                   (fitcol3, fitcol2)
- plt.gcf() / plt.gca()                   (calcZC, fitcol3)
- plt.autoscale(False) + plt.axis() tuple (calcZC)
- plt.figtext()                            (fitcol, fitcol2)
- plt.setp(text_obj, text=…)              (onkeypress, onclick)
- plt.subplots(squeeze=False)             (absphot multi-colour grid)
- plt.axes(existing_ax)                   (absphot)
- fig.canvas.mpl_connect / mpl_disconnect (fitcol, fitcol3)
- line artist .remove() / lines.pop(0)   (onclick, onkeypress)
- scatter with picker kwarg               (calcZC)
- dual errorbar with xerr + linestyle     (calcZC)
- plt.xlim() / plt.ylim() module calls    (fitcol)
- multi-line overlay on same axes         (myloopdef / fitcol2)
- plt.draw() and plt.close()              (fitcol, fitcol2)
- plt.ion() entry                         (fitcol, fitcol2, myloopdef)
"""
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _use_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Import and backend
# ---------------------------------------------------------------------------

class TestMatplotlibImport:
    def test_matplotlib_importable(self):
        import matplotlib
        assert hasattr(matplotlib, "__version__")

    def test_pyplot_importable(self):
        plt = _use_agg()
        assert hasattr(plt, "plot")

    def test_agg_backend_set(self):
        import matplotlib
        matplotlib.use("Agg")
        assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# Figure / axes creation — mirrors myloopdef.py plt.plot() calls
# ---------------------------------------------------------------------------

class TestFigureCreation:
    def test_create_figure(self):
        plt = _use_agg()
        fig = plt.figure()
        assert fig is not None
        plt.close(fig)

    def test_subplots_returns_ax(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        assert ax is not None
        plt.close(fig)

    def test_multiple_subplots_shape(self):
        plt = _use_agg()
        fig, axes = plt.subplots(2, 2)
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_figure_size(self):
        plt = _use_agg()
        fig = plt.figure(figsize=(8, 6))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8.0)
        assert h == pytest.approx(6.0)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot commands — mirrors residual/scatter plots in myloopdef.py
# ---------------------------------------------------------------------------

class TestPlotCommands:
    def test_line_plot(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        line, = ax.plot(x, y)
        assert len(line.get_xdata()) == 50
        plt.close(fig)

    def test_scatter_plot(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 360, 30)
        y = rng.uniform(-1, 1, 30)
        sc = ax.scatter(x, y, c="red")
        assert sc is not None
        plt.close(fig)

    def test_plot_with_fmt_string(self):
        """Mirrors myloopdef.py: plt.plot(rra, ddec, 'or')"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        rng = np.random.default_rng(1)
        rra = rng.normal(0, 1, 20)
        ddec = rng.normal(0, 1, 20)
        ax.plot(rra, ddec, "or")
        ax.plot(rra[:5], ddec[:5], "xb", markersize=10)
        plt.close(fig)

    def test_histogram(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1, 200)
        n, bins, patches = ax.hist(data, bins=20)
        assert sum(n) == 200
        plt.close(fig)

    def test_errorbar(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        x = np.arange(5, dtype=float)
        y = np.ones(5) * 18.5
        yerr = np.ones(5) * 0.05
        eb = ax.errorbar(x, y, yerr=yerr, fmt="o")
        assert eb is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Axis labels / tick labels — mirrors myloopdef.py setp/getp calls
# ---------------------------------------------------------------------------

class TestAxisLabels:
    def test_xlabel_ylabel(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.set_xlabel("ra (arcsec)")
        ax.set_ylabel("dec (arcsec)")
        assert ax.get_xlabel() == "ra (arcsec)"
        assert ax.get_ylabel() == "dec (arcsec)"
        plt.close(fig)

    def test_title(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.set_title("Astrometric residuals")
        assert "residuals" in ax.get_title()
        plt.close(fig)

    def test_xlim_ylim(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        assert ax.get_xlim() == pytest.approx((-5.0, 5.0))
        assert ax.get_ylim() == pytest.approx((-5.0, 5.0))
        plt.close(fig)

    def test_tick_label_fontsize_via_setp(self):
        """Mirrors myloopdef.py: plt.setp(xticklabels, fontsize='20')"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        xticklabels = ax.get_xticklabels()
        plt.setp(xticklabels, fontsize=20)
        # setp doesn't raise — just verify no exception
        plt.close(fig)


# ---------------------------------------------------------------------------
# Save to file / buffer (non-interactive rendering)
# ---------------------------------------------------------------------------

class TestSaveFig:
    def test_savefig_to_png(self, tmp_path):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        path = tmp_path / "test.png"
        fig.savefig(str(path))
        assert path.exists()
        assert path.stat().st_size > 0
        plt.close(fig)

    def test_savefig_to_pdf(self, tmp_path):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.scatter([0, 1], [0, 1])
        path = tmp_path / "test.pdf"
        fig.savefig(str(path))
        assert path.exists()
        plt.close(fig)

    def test_savefig_to_bytes_buffer(self):
        import io
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        header = buf.read(8)
        # PNG magic bytes
        assert header[:4] == b"\x89PNG"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Clear-axis / clear-figure / draw — mirrors fitcol3, fitcol2, fitcol
# ---------------------------------------------------------------------------

class TestClearAndDraw:
    """Covers plt.cla(), plt.clf(), plt.draw() used in fitcol3 / fitcol2."""

    def test_cla_clears_axes(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        assert len(ax.lines) == 1
        plt.cla()  # fitcol3 calls plt.cla() in interactive mode
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_clf_clears_figure(self):
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        plt.clf()  # fitcol2 calls plt.clf() before re-drawing
        # After clf the original axes object is detached; fig should have fresh axes
        assert len(fig.get_axes()) == 0
        plt.close(fig)

    def test_draw_renders_without_error(self):
        """plt.draw() used in fitcol and fitcol2 after updating data."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [10, 11, 12])
        plt.draw()  # should not raise on Agg backend
        plt.close(fig)

    def test_close_current_figure(self):
        """plt.close() without args — used at end of fitcol."""
        plt = _use_agg()
        fig = plt.figure()
        n_before = len(plt.get_fignums())
        plt.close()  # closes the current figure
        assert len(plt.get_fignums()) == n_before - 1


# ---------------------------------------------------------------------------
# figtext / setp — mirrors fitcol and onkeypress / onclick handlers
# ---------------------------------------------------------------------------

class TestFigtextAndSetp:
    """Covers plt.figtext() and plt.setp(text, text=…) used in fitcol."""

    def test_figtext_returns_text_artist(self):
        """fitcol: testo = plt.figtext(.2, .85, '…')"""
        plt = _use_agg()
        fig = plt.figure()
        testo = plt.figtext(0.2, 0.85, "12.345 + r* 0.054 [0.002  0.001]")
        assert testo is not None
        assert testo.get_text() == "12.345 + r* 0.054 [0.002  0.001]"
        plt.close(fig)

    def test_setp_updates_text_object(self):
        """onkeypress/onclick: plt.setp(testo, text='…') updates label."""
        plt = _use_agg()
        fig = plt.figure()
        testo = plt.figtext(0.2, 0.85, "old label")
        plt.setp(testo, text="new label")
        assert testo.get_text() == "new label"
        plt.close(fig)

    def test_setp_on_list_of_text_objects(self):
        """setp accepts a list, e.g. tick labels."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        labels = ax.get_xticklabels()
        plt.setp(labels, fontsize=14)  # should not raise
        plt.close(fig)


# ---------------------------------------------------------------------------
# Module-level xlim/ylim, gcf, gca — mirrors fitcol / calcZC
# ---------------------------------------------------------------------------

class TestModuleLevelAxState:
    """plt.xlim, plt.ylim, plt.gcf, plt.gca as used in fitcol and calcZC."""

    def test_pyplot_xlim_ylim_set_and_get(self):
        """fitcol: plt.xlim(min(xx), max(xx)) / plt.ylim(min,max)"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        xx = [-0.5, 2.5]
        plt.xlim(min(xx), max(xx))
        plt.ylim(-0.2, 0.5)
        xlo, xhi = plt.xlim()
        ylo, yhi = plt.ylim()
        assert xlo == pytest.approx(-0.5)
        assert xhi == pytest.approx(2.5)
        assert ylo == pytest.approx(-0.2)
        assert yhi == pytest.approx(0.5)
        plt.close(fig)

    def test_gcf_returns_current_figure(self):
        """plt.gcf() used in fitcol3 for canvas event connection."""
        plt = _use_agg()
        fig = plt.figure()
        assert plt.gcf() is fig
        plt.close(fig)

    def test_gca_returns_current_axes(self):
        """plt.gca() used in calcZC to check autoscale state."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        assert plt.gca() is ax
        plt.close(fig)

    def test_gca_get_autoscale_on_default_true(self):
        """calcZC: if not plt.gca().get_autoscale_on() → read lims."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        assert ax.get_autoscale_on() is True
        plt.close(fig)

    def test_autoscale_false_freezes_limits(self):
        """calcZC: plt.autoscale(False) so subsequent plots don't rescale."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plt.autoscale(False)
        assert ax.get_autoscale_on() is False
        # Add data far outside range — limits should not change
        before = ax.get_xlim()
        ax.plot([100, 200], [100, 200])
        after = ax.get_xlim()
        assert before == pytest.approx(after)
        plt.close(fig)

    def test_pyplot_axis_get_returns_4tuple(self):
        """calcZC: xx = np.array(plt.axis()[0:2])"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        lims = plt.axis()
        assert len(lims) == 4
        plt.close(fig)

    def test_pyplot_axis_set_from_list(self):
        """calcZC: plt.axis(lims) — set with a 4-element sequence."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plt.axis([0.0, 2.0, 0.0, 3.0])
        xlo, xhi, ylo, yhi = plt.axis()
        assert xlo == pytest.approx(0.0)
        assert xhi == pytest.approx(2.0)
        assert ylo == pytest.approx(0.0)
        assert yhi == pytest.approx(3.0)
        plt.close(fig)


# ---------------------------------------------------------------------------
# scatter + dual errorbar — mirrors calcZC show path
# ---------------------------------------------------------------------------

class TestCalcZCPlots:
    """Covers scatter(picker=5) and dual-colour errorbar(xerr=…, linestyle='none')."""

    def _make_data(self):
        rng = np.random.default_rng(42)
        n = 20
        colors = rng.uniform(-0.5, 1.5, n)
        deltas = 0.05 * colors + rng.normal(0, 0.02, n) + 23.5
        dcolors = rng.uniform(0.01, 0.05, n)
        ddeltas = rng.uniform(0.01, 0.05, n)
        keep = np.ones(n, dtype=bool)
        keep[np.abs(deltas - deltas.mean()) > 2 * deltas.std()] = False
        return colors, deltas, dcolors, ddeltas, keep

    def test_scatter_with_picker(self):
        """calcZC: plt.scatter(colors, deltas, marker='.', picker=5)"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        colors, deltas, _, _, _ = self._make_data()
        sc = plt.scatter(colors, deltas, marker=".", picker=5)
        assert sc is not None
        plt.close(fig)

    def test_errorbar_green_kept_stars(self):
        """calcZC: plt.errorbar(colors[keep], …, color='g', linestyle='none')"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        colors, deltas, dcolors, ddeltas, keep = self._make_data()
        eb = plt.errorbar(
            colors[keep], deltas[keep],
            xerr=dcolors[keep], yerr=ddeltas[keep],
            color="g", marker="o", linestyle="none",
        )
        assert eb is not None
        plt.close(fig)

    def test_errorbar_red_rejected_stars(self):
        """calcZC: plt.errorbar(colors[~keep], …, color='r', linestyle='none')"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        colors, deltas, dcolors, ddeltas, keep = self._make_data()
        # force at least one rejection
        keep[-1] = False
        eb = plt.errorbar(
            colors[~keep], deltas[~keep],
            xerr=dcolors[~keep], yerr=ddeltas[~keep],
            color="r", marker="o", linestyle="none",
        )
        assert eb is not None
        plt.close(fig)

    def test_autoscale_off_then_axis_read(self):
        """calcZC: autoscale(False) then xx = np.array(plt.axis()[0:2])"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        colors, deltas, _, _, _ = self._make_data()
        plt.scatter(colors, deltas, marker=".")
        plt.autoscale(False)
        xx = np.array(plt.axis()[0:2])
        assert xx.shape == (2,)
        assert xx[0] < xx[1]
        plt.close(fig)

    def test_dashed_regression_line(self):
        """calcZC: plt.plot(xx, yy, '--')"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        xx = np.array([-0.5, 1.5])
        yy = 23.5 + 0.05 * xx
        line, = plt.plot(xx, yy, "--")
        assert line.get_linestyle() in ("--", "dashed")
        plt.close(fig)


# ---------------------------------------------------------------------------
# subplots(squeeze=False) and axes() — mirrors absphot multi-colour grid
# ---------------------------------------------------------------------------

class TestSubplotsMulticolumn:
    """Covers plt.subplots(ncols=N, squeeze=False) and plt.axes(axarr[0,i])."""

    def test_subplots_squeeze_false_shape(self):
        """absphot: fig, axarr = plt.subplots(ncols=3, figsize=(24,6), squeeze=False)"""
        plt = _use_agg()
        fig, axarr = plt.subplots(ncols=3, figsize=(24, 6), squeeze=False)
        assert axarr.shape == (1, 3)
        plt.close(fig)

    def test_axes_select_subplot(self):
        """absphot: plt.axes(axarr[0, i]) — make axarr[0,i] the current axes."""
        plt = _use_agg()
        fig, axarr = plt.subplots(ncols=2, squeeze=False)
        target_ax = axarr[0, 1]
        plt.sca(target_ax)          # equivalent to plt.axes(existing_ax)
        assert plt.gca() is target_ax
        plt.close(fig)

    def test_plot_into_each_subplot(self):
        """Mirrors the per-color-index loop in absphot."""
        plt = _use_agg()
        fig, axarr = plt.subplots(ncols=3, figsize=(24, 6), squeeze=False)
        rng = np.random.default_rng(7)
        for i in range(3):
            plt.sca(axarr[0, i])
            colors = rng.uniform(-0.5, 1.5, 15)
            deltas = rng.normal(0, 0.05, 15) + 23.0
            plt.scatter(colors, deltas, marker=".")
            plt.xlabel("Color index %d" % i)
            plt.title("Filter g col %d" % i)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Canvas event connect / disconnect — mirrors fitcol and fitcol3
# ---------------------------------------------------------------------------

class TestCanvasEvents:
    """Covers fig.canvas.mpl_connect and mpl_disconnect on Agg canvas."""

    def test_mpl_connect_returns_cid(self):
        """fitcol: cid = fig.canvas.mpl_connect('key_press_event', handler)"""
        plt = _use_agg()
        fig = plt.figure()
        cid = fig.canvas.mpl_connect("key_press_event", lambda event: None)
        assert isinstance(cid, int)
        plt.close(fig)

    def test_mpl_disconnect_does_not_raise(self):
        """fitcol: fig.canvas.mpl_disconnect(kid)"""
        plt = _use_agg()
        fig = plt.figure()
        cid = fig.canvas.mpl_connect("button_press_event", lambda event: None)
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

    def test_gcf_canvas_mpl_connect_and_disconnect(self):
        """fitcol3: cid = plt.gcf().canvas.mpl_connect('pick_event', …)
                    plt.gcf().canvas.mpl_disconnect(cid)"""
        plt = _use_agg()
        fig = plt.figure()
        cid = plt.gcf().canvas.mpl_connect("pick_event", lambda event: None)
        plt.gcf().canvas.mpl_disconnect(cid)
        plt.close(fig)

    def test_multiple_event_types_connectable(self):
        """fitcol uses both 'key_press_event' and 'button_press_event'."""
        plt = _use_agg()
        fig = plt.figure()
        cid1 = fig.canvas.mpl_connect("key_press_event", lambda e: None)
        cid2 = fig.canvas.mpl_connect("button_press_event", lambda e: None)
        assert cid1 != cid2
        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Line artist removal — mirrors onclick / onkeypress in lscabsphotdef.py
# ---------------------------------------------------------------------------

class TestLineArtistRemoval:
    """Covers lines.pop(0).remove() pattern used in onclick / onkeypress."""

    def test_line_remove_from_axes(self):
        """onkeypress: lines.pop(0).remove()"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        lines = ax.plot([0, 1], [1, 2], "r-")
        assert len(ax.lines) == 1
        lines.pop(0).remove()
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_replace_line_pattern(self):
        """onclick/onkeypress: remove old regression line, add new one."""
        plt = _use_agg()
        fig, ax = plt.subplots()
        xx = [0.0, 1.0]
        yy_old = [23.5, 23.55]
        lines = ax.plot(xx, yy_old, "r-")
        # --- update step (mirrors the handler body) ---
        lines.pop(0).remove()
        yy_new = [23.6, 23.62]
        lines = ax.plot(xx, yy_new, "r-")
        assert len(ax.lines) == 1
        assert list(ax.lines[0].get_ydata()) == pytest.approx(yy_new)
        plt.close(fig)

    def test_ok_and_ow_overlay(self):
        """onclick: plt.plot(_col, _dmag, 'ok') then plt.plot(nonincl, …, 'ow')"""
        plt = _use_agg()
        fig, ax = plt.subplots()
        rng = np.random.default_rng(3)
        col = rng.uniform(-0.5, 1.5, 10)
        dmag = rng.normal(0, 0.1, 10)
        nonincl = [2, 5, 8]
        ax.plot(col, dmag, "ok")
        ax.plot(col[nonincl], dmag[nonincl], "ow")
        assert len(ax.lines) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# ion() entry and plt.close() — mirrors fitcol, fitcol2, myloopdef
# ---------------------------------------------------------------------------

class TestIonAndClose:
    """plt.ion() and plt.close() used in interactive pipeline sections."""

    def test_ion_does_not_raise(self):
        """fitcol / fitcol2 / myloopdef: plt.ion() before drawing."""
        plt = _use_agg()
        plt.ion()  # should not raise on Agg backend
        plt.ioff()  # restore for other tests

    def test_close_no_args_removes_figure(self):
        """fitcol: plt.close() at end of interactive session."""
        plt = _use_agg()
        fig = plt.figure()
        n_before = len(plt.get_fignums())
        plt.close()
        assert len(plt.get_fignums()) == n_before - 1

    def test_close_with_figure_ref(self):
        """Explicit plt.close(fig) used throughout pipeline."""
        plt = _use_agg()
        fig = plt.figure()
        num = fig.number
        plt.close(fig)
        assert num not in plt.get_fignums()


# ---------------------------------------------------------------------------
# Multi-line overlay — mirrors fitcol2 show path and myloopdef residuals
# ---------------------------------------------------------------------------

class TestMultiLineOverlay:
    """Covers plt.ion() + multiple plt.plot() calls on the same figure."""

    def test_fitcol2_show_path_overlay(self):
        """fitcol2 show: plt.plot(_col,_dmag,'ob'), plt.plot(xx0,yy0,'xr'), plt.plot(_col,yy,'-g')"""
        plt = _use_agg()
        plt.ion()
        fig = plt.figure()
        rng = np.random.default_rng(5)
        _col = rng.uniform(-0.5, 1.5, 15)
        _dmag = rng.normal(23.5, 0.05, 15)
        yy = 23.5 + 0.05 * _col
        plt.clf()
        plt.plot(_col, _dmag, "ob")
        plt.plot(_col[:5], _dmag[:5], "xr")
        plt.plot(_col, yy, "-g")
        ax = plt.gca()
        assert len(ax.lines) == 3
        plt.ylabel("zeropoint")
        plt.xlabel("g")
        plt.title("r")
        plt.draw()
        plt.close(fig)
        plt.ioff()

    def test_myloopdef_residual_overlay(self):
        """myloopdef: plt.ion(); plt.plot(rra, ddec, 'or'); plt.plot(rracut, ddeccut, 'xb')"""
        plt = _use_agg()
        plt.ion()
        rng = np.random.default_rng(6)
        rra = rng.normal(0, 0.3, 50)
        ddec = rng.normal(0, 0.3, 50)
        mask = np.abs(rra) < 0.2
        rracut, ddeccut = rra[mask], ddec[mask]
        fig = plt.figure()
        plt.plot(rra, ddec, "or")
        plt.plot(rracut, ddeccut, "xb")
        ax = plt.gca()
        assert len(ax.lines) == 2
        plt.close(fig)
        plt.ioff()
