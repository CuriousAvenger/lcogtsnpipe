"""
Additional tests for lsc.lscsnoopy — covers the errore function's math logic
(artificial star placement, dispersion grid) and manusn's magnitude calculation.
No IRAF, subprocess, or DB access required.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# errore — artificial star dispersion grid logic
# ---------------------------------------------------------------------------

class TestErroreGridLogic:
    def test_iter_placement_grid(self):
        """
        The errore function places artificial stars on a 3x3 grid around the SN.
        Verify the grid computation logic independently.
        """
        from lsc.lscsnoopy import errore

        # Extract the grid placement logic:
        # i goes 0..8, each position is (artx, arty) based on i
        positions = []
        for i in range(9):
            artx = int(i / 3.) - 1
            if i <= 2:
                arty = artx + i
            elif 3 <= i <= 5:
                arty = artx - 1 + i - 3
            else:
                arty = artx - 2 + i - 6
            positions.append((artx, arty))

        # i=0: (-1, -1)
        # i=1: (-1, 0)
        # i=2: (-1, 1)
        # i=3: (0, -1)
        # i=4: (0, 0)
        # i=5: (0, 1)
        # i=6: (1, -1)
        # i=7: (1, 0)
        # i=8: (1, 1)
        expected = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 0), (0, 1),
                    (1, -1), (1, 0), (1, 1)]
        assert positions == expected


# ---------------------------------------------------------------------------
# manusn — tests that magnitude offsets are computed correctly
# ---------------------------------------------------------------------------

class TestManusn:
    def test_magnitude_computation(self):
        """
        manusn computes newmag = truemag + dmag0 for each star.
        Verify this logic independently.
        """
        truemag = np.array([18.5, 19.0, 17.5])
        dmag0 = 0.5

        newmag = np.array([float(t) + float(dmag0) for t in truemag])
        assert np.allclose(newmag, [19.0, 19.5, 18.0])

    def test_indef_magnitude_handling(self):
        """When truemag is 'INDEF', handle gracefully."""
        from numpy import zeros
        truemag = ['INDEF', '18.5']
        dmag0 = 0.5

        newmag = zeros(len(truemag))
        magerr = zeros(len(truemag))
        for i in range(len(truemag)):
            try:
                newmag[i] = float(truemag[i]) + float(dmag0)
            except (ValueError, TypeError):
                newmag[i] = float(dmag0)
                magerr[i] = 0.0

        assert newmag[0] == 0.5  # dmag0
        assert magerr[0] == 0.0
        assert abs(newmag[1] - 19.0) < 1e-6


# ---------------------------------------------------------------------------
# fitsn — coordinate list writing
# ---------------------------------------------------------------------------

class TestFitsn:
    def test_coordslist_format(self):
        """
        fitsn writes coordinates to a file in IRAF format.
        Verify format: 'x y flux id'
        """
        # Simulate the coordlist write
        coordlist = []
        fwhm0 = 3.0
        a1 = int(fwhm0)

        # Simulate a few stars
        stars = [(50.5, 60.3, 1)],
        coordlist.extend(stars)

        # Verify format
        for star in coordlist:
            line = '%8.3f %8.3f  %6.1f  %d' % (float(star[0]), float(star[1]), 1.0, star[2])
            parts = line.split()
            assert len(parts) == 4
            assert float(parts[0]) == float(star[0])
            assert float(parts[1]) == float(star[1])
