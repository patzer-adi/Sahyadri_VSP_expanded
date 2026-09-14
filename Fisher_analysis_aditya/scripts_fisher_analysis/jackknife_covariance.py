"""
jackknife_covariance.py — delete-one-region jackknife covariance
estimator for wp(rp). Consumes an ALREADY region-tagged catalogue
(x, y, z, jn_region) — e.g. loaded from the saved jackknife FITS file
via hod_pipeline.load_jackknife_catalog — rather than re-deriving
region assignment itself. This keeps "assign regions -> save -> read
-> compute" as one genuinely reproducible chain instead of silently
repeating the assignment step.
"""
import numpy as np


class JackknifeCovariance:
    """
    Generic delete-one-region jackknife covariance estimator.

    statistic_func : callable, signature (mock_dict, lbox=...) -> (x_values, y_values)
                      e.g. hod_pipeline.compute_wp
    """

    def __init__(self, tagged_data, Lbox, statistic_func, njn=100):
        """
        tagged_data : (N, 4) array — x, y, z, jn_region (already assigned,
                      e.g. from hod_pipeline.load_jackknife_catalog)
        Lbox        : box size, passed through to statistic_func
        njn         : expected number of regions (for a sanity check)
        """
        self.tagged_data = tagged_data
        self.Lbox = Lbox
        self.statistic_func = statistic_func
        self.njn = njn

    def compute(self, position_to_mock, verbose=True):
        region_ids = self.tagged_data[:, 3].astype(int)
        unique_regions = np.unique(region_ids)
        assert len(unique_regions) == self.njn, \
            f"Expected {self.njn} regions in tagged_data, got {len(unique_regions)}"

        x_ref = None
        y_stack = []
        for i, reg in enumerate(unique_regions):
            mask = region_ids != reg   # delete this region
            sub_positions = self.tagged_data[mask, :3]
            mock_sub = position_to_mock(sub_positions)
            x_vals, y_vals = self.statistic_func(mock_sub, lbox=self.Lbox)
            if x_ref is None:
                x_ref = x_vals
            y_stack.append(y_vals)
            if verbose and (i + 1) % 20 == 0:
                print(f'  ... region {i+1}/{self.njn} done')

        y_stack = np.array(y_stack)
        y_mean = np.mean(y_stack, axis=0)
        diff = y_stack - y_mean
        n = self.njn
        cov = (n - 1) / n * (diff.T @ diff)
        diag_err = np.sqrt(np.diag(cov))

        self.x = x_ref
        self.y_mean = y_mean
        self.cov = cov
        self.diag_err = diag_err
        self.y_stack = y_stack

        if verbose:
            print(f'Jackknife covariance complete: njn={self.njn}, {len(x_ref)} bins.')
        return x_ref, y_mean, cov, diag_err

    def sanity_check(self, full_x, full_y, rtol=0.15):
        rel_diff = np.abs(self.y_mean - full_y) / np.abs(full_y)
        max_rel_diff = np.max(rel_diff)
        ok = max_rel_diff < rtol
        print(f'Jackknife mean vs full-sample: max rel diff = {max_rel_diff:.2%}  '
              f'({"OK" if ok else "CHECK THIS"}, tol={rtol:.0%})')
        return ok