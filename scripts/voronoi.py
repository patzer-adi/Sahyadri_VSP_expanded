import numpy as np
from scipy.spatial import cKDTree


class Voronoi:
    """
    Monte-Carlo estimator for Voronoi cell volume fractions
    in a periodic simulation box.

    Parameters
    ----------
    positions : (N,3) ndarray
        Cartesian coordinates of tracer particles.
    Lbox : float
        Simulation box size.
    seed : int, optional
        Random seed.
    """

    def __init__(self, positions, Lbox, seed=10):
        """ Initializes cKDTree and rng. """
        self.positions = np.asarray(positions, dtype=float) % Lbox
        self.Lbox = float(Lbox)
        self.seed = seed

        self.Ntrc = self.positions.shape[0]

        print("Building periodic cKDTree...")
        self.tree = cKDTree(self.positions, boxsize=self.Lbox)
        self.rng = np.random.RandomState(seed)

    def voronoi_periodic_box(self,
                             ran_fac,
                             chunk_size=int(1e7),
                             return_counts=False):
        """
        Estimate Voronoi volume fractions using random sampling.
        This is a space optimized code where the KNN computation is done in chunks
        rather than all at once. There is no loss in time complexity.

        Parameters
        ----------
        ran_fac : int
            Number of random points per tracer.
        chunk_size : int
            Number of random points processed simultaneously.
        return_counts : bool
            If True also return neighbour counts.

        Returns
        -------
        y : ndarray
            Voronoi volume fraction for every tracer.

        (optional)
        nbr_count : ndarray
            Number of random points assigned to each tracer.
        """

        Nran = int(ran_fac * self.Ntrc)

        nbr_count = np.zeros(self.Ntrc, dtype=np.int64)

        Nran_done = 0

        while Nran_done < Nran:

            n = min(chunk_size, Nran - Nran_done)

            ran = self.Lbox * self.rng.rand(n, 3)

            _, idx = self.tree.query(ran, k=1, workers=-1)

            nbr_count += np.bincount(idx, minlength=self.Ntrc)

            Nran_done += n

        delta = ran_fac / (nbr_count + 1e-15) - 1.0
        y = 1.0 / (1.0 + delta)

        if return_counts:
            return y, nbr_count

        return y
