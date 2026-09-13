import numpy as np

class JackKnife:
    """
    JackKnife class for performing JackKnife operations on a given dataset.
    This class provides method to add JackKnife regions to a dataset.
    """

    def __init__(self, positions, Lbox):
        """
        Initializes JackKnife class with postions and boxsize
        """
        self.positions = positions % Lbox
        self.Lbox = float(Lbox)
        self.Ntrc = self.positions.shape[0]

    def is_perfect_power(self, number, power):
        number = abs(number)
        return round(number ** (1 / power)) ** power == number
        
    def add_jackknife_regions(self, njn=125, rand=None, los=1):
        """
        This function adds a column to the positions array indicating the JackKnife Region each point belongs to.

        njn: number of jackknife regions. Must be a perfect square or a perfect cube
        los: line of sight. If 1, then gives 2d
        """
        if (self.is_perfect_power(njn, 3)):
            jntype = '3d'
            NJNx = int(np.round(njn ** (1. / 3)))
            NJNy = NJNx
            NJNz = NJNx
        elif (self.is_perfect_power(njn, 2)):
            jntype = '2d'
            NJNx = int(np.sqrt(njn))
            NJNy = int(njn / NJNx)
        else:
            raise ValueError("njn must be a perfect square or perfect cube")
    
        if (njn > 0 and los == 1):
            POS_min = [0, 0, 0]
            POS_max = [self.Lbox, self.Lbox, self.Lbox]
            blen = POS_max
    
            for ii in range(0, 2):
                if (ii == 0):
                    mat = self.positions
                elif rand is None:
                    continue
                else:
                    mat = rand
    
                indx = np.zeros(mat[:, 0].size, dtype=int)
                indy = np.zeros(mat[:, 0].size, dtype=int)
    
                # for kk in range(0, indx.size):
                #     indx[kk] = int(NJNx * (mat[kk, 0] - POS_min[0]) / blen[0])
                #     indy[kk] = int(NJNy * (mat[kk, 1] - POS_min[1]) / blen[1])
                indx = (NJNx * (mat[:, 0] - POS_min[0]) / blen[0]).astype(int)
                indy = (NJNy * (mat[:, 1] - POS_min[1]) / blen[1]).astype(int)

                indx = np.mod(indx, NJNx)
                indy = np.mod(indy, NJNy)
    
                if jntype == '2d':
                    jnreg = NJNy * indx + indy
                elif jntype == '3d':
                    indz = np.zeros(mat[:, 0].size, dtype=int)
                    for kk in range(0, indz.size):
                        indz[kk] = int(NJNz * (mat[kk, 2] - POS_min[2]) / blen[2])
                    indz = np.mod(indz, NJNz)
                    jnreg = NJNz * (NJNy * indx + indy) + indz
    
                mat = np.column_stack([mat, jnreg])
    
                if ii == 0:
                    data = mat
                    self.jnreg = jnreg
                    self.njn = njn

                else:
                    rand = mat
    
            return data, rand
        else:
            return 0
        
        
    
    def jackknife_percentile_stats(self, values, percentiles=(2.5, 16.0, 50.0, 84.0, 97.5), jnreg=None):
        """
        Computes jackknife-resampled percentile statistics of a per-point quantity
        (e.g. Voronoi cell volumes), following the same leave-one-region-out
        procedure as compute_voronoi_jack.py.
        
        Parameters
        ----------
        values : array_like, shape (Ntrc, Vtrc)
            Per-point quantity to compute percentiles of. Must be aligned
            (same ordering) with the positions/jnreg used to create it.
        percentiles : array_like, optional
            Percentiles (0-100) to compute. Defaults to (2.5, 16.0, 50.0, 84.0, 97.5).
        jnreg : array_like, shape (Ntrc,), optional
            Jackknife region label for each point. If not provided, uses
            self.jnreg, which is set automatically by add_jackknife_regions.
        
        Returns
        -------
        result : dict
            'percentiles'  : array of the requested percentiles
            'full'         : percentiles computed over the full sample
            'mean'         : jackknife mean of the percentiles across regions
            'err'          : jackknife error (sqrt(njn-1) * std) of the percentiles
            'realizations' : array, shape (n_percentiles, njn), the leave-one-
                             region-out percentile values for each region
            'njn'          : number of jackknife regions used

        """
        values = np.asarray(values, dtype=float)
        
        if jnreg is None:
            if not hasattr(self, 'jnreg'):
                raise ValueError(
                    "No jnreg available. Either pass jnreg explicitly or call "
                    "add_jackknife_regions first."
                )
            jnreg = self.jnreg
        
        else:
            jnreg = np.asarray(jnreg)
            
        if values.shape[0] != jnreg.shape[0]:
            raise ValueError("values and jnreg must have the same length")
        
        percentiles = np.asarray(percentiles, dtype=float)
        regions = np.unique(jnreg)
        njn = regions.size

        realizations = np.zeros((percentiles.size, njn), dtype=float)
        for i, reg in enumerate(regions):
            mask = jnreg != reg
            realizations[:, i] = np.percentile(values[mask], percentiles)
 
        full = np.percentile(values, percentiles)
        mean = np.mean(realizations, axis=1)
        err = np.sqrt(njn - 1) * np.std(realizations, axis=1)
 
        return {
            'percentiles': percentiles,
            'mean': mean,
            'err': err,
            'full': full,
            'realizations': realizations,
            'njn': njn,
        }
        
