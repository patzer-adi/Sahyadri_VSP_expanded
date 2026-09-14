import numpy as np


class JackKnife:
    """
    JackKnife class for performing JackKnife operations on a given dataset.
    This class provides a method to add JackKnife regions to a dataset.
    """

    def __init__(self, positions, Lbox):
        self.positions = positions % Lbox
        self.Lbox = float(Lbox)
        self.Ntrc = self.positions.shape[0]

    def is_perfect_power(self, number, power):
        number = abs(number)
        return round(number ** (1.0 / power)) ** power == number

    def add_jackknife_regions(self, njn=125, rand=None, los=1):
        """
        njn : number of jackknife regions (perfect square if los=1, perfect cube if los=0)
        los : 1 -> 2D jackknife (x,y binned, z/LOS left untouched)
              0 -> 3D jackknife (x,y,z all binned)
              Explicitly decides dimensionality rather than inferring from
              njn's factorization (njn=64 is ambiguous: 8^2 == 4^3).
        """
        if los == 1:
            jntype = '2d'
            if not self.is_perfect_power(njn, 2):
                raise ValueError(f"los=1 (2D) requires njn to be a perfect square; got njn={njn}")
            NJNx = int(round(np.sqrt(njn)))
            NJNy = NJNx
        elif los == 0:
            jntype = '3d'
            if not self.is_perfect_power(njn, 3):
                raise ValueError(f"los=0 (3D) requires njn to be a perfect cube; got njn={njn}")
            NJNx = int(round(njn ** (1.0 / 3)))
            NJNy = NJNx
            NJNz = NJNx
        else:
            raise ValueError("los must be 0 (3D) or 1 (2D)")

        POS_min = [0.0, 0.0, 0.0]
        POS_max = [self.Lbox, self.Lbox, self.Lbox]
        blen = POS_max

        rand_out = None
        for ii in range(2):
            if ii == 0:
                mat = self.positions
            elif rand is None:
                continue
            else:
                mat = rand

            indx = (NJNx * (mat[:, 0] - POS_min[0]) / blen[0]).astype(int)
            indy = (NJNy * (mat[:, 1] - POS_min[1]) / blen[1]).astype(int)
            indx = np.mod(indx, NJNx)
            indy = np.mod(indy, NJNy)

            if jntype == '2d':
                jnreg = NJNy * indx + indy
            else:
                indz = (NJNz * (mat[:, 2] - POS_min[2]) / blen[2]).astype(int)
                indz = np.mod(indz, NJNz)
                jnreg = NJNz * (NJNy * indx + indy) + indz

            mat_out = np.column_stack([mat, jnreg])

            if ii == 0:
                data = mat_out
                self.jnreg = jnreg
                self.njn = njn
            else:
                rand_out = mat_out

        return data, rand_out