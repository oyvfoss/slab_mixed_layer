import numpy as np
import matplotlib as mpl
import pdb, sys

## SOLVING THE MIXED LAYER SLAB MODEL ##

'''
Option of MLD varying in time, but not in space.

r taken as a constant by default. Option of taking it as a fraction of
f (r is taken as a fraction of f if rfrac is toggled on).

Reads t in days.
Reads f in Hz.
'''

class slab_solve:

    def __init__(self, r, lat, mld, uv0, rho, rfrac = False):

        self.f = 4*np.pi/86400.0 * np.sin(lat*np.pi/180.0)
        if rfrac:
            self.r = r * self.f
        else:
            self.r = r

        self.mld = np.ma.array(mld)
        self.uv0 = np.ma.array(uv0)
        self.rho = rho

    def ws(self, t, Tx, Ty, string_out = True):
        '''
        Reads time and wind stress vector. Tx can either be 3D
        ([t, x, y]) or 1D ([t]).
        '''
        self.t = t * 86400
        self.Tx = Tx
        self.Ty = Ty
        self.check_dims(string_out = string_out)


    def check_dims(self, string_out = True):
        '''
        Checking that dimensions are consistent. Assuming that the
        shape of T is the 'correct' one.
        In the case where either of MLD, uv0 or f are given as a constant,
        these are expanded into spatial arrays in the 3D case.
        '''

        if self.Tx.shape != self.Ty.shape:
            raise ValueError('T components have different shapes.')

        if self.Tx.shape[0] != self.t.shape[0]:
            raise ValueError('t should have same length as 0th'+
                              ' dimension of T.')

        if np.isnan(self.Tx).any() or np.isnan(self.Tx).any():
            raise ValueError('Wind stress forcing contains NaNs.')

        self.dims = self.Tx.ndim
        self.shape = self.Tx.shape


        if self.mld.ndim == 0 :
            self.mld = self.mld * np.ones(self.shape[0])

        if self.mld.shape[0] != self.shape[0]:
            raise ValueError('MLD must be single value or a time' +\
                             'series matching dimensions of t.')

        varstrs =  ['uv0', 'f']

        if self.dims == 1:
            for varstr in varstrs:
                self.check_0d(varstr)

        elif self.dims == 3:
            for varstr in varstrs[:-1]:
                self.check_2d(varstr)

            if self.f.ndim == 1:
                if len(self.f) == self.shape[2]:
                # Latitude can be given as a 1D ([y]) array.
                    self.f = self.f * np.ones(self.shape[1:])
            else:
                self.check_2d('f')

        else:
            raise ValueError('Only accept either 1D ([t]) or 3D'+
                             '([t, x, y] forcing.')

        for varstr in varstrs + ['Tx', 'Ty']:
            self.check_mask(varstr)
        if string_out:
            print('Dimensions OK (%iD)'%self.dims)


    def check_0d(self, varstr):
        '''
        Subfunction of check_dims
        '''
        variab = getattr(self, varstr)
        if variab.ndim > 0:
            raise ValueError('For 1D forcing, '+varstr+ ' should be'+
                             'one single value.')


    def check_2d(self, varstr):
        '''
        Subfunction of check_dims
        '''
        variab = getattr(self, varstr)
        good = False

        if variab.ndim == 0 :
            good = True
            variab = variab * np.ones(self.shape[1:])
        elif variab.ndim == 2:
            if variab.shape == self.shape[1:]:
                good = True

        if varstr == 'f':
            latstr = ', have dimensions of the [y] grid (1D), '
        else:
            latstr = ''
        msg = 'For 3D forcing, '+varstr+ ' must either be a constant'\
              +latstr+' or have dimensions of the spatial grid (2D).'
        if good == False:
            raise ValueError(msg)

        setattr(self, varstr, variab)


    def check_mask(self, varstr):
        varm = np.ma.masked_invalid(getattr(self, varstr))
        if varm.mask.any():
            raise ValueError(varstr+' contains masked/invalid indices.')


    def printshapes(self):
        '''
        Diagnostic tool: Printing the shapes of key attributes.
        '''
        print('SHAPES:')
        varstrs =  ['mld', 'uv0', 'f', 't', 'Tx', 'Ty']
        for varstr in varstrs:
            V = getattr(self, varstr)
            print(varstr+': '+str(V.shape))


    def solve_fwd(self):

        Z = np.complex_(np.zeros(self.shape))
        Z[0] = self.uv0

        E_in = np.zeros(self.shape) # Energy flux in

        T = (self.Tx + 1j * self.Ty)/self.rho
        om = self.r + 1j * self.f


        for ii in np.arange(1, len(self.t)-1):
            perc = 100 * ii/float(len(self.t)-1)
            sys.stdout.write('\rSolving: %2d%%'% perc)
            dt = self.t[ii+1] - self.t[ii]
            dTi_dt = (T[ii+1] - T[ii])/dt
            dZi_dt =  - om * Z[ii] - dTi_dt/(om*self.mld[ii])
            Z[ii+1] = Z[ii] + dt * dZi_dt
            E_in[ii] = - np.real(self.rho *Z[ii].conj() * \
                            dTi_dt/om)

            if np.isnan(Z[ii+1]).any():
                dbyn = input('Blowup. Debug (1/0)?: ')
                if dbyn:
                    pdb.set_trace()
                else:
                    raise ValueError('Blowup (check damping parameter..)')
        sys.stdout.write('\rDone.')

        # IO currents
        self.ui = Z.real
        self.vi = Z.imag

        # Ekman transport
        if om.ndim >1: # (Index object for multiplication)
            idx_obj = [slice(None)] + [None for i in xrange(om.ndim)]
        else:
            idx_obj = slice(None)

        ET = T/(self.mld[idx_obj] * om)
        self.ue = ET.real
        self.ve = ET.imag

        # Total
        self.u = self.ui + self.ue
        self.v = self.vi + self.ve

        # Near-inertial energy flux wind -> mixed layer
        self.E_in = E_in

        # Energy flux mixed layer -> dissipation
        self.E_out = self.r * Z * Z.conj() * self.rho * self.mld[idx_obj]
