import math
import numpy as np
import matplotlib.pyplot as plt
import ipdb

'''
This file is an evolution of grid classes in sweverett/Balrog-GalSim
'''

# The following base class is useful for accessing allowed parameter values
# without constructing a full config
class BaseGrid(object):

    _valid_grid_types = ['RectGrid', 'HexGrid']
    _valid_mixed_types = ['MixedGrid']

class Grid(BaseGrid):

    def __init__(self, grid_spacing, wcs, Npix_x, Npix_y, pix_scale,
                 rot_angle=None, pos_offset=None, angle_unit='rad'):
        '''
        grid_spacing: float
            Distance between "core" grid points in arcsec. May not be
            constant for all edges depending on the grid type
        wcs: astropy.WCS instance
            The world coordinate system of the image to set the grid on
        Npix_x: int
            The number of pixels along the image x-axis
        Npiy_y: int
            The number of pixels along the image y-ayis
        pix_scale: float
            The pixel scale in arcsec per pixel
        rot_angle: float
            Rotation angle of the grid wrt the image
        pos_offset: list of floats
            The positional offset of the grid in arcsec
        angle_unit: str
            The astropy unit of the rotation angle
        '''

        self.grid_spacing = grid_spacing  # arcsec
        self.im_gs = grid_spacing * (1.0 / pix_scale)  # pixels
        self.Npix_x, self.Npix_y = Npix_x, Npix_y
        self.pix_scale = pix_scale  # arcsec / pixel
        self.wcs = wcs
        self.rot_angle = rot_angle # rotation angle, in rad
        self.angle_unit = angle_unit

        if pos_offset:
            self.pos_offset = pos_offset
        else:
            self.pos_offset = [0., 0.]

        # May have to modify grid corners if there is a rotation
        if rot_angle:
            dx = Npix_x / 2.
            dy = Npix_y / 2.
            if angle_unit == 'deg':
                theta = np.deg2rad(rot_angle)
            else:
                theta = rot_angle
            self.startx = (0.-dx) * np.cos(theta) - (Npix_y-dy) * np.sin(theta) + dx
            self.endx = (Npix_x-dx) * np.cos(theta) - (0.-dy) * np.sin(theta) + dx
            self.starty = (0.-dx) * np.cos(theta) + (0.-dy) * np.sin(theta) + dx
            self.endy = (Npix_x-dx) * np.cos(theta) + (Npix_y-dy) * np.sin(theta) + dx
        else:
            self.startx, self.endx= 0., Npix_x
            self.starty, self.endy= 0., Npix_y

        return

    def rotate_grid(self, theta, offset=None, angle_unit='rad'):
        '''
        theta: float
            The rotation angle
        offset: list of floats
            The grid offset in arcsec
        angle_unit: str
            The astropy unit of theta
        '''

        if angle_unit == 'deg':
            theta = np.deg2rad(theta)
        elif angle_unit != 'rad':
            raise ValueError('`angle_unit` can only be `deg` or `rad`! ' +
                                  'Passed unit of {}'.format(angle_unit))

        if not offset: offset = [0., 0.]

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))

        offset_grid = np.array(
            [self.im_ra - offset[0], self.im_dec - offset[1]]
            )
        translate = np.empty_like(offset_grid)
        translate[0,:] = offset[0]
        translate[1,:] = offset[1]

        rotated_grid = np.dot(R, offset_grid) + translate

        self.im_pos = rotated_grid.T
        self.im_ra, self.im_dec = self.im_pos[0,:], self.im_pos[1,:]

        return

    def cut2buffer(self):
        '''
        Remove objects outside of image (and buffer).
        We must sample points in the buffer zone in the beginning due to
        possible rotations.
        '''

        b = self.im_gs
        in_region = np.where( (self.im_pos[:,0] > b) &\
                              (self.im_pos[:,0] < (self.Npix_x-b)) &\
                              (self.im_pos[:,1] > b) &\
                              (self.im_pos[:,1] < (self.Npix_y-b))
                              )

        self.im_pos = self.im_pos[in_region]
        self.im_ra = self.im_pos[:,0]
        self.im_dec = self.im_pos[:,1]

        # Get all image coordinate pairs
        self.pos = self.wcs.wcs_pix2world(self.im_pos, 1)
        self.ra = self.pos[:,0]
        self.dec = self.pos[:,1]

        return

class RectGrid(Grid):
    def __init__(self, grid_spacing, wcs, Npix_x, Npix_y, pix_scale,
                 rot_angle=None, pos_offset=None, angle_unit='rad'):
        '''
        See Grid
        '''

        super(RectGrid, self).__init__(
            grid_spacing, wcs, Npix_x=Npix_x, Npix_y=Npix_y,
            pix_scale=pix_scale, rot_angle=rot_angle,
            pos_offset=pos_offset, angle_unit=angle_unit
            )

        self._create_grid()

        return

    def _create_grid(self):

        im_gs = self.im_gs

        po = self.pos_offset
        im_po = po / self.pix_scale
        self.im_ra  = np.arange(self.startx, self.endx, im_gs)
        self.im_dec = np.arange(self.starty, self.endy, im_gs)

        # Get all image coordinate pairs
        self.im_pos = np.array(
            np.meshgrid(self.im_ra, self.im_dec)
            ).T.reshape(-1, 2)

        self.im_ra  = self.im_pos[:,0]
        self.im_dec = self.im_pos[:,1]

        if self.rot_angle:
            self.rotate_grid(
                self.rot_angle, angle_unit=self.angle_unit,
                offset=[(self.Npix_x+im_po[0])/2., (self.Npix_y+im_po[1])/2.]
                )

        self.cut2buffer()

        return

class HexGrid(Grid):
    def __init__(self, grid_spacing, wcs, Npix_x, Npix_y, pix_scale,
                 rot_angle=None, pos_offset=None, angle_unit='rad'):
        '''
        See Grid
        '''

        super(HexGrid, self).__init__(
            grid_spacing, wcs, Npix_x=Npix_x, Npix_y=Npix_y,
            pix_scale=pix_scale, rot_angle=rot_angle,
            pos_offset=pos_offset, angle_unit=angle_unit
            )

        self._create_grid()

        return

    def _create_grid(self):

        im_gs = self.im_gs

        po = self.pos_offset
        im_po = [p / self.pix_scale for p in po]
        self.im_pos = HexGrid.calc_hex_coords(
            self.startx, self.starty, self.endx, self.endy, im_gs
            )

        self.im_ra  = self.im_pos[:,0]
        self.im_dec = self.im_pos[:,1]

        if self.rot_angle:
            self.rotate_grid(
                self.rot_angle, angle_unit=self.angle_unit,
                offset=[(self.Npix_x+im_po[0])/2., (self.Npix_y+im_po[1])/2.]
                )

        self.cut2buffer()

        return

    @classmethod
    def calc_hex_coords(cls, startx, starty, endx, endy, radius):
        # Geoemtric factors of given hexagon
        r = radius
        p = r * np.tan(np.pi / 6.) # side length / 2
        h = 4. * p
        dx = 2. * r
        dy = 2. * p

        row = 1

        xs = []
        ys = []

        while startx < endx:
            x = [startx,
                 startx,
                 startx + r,
                 startx + dx,
                 startx + dx,
                 startx + r,
                 startx + r]
            xs.append(x)
            startx += dx

        while starty < endy:
            y = [starty + p,
                 starty + 3*p,
                 starty + h,
                 starty + 3*p,
                 starty + p,
                 starty,
                 starty + dy]
            ys.append(y)
            starty += 2*p
            row += 1

        polygons = [list(zip(x, y)) for x in xs for y in ys] #MEGAN added list()
        hexgrid = cls.polygons2coords(polygons)

        # Some hexagonal elements go beyond boundary; cut these out
        indx = np.where( (hexgrid[:,0] < endx) & (hexgrid[:,1] < endy) )
        return hexgrid[indx]

    @classmethod
    def polygons2coords(HexGrid, p):
        s = np.shape(p)
        L = s[0]*s[1]
        pp = np.array(p).reshape(L,2)
        c = np.vstack({tuple(row) for row in pp})
        # Some of the redundant coordinates are offset by ~1e-10 pixels
        return np.unique(c.round(decimals=6), axis=0)

    def rotate_polygons():
        return

class MixedGrid(BaseGrid):
    def __init__(self, grid_type, N_inj_types, inj_frac=None):
        '''
        grid_type: str
            The name of the grid type
        N_inj_types: int
            The number of unique injection types
        inj_fraction: float or dict
            A mapping of object type names to their fraction of grid points
        '''

        self.grid_type = grid_type
        self.N_inj_types = N_inj_types

        if inj_frac is not None:
            if isinstance(inj_frac, float):
                self.inj_frac = {inj_type : inj_frac}
                self.curr_N_types = 1
            elif isinstance(inj_frac, dict):
                for val in inj_frac.values():
                    if not isinstance(val, float):
                        raise TypeError('Each `inj_frac` entry must be ' +
                                        'a float!')
                self.inj_frac = inj_frac
                self.curr_N_types = len(inj_frac)
            else:
                raise TypeError('`inj_frac` can only be passed as a float ' +
                                'or a dict!')

        self._check_inj_frac()

        self.pos = {}
        self.indx = {}
        self.nobjects = {}

        self.assigned_objects = False

        return

    def _check_inj_frac(self, final=False):
        sum = 0
        for frac in self.inj_frac.values():
            if (frac <= 0) or (frac >= 1):
                raise ValueError('Each injection fraction must be 0<=frac<=1')
            sum += frac

        if len(self.inj_frac) == self.N_inj_types:
            if sum != 1:
                raise ValueError('The sum of injection fractions must equal ' +
                                 '1 after all types have been set!')

        if (final is True) and (len(self.inj_frac) != self.N_inj_types):
            raise ValueError('Cannot continue until all injection ' +
                             'fractions are set!')

        return

    def add_injection(self, inj_type, inj_frac):
        self.inj_frac[inj_type] = inj_frac
        self.curr_N_types += 1

        if self.curr_N_types == self.N_inj_types:
            self._assign_objects()

        return

    def build_grid(self, **kwargs):
        self.grid = build_grid(self.grid_type, **kwargs)

        # Only assign objects if all injection fractions are set
        if self.curr_N_types == self.N_inj_types:
            self._assign_objects()

        return

    def _assign_objects(self):

        if self.curr_N_types != self.N_inj_types:
            raise ValueError('Cannot assign injection objects to grid until the MixedGrid has '
                             'all input types set!')

        self._check_inj_frac(final=True)

        N = len(self.grid.pos)
        Ninj = self.N_inj_types

        if N==0:
            raise ValueError('The constructed grid has zero objects to assign!')

        indx = np.arange(N)

        icount = 0
        for inj_type, inj_frac in self.inj_frac.items():
            icount += 1
            # Always rounds down
            n = int(self.inj_frac[inj_type] * N)
            if icount < Ninj:
                nobjs = n
            else:
                # Grab remaining items
                nobjs = len(indx)
                assert n <= nobjs <= n+Ninj

            self.nobjects[inj_type] = nobjs

            i = np.random.choice(indx, nobjs, replace=False)
            self.indx[inj_type] = i
            self.pos[inj_type] = self.grid.pos[i]

            indx = np.setdiff1d(indx, i)


        assert(np.sum(list(self.nobjects.values())) == N) #MEGAN added list()
        assert(len(indx) == 0)

        self.assigned_objects = True

        return

def build_grid(grid_type, **kwargs):
    '''
    grid_type: str
        The name of the grid type
    kwargs: dict
        All args needed for the called grid constructor
    '''

    if grid_type in GRID_TYPES:
        # User-defined grid construction
        return GRID_TYPES[grid_type](**kwargs)
    else:
        raise ValueError(f'{grid_type} is not a valid grid type!')

GRID_TYPES = {
    'default': HexGrid,
    'rect_grid' : RectGrid,
    'hex_grid' : HexGrid
    }
