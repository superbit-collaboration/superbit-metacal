# Standard imports
import galsim
import warnings

__all__ = ["nfw_lensing"]

def nfw_lensing(nfw_halo, pos, nfw_z_source):
    """     
    - Shears a given NFW halo as the given positon and z of source
    - Usage is borrowed from demo9.py
    - nfw_halo is galsim.NFW() object created in main()
    - pos is position of galaxy in image
    - nfw_z_source is background galaxy redshift
    Args:
        nfw_halo ([type]): NFW halo object
        pos ([type]): position
        nfw_z_source ([type]): redshift of the source

    Returns:
        nfw_shear: shear
        nfw_mu: magnification
    """

    g1,g2 = nfw_halo.getShear(pos=pos, z_s=nfw_z_source )
    nfw_shear = galsim.Shear(g1=g1,g2=g2)
    nfw_mu = nfw_halo.getMagnification(pos=pos, z_s=nfw_z_source )

    if nfw_mu < 0:
        warnings.warn("Warning: mu < 0 means strong lensing! Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        warnings.warn("Warning: mu > 25 means strong lensing! Using mu=25.")
        nfw_mu = 25  

    return nfw_shear, nfw_mu     

   