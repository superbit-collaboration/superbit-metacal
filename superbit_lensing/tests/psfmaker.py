

import psfex
import galsim,galsim.des
import piff
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from astropy.table import Table
import pdb
from starmaker import StarMaker, StampBackground
import treecorr
import glob



def make_output_table(makers=None,prefix=None,outfile='hsm_fit_result.fits'):
    """
    Concatenate arbitrary number of Maker() objects with HSM fits
    into an output FITS table & save to file

    : data :   list of Maker() instances
    : prefix : list of prefixes for column names
    """

    # Bit of sanity checking
    assert type(makers) == list
    assert type(prefix) == list
    assert type(makers[0]) in [PSFMaker,StarMaker]

    mtab = {}
    mtab['x'] = makers[0].x
    mtab['y'] = makers[1].y

    # First, go through and make sub_tables:
    for i,maker in enumerate(makers):
        mtab['_'.join([prefix[i],'hsm_sig'])] = maker.hsm_sig
        mtab['_'.join([prefix[i],'hsm_g1'])] = maker.hsm_g1
        mtab['_'.join([prefix[i],'hsm_g2'])] = maker.hsm_g2
        mtab['_'.join([prefix[i],'fwhm'])] = maker.fwhm

    t = Table(mtab)
    t.write(outfile,format='fits',overwrite=True)

    return t


def make_quiverplot(psf=None,stars=None,outname='quiverplot.png'):
    """
    Take a table or whatever, make quiverplot with it
    Filter out failed fits first!
    """

    fscl = 2.355*psf.pixel_scale # convert sigma in pixels --> FWHM in arcsec

    if (psf==None) or (stars==None):
        print("Can't make quiverplot without star and PSF HSM fits")
        sys.exit()

    wg = (psf.hsm_g1 > -9999) & (stars.hsm_g1 > -9999)

    x = stars.x[wg]; y = stars.y[wg]
    star_g1 = stars.hsm_g1[wg]
    star_g2 = stars.hsm_g2[wg]
    psf_g1  = psf.hsm_g1[wg]
    psf_g2  = psf.hsm_g2[wg]
    psf_sig = psf.hsm_sig[wg] * fscl
    star_sig = stars.hsm_sig[wg] * fscl

    mean_star_g = np.median(np.sqrt(star_g1**2+star_g2**2))
    mean_psf_g  = np.median(np.sqrt(psf_g1**2+psf_g2**2))

    mean_star_sig = np.median(star_sig)
    mean_psf_sig  = np.median(psf_sig)

    gdiff1 = star_g1-psf_g1
    gdiff2 = star_g2-psf_g2
    sig_diff = star_sig - psf_sig

    #norm=colors.Normalize()
    norm = colors.TwoSlopeNorm(np.median(star_sig))

    fig,[ax1,ax2,ax3]=plt.subplots(nrows=1,ncols=3,sharey=True,figsize=[21,6])

    q_star = ax1.quiver(y,x,star_g1,\
        star_g2,star_sig,cmap='cool',\
        units='xy',angles='uv',pivot='mid',headaxislength=0,headwidth=0,\
        headlength=0,norm=norm)
    ax1.set_title('avg star HSM ellip = %.5f fwhm = %.3f' %
        (mean_star_g, mean_star_sig),fontsize=10)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(q_star, cax=cax)

    q_res = ax2.quiver(y,x,psf_g1,\
        psf_g2,psf_sig,cmap='cool', \
        units='xy',angles='uv',pivot='mid',\
        headaxislength=0,headwidth=0,headlength=0,\
        scale=q_star.scale,norm=norm)
    ax2.set_title('avg psf ellip = %.5f fwhm = %.3f' % \
        (mean_psf_g,mean_psf_sig), fontsize=10)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(q_res, cax=cax)

    this_norm = colors.TwoSlopeNorm(np.median(sig_diff))
    q_diff = ax3.quiver(y,x,gdiff1,gdiff2,sig_diff,units='xy',\
        angles='uv',pivot='mid',headaxislength=0,headwidth=0,headlength=0,\
        cmap='cool',scale=q_star.scale,norm=this_norm)
    ax3.set_title('avg psf g_hsm resid = %.5f avg psf sig_hsm diff = %.3f' % \
        (np.median(np.sqrt(gdiff1**2+gdiff2**2)),\
        np.median(sig_diff)),fontsize=10)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(q_diff, cax=cax)

    plt.tight_layout()
    plt.savefig(outname)

    return

def make_resid_plot(psf=None,stars=None,outname='star_psf_resid.png',vb=False):
    """
    make figures of average stars, psf renderings,
    and residuals between the two

    :avg_psf : should be an instance of PSFMaker()
    :avg_star: should be an instance of StarMaker()
    """

    avg_stars = np.nanmean(stars.star_stamps,axis=0)
    avg_psfim = np.nanmean(psf.stamps,axis=0)
    avg_resid = np.nanmean(psf.resids,axis=0)

    if vb==True:
        print("avg_star total flux = %.3f" % np.sum(avg_stars))
        print("avg_psf total flux = %.3f" % np.sum(avg_psfim))

    # Calculate average sizes to display in image
    wg = (psf.hsm_sig > -9999.) & (stars.hsm_sig > -9999.)
    psf_fwhm  = np.nanmean(psf.fwhm[wg])
    psf_sigma = np.nanmean(psf.hsm_sig[wg])
    star_fwhm = np.nanmean(stars.fwhm[wg])
    star_sigma = np.nanmean(stars.hsm_sig[wg])

    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(21,7))

    norm = colors.TwoSlopeNorm(0)
    f1 = axs[0].imshow(avg_stars,norm=colors.TwoSlopeNorm(0),\
        cmap=plt.cm.seismic_r)
    axs[0].set_title('avg star HSM sigma = %.5f\ngs.calculateFWHM() = %.5f'\
        % (star_sigma,star_fwhm))
    axs[0].axvline((avg_stars.shape[0]-1)*0.5,color='black')
    axs[0].axhline((avg_stars.shape[1]-1)*0.5,color='black')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(f1, cax=cax)

    f2 = axs[1].imshow(avg_psfim,norm=colors.TwoSlopeNorm(0),\
        cmap=plt.cm.seismic_r)
    axs[1].set_title('avg PSF HSM sigma = %.5f\ngs.calculateFWHM() = %.5f' \
        % (psf_sigma,psf_fwhm))
    axs[1].axvline((avg_stars.shape[0]-1)*0.5,color='black')
    axs[1].axhline((avg_stars.shape[1]-1)*0.5,color='black')

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right",size="5%", pad=0.05)
    plt.colorbar(f2, cax=cax)

    f3 = axs[2].imshow(avg_resid,norm=colors.TwoSlopeNorm(0), cmap=plt.cm.seismic_r)
    axs[2].set_title('avg star resid sum=%.4f\nmean=%.4e std=%.4e' \
        % (np.nansum(avg_resid),np.nanmean(avg_resid),np.nanstd(avg_resid)),fontsize=12)
    axs[2].axvline((avg_stars.shape[0]-1)*0.5,color='black')
    axs[2].axhline((avg_stars.shape[1]-1)*0.5,color='black')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(f3, cax=cax)

    fig.suptitle(outname.split('.')[0])
    plt.tight_layout()
    plt.savefig(outname)

    return 0


class PSFMaker:
    """
    Do HSM fits to psf vignettes
    Save output to file
    Could even make a few correlation functions if you like

    Note: psf_type is
    Possible improvements: make more general?
    """

    def __init__(self,psf_file=None,psf_type='epsfex',noisefree=False):
        """
        psf_obj is the file name of psf,
        or alternatively an instance of it

        psf_type is the method to use to render the PSF.
        Should be one of the following:
            - 'epsfex:' Erin Sheldon psfex Python package
            - 'gpsfex:' Use galsim.des PSFEx method
            - 'piff:  ' Use PIFF rendering
        """

        self.psf = psf_file
        self.psf_type = psf_type
        self.noisefree = noisefree
        self.psf_size = 19
        self.stamps = []
        self.resids = []

        self.pixel_scale = 0.144
        self.sky_level = 0.0
        self.sky_std = 0.0

        self.x = []
        self.y = []
        self.hsm_sig = []
        self.hsm_g1 = []
        self.hsm_g2 = []
        self.fwhm = []

        if psf_type not in ['epsfex','gpsfex','piff']:
            print("psf_type is the method to use to render the PSF,")
            print("Should be one of the following:")
            print("     - 'epsfex': Erin Sheldon psfex Python package")
            print("     - 'gpsfex': Use galsim.des PSFEx method")
            print("     - 'piff'  : Use PIFF rendering")


    def render_psf(self,x=None,y=None,flux=None,psf_type='epsfex',vb=False):
        """
        Method to decide which rendering method to call, then call it.
        Appends PSF cutouts to self

        would be cool to try to make an assertion error pop out
        """

        if self.psf_type=='epsfex':
            if vb==True: print("rendering epsfex psf")
            psf_im = self._make_pexim(x_pos=x,y_pos=y,flux=flux,vb=vb)

        elif self.psf_type=='gpsfex':
            if vb==True: print("rendering psfex_des psf")
            psf_im = self._make_gpsf(x_pos=x,y_pos=y,flux=flux,vb=vb)

        elif self.psf_type=='piff':
            if vb==True: print("rendering PIFF psf")
            psf_im = self._make_piff(x_pos=x,y_pos=y,flux=flux,vb=vb)

        else:
            allowed=['piff','epsfex','gpsfex']
            print("PSF not one of ['piff','epsfex','gpsfex']")

        return psf_im

    def _make_pexim(self,x_pos,y_pos,flux=None,vb=False):
        """
        Generate a esheldon.psfex rendering at position x_pos,y_pos
        incorporating sky noise & star flux
        """

        pix_scale = self.pixel_scale
        im_wcs = galsim.PixelScale(self.pixel_scale)

        if flux == None:
            print("using unit flux")
            star_flux = 1
        else:
            star_flux=flux
            if vb==True: print("using flux=%.2f" % flux)

        this_pexim = self.psf.get_rec(y_pos,x_pos)
        pexim_rs = (this_pexim)/np.sum(this_pexim)*flux

        if self.noisefree==False:
            noise = np.random.normal(loc=self.sky_level,scale=self.sky_std,size=this_pexim.shape)
            pexim_rs+=noise

        return pexim_rs


    def _make_gpsf(self,x_pos,y_pos,flux=None,vb=False):
        """
        Generate a gs.des.des_psfex()rendering at position x_pos,y_pos
        incorporating sky noise & star flux
        """

        pix_scale = self.pixel_scale
        im_wcs = galsim.PixelScale(pix_scale)

        if flux == None:
            star_flux = 1
        else:
            star_flux=flux
            if vb==True:print("using flux=%.2f" % flux)

        psfex_des = self.psf

        this_pos = galsim.PositionD(x_pos,y_pos)
        this_psf_des = psfex_des.getPSF(this_pos)
        gpsf_im = this_psf_des.drawImage(method='no_pixel',
                    nx=self.psf_size, ny=self.psf_size, \
                    scale=pix_scale, use_true_center=True).array

        gpsf_im_rs = (gpsf_im)/np.sum(gpsf_im)*flux

        if self.noisefree==False:
            noise = np.random.normal(loc=self.sky_level,scale=self.sky_std,size=gpsf_im_rs.shape)
            gpsf_im_rs+=noise

        return gpsf_im_rs

    def _make_piff(self,x_pos,y_pos,flux=None,vb=False):
        """
        Render a PIFF psf model and add noise
        piff.PSF.draw returns a GalSim image of PSF, so can treat
        it in the same way as gpsf/des_psfex
        docs:
        https://rmjarvis.github.io/Piff/_build/html/psf.html?highlight=draw#

        Note piff.draw() outputs complex WCS and so calculateFWHM() will fail
        unless array part is redrawn into a gs.Image() with a pixel scale WCS.
        """

        if flux == None:
            star_flux = 1
        else:
            star_flux=flux
            if vb==True:print("using flux=%.2f" % flux)

        piff_psf = self.psf
        piff_im = piff_psf.draw(x=x_pos,y=y_pos,\
                    stamp_size=self.psf_size).array

        piff_im_rs = piff_im/np.sum(piff_im)*flux

        if self.noisefree==False:
            noise = np.random.normal(loc=self.sky_level,scale=self.sky_std,size=piff_im_rs.shape)
            piff_im_rs+=noise

        return piff_im_rs


    def _do_hsm_fits(self,verbose=False):
        """
        It would be great to somehow make this inheritable by any class

        Might be strange to take galsim.Image() output of des_psfex and
        piff.draw, take array part of galsim.Image and redraw with a pixel scale
        wcs since the GSObjects already *had* a WCS. However, it can be shown
        that the result of HSM fit is the same in both cases.
        """

        for i,stamp in enumerate(self.stamps):
            try:
                gs_object = galsim.Image(stamp, wcs=galsim.PixelScale(self.pixel_scale))
                HSM_fit=gs_object.FindAdaptiveMom()

                self.hsm_sig.append(HSM_fit.moments_sigma)
                self.hsm_g1.append(HSM_fit.observed_shape.g1)
                self.hsm_g2.append(HSM_fit.observed_shape.g2)
                self.fwhm.append(gs_object.calculateFWHM())

            except:
                print("HSM fit for stamp #%d failed, skipping" % i)
                self.hsm_sig.append(-9999)
                self.hsm_g1.append(-9999)
                self.hsm_g2.append(-9999)
                self.fwhm.append(gs_object.calculateFWHM())

        self.hsm_sig = np.array(self.hsm_sig)
        self.hsm_g1  = np.array(self.hsm_g1)
        self.hsm_g2  = np.array(self.hsm_g2)
        self.fwhm    = np.array(self.fwhm)

        return

    def run_rho_stats(self,stars=None,rparams=None,vb=False,outdir='./'):
        """
        Method to obtain rho-statistics for current PSF fit & save plots
        Requires StarMaker to be provided through 'stars' parameter

        default rparams={'min_sep':100,'max_sep':3000,'nbins':60}
        """

        # First do calculations
        rho1,rho2,rho3,rho4,rho5 = self._run_rho_stats(stars,rparams=rparams,
            vb=vb,outdir=outdir)

        # Then make plots
        outname=os.path.join(outdir,'_'.join([str(self.psf_type),'rho_stats']))
        self._plot_rho_stats(rho1,rho2,rho3,rho4,rho5,outname=outname)

        print("Finished rho stat computation & plotting")

        return


    def _run_rho_stats(self,stars=None,rparams=None,outdir=None,vb=False):

        if rparams==None:
            rparams = {'min_sep':150,'max_sep':4000,'nbins':15}
        min_sep = rparams['min_sep']
        max_sep = rparams['max_sep']
        nbins = rparams['nbins']

        # Define quantities to be used
        wg = (self.hsm_g1 > -9999) & (stars.hsm_g1 > -9999)
        star_g1 = stars.hsm_g1[wg]
        star_g2 = stars.hsm_g2[wg]
        psf_g1 = self.hsm_g1[wg]
        psf_g2 = self.hsm_g2[wg]

        dg1 = star_g1 - psf_g1
        dg2 = star_g2 - psf_g2

        T  = (2*(self.hsm_sig[wg]))**2
        Tpsf = (2*(stars.hsm_sig[wg]))**2
        dT = T-Tpsf; dTT = dT/T

        # Stars & size-residual-scaled stars
        starcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=star_g1,g2=star_g2)
        rs_starcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=star_g1*dTT,g2=star_g2*dTT)

        # PSFs & size-residual-scaled PSFs
        psfcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=psf_g1,g2=psf_g2)
        rs_psfcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=psf_g1*dTT,g2=psf_g2*dTT)

        # PSF Resids
        psf_resid_cat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=dg1,g2=dg2)


        # rho-1: psf_ellip residual autocorrelation
        rho1 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho1.process(psf_resid_cat)
        rho1.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_1.txt'])))
        if vb==True:
            print('bin_size = %.6f'%rho1.bin_size)
            print('mean rho1 = %.4e median = %.4e std = %.4e' % \
                (np.mean(rho1.xip),np.median(rho1.xip),\
                np.std(rho1.xip)))

        # rho-2: psf_ellip x psf_ellip residual correlation
        rho2 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho2.process(starcat,psf_resid_cat)
        rho2.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_2.txt'])))
        if vb==True:
            print('mean rho2 = %.4e median = %.4e std = %.4e' % \
                (np.mean(rho2.xip),np.median(rho2.xip),\
                np.std(rho2.xip)))

        # My *guess* at rho-3: psf_ellip x psf_size residual correlation
        rho3 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho3.process(rs_starcat)
        rho3.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_3.txt'])))
        if vb==True:
            print('mean rho3 = %.4e median = %.4e std = %.4e' % \
                (np.mean(rho3.xip),np.median(rho3.xip),
                np.std(rho3.xip)))

        # My *guess* at rho-4: psf ellip resid x (psf ellip *size resid)
        rho4 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho4.process(psf_resid_cat,rs_starcat)
        rho4.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_4.txt'])))
        if vb==True:
            print('mean rho4 = %.4e median = %.4e std = %.4e' % \
            (np.mean(rho4.xip),np.median(rho4.xip), np.std(rho4.xip)))

        # My *guess* at rho-4: psf ellip  x (psf ellip *size resid)
        rho5 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho5.process(starcat,rs_starcat)
        rho5.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_5.txt'])))

        if vb==True:
            print('mean rho5 = %.4e median = %.4e std = %.4e' % \
            (np.mean(rho5.xip),np.median(rho5.xip), np.std(rho5.xip)))

        return rho1,rho2,rho3,rho4,rho5

    def _plot_rho_stats(self,rho1,rho2,rho3,rho4,rho5,outname=None):

        ##
        ## rho1 correlation: dg x dg
        ##
        fig,axes=plt.subplots(nrows=2,ncols=1,figsize=[12,8])
        r = np.exp(rho1.meanlogr) * .144 / 60 # from pixels --> arcminutes
        xip = np.abs(rho1.xip)
        sig = np.sqrt(rho1.varxip)

        lp1 = axes[0].plot(r, xip, color='green',marker='.',ls='-',lw=1,label=r'$\rho_1(\theta)$')
        axes[0].plot(r, -xip, color='green', marker='.',ls=':',lw=1)
        axes[0].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='green', lw=0.3, ls='')
        axes[0].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='green', lw=0.3, ls='')
        axes[0].errorbar(-r, xip, yerr=sig, color='green')

        axes[0].set_xscale('log')
        axes[0].set_yscale('log', nonpositive='clip')
        axes[0].set_xlabel(r'$\theta$ (arcmin)')
        axes[0].set_ylabel(r'$\xi_+(\theta)$')


        ##
        ## rho3 correlation: g x dg
        ##
        r = np.exp(rho3.meanlogr)*.144/60
        xip = np.abs(rho3.xip)
        sig = np.sqrt(rho3.varxip)

        lp3 = axes[0].plot(r, xip, color='rebeccapurple',marker='.',ls='-',lw=1,label=r'$\rho_3(\theta)$')
        axes[0].plot(r, -xip, color='rebeccapurple', marker='.',ls=':',lw=1)
        axes[0].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='rebeccapurple', lw=0.3, ls='')
        axes[0].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='rebeccapurple', lw=0.3, ls='')
        axes[0].errorbar(-r, xip, yerr=sig, color='rebeccapurple')

        ##
        ## rho4 correlation
        ##
        r = np.exp(rho4.meanlogr)* .144 / 60
        xip = rho4.xip
        sig = np.sqrt(rho4.varxip)

        lp4 = axes[0].plot(r, xip, color='cornflowerblue',marker='.',ls='-',lw=1,label=r'$\rho_4(\theta)$')
        axes[0].plot(r, -xip, color='cornflowerblue', marker='.',ls=':',lw=1)
        axes[0].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='cornflowerblue', lw=0.3, ls='')
        axes[0].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='cornflowerblue', lw=0.3, ls='')
        axes[0].errorbar(-r, xip, yerr=sig, color='cornflowerblue')

        #axes[0].legend([lp1,lp3,lp4])
        plt.legend()
        #axes[0].set_ylim([2e-7,5e-3])

        ##
        ## rho 2 correlation: g x dg
        ##
        r = np.exp(rho2.meanlogr)*.144/60
        xip = np.abs(rho2.xip)
        sig = np.sqrt(rho2.varxip)

        lp2 = axes[1].plot(r, xip, color='magenta',marker='.',ls='-',lw=1,label=r'$\rho_2(\theta)$')
        axes[1].plot(r, -xip, color='magenta', marker='.',ls=':',lw=1)
        axes[1].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='magenta', lw=0.3, ls='')
        axes[1].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='magenta', lw=0.3, ls='')
        axes[1].errorbar(-r, xip, yerr=sig, color='magenta')

        axes[1].set_xlabel(r'$\theta$ (arcmin)')
        axes[1].set_ylabel(r'$\xi_(\theta)$')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log', nonpositive='clip')

        ##
        ## rho5 correlation
        ##
        r = np.exp(rho5.meanlogr)* .144 / 60.
        xip = rho5.xip
        sig = np.sqrt(rho5.varxip)

        lp5 = axes[1].plot(r, xip, color='darkblue',marker='.',ls='-',lw=1.,label=r'$\rho_5(\theta)$')
        axes[1].plot(r, -xip, color='darkblue', marker='.',ls=':',lw=1.)
        axes[1].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='darkblue', lw=0.3, ls='')
        axes[1].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='darkblue', lw=0.3, ls='')
        axes[1].errorbar(-r, xip, yerr=sig, color='darkblue')

        #axes[1].legend([lp2,lp5])
        #axes[1].set_ylim([2e-7,5e-3])
        plt.legend()

        fig.tight_layout()
        fig.savefig(outname)


        return

    def _write_to_file(self,outdir=None,vb=False):
        """
        This is currently more of a cleanup operation, but could
        be expanded to write HSM fits individually as well

        """

        pass
        return


    def run_all(self,stars=None,vb=False,outdir='./psf_diagnostics'):
        """
        starmaker is expected to be an instance of the StarMaker class
        Possible improvements: allow user to supply just X,Y?
        Allow a freestanding bg value?
        """

        if stars == None:
            print("StarMaker() instance not supplied, exiting")
            sys.exit()

        self.sky_level = stars.sky_level
        self.sky_std = stars.sky_std
        self.x = stars.x
        self.y = stars.y

        # Render PSF, take residual against stars
        for i in range(len(stars.x)):
            xpos = stars.x[i]; ypos = stars.y[i]
            flux = stars.star_flux[i]
            star_stamp=stars.star_stamps[i]

            pim = self.render_psf(x=xpos,y=ypos,flux=flux)
            self.stamps.append(pim)
            self.resids.append(pim-star_stamp)

        # Do HSM fitting
        self._do_hsm_fits(verbose=vb)

        # Make output quiverplot
        outname = os.path.join(outdir,'_'.join([self.psf_type,'quiverplot.png']))
        make_quiverplot(psf=self,stars=stars,outname=outname)

        # Make output star-psf residuals plot
        outname = os.path.join(outdir,'_'.join([self.psf_type,'star_psf_resid.png']))
        make_resid_plot(psf=self,stars=stars,outname=outname,vb=vb)

        # Compute & make output rho-statistics figures
        rparams={'min_sep':200,'max_sep':5000,'nbins':50}
        self.run_rho_stats(stars=stars,rparams=rparams,vb=vb,outdir=outdir)

        #self._write_to_file(outdir=outdir,vb=vb)

        print("finished running PSFMaker()")
        return

def make_rho_rhatios(file_path='./',rho_files=None):
    """
    Make rho ratio plots for different PSF types.
    Note thatthe master_psf_diagnostics.py file nomenclature
    is assumed: `[psf_type]_rho_[1-5].txt`
    psf_type=['epsfex','gpsfex','piff'],
    """

    import glob

    for i in range(1,6):
        """
        if rho_files is not None:
            rho_files = list(rho_files)
        else:

            "lmao there has to be a better way to do this"
            file_root = ''.join(['*_rho_',str(i),'.txt'])
            rho_files = glob.glob(file_root)

        for rho in rho_files:
        This should be some sort of os.path.exists thing
        """
        try:
            pexn=os.path.join(file_path,''.join(['epsfex_rho_',str(i),'.txt']))
            pex=Table.read(pexn,format='ascii',header_start=1)
        except:
            pex=None
        try:
            gpsfn = os.path.join(file_path,''.join(['gpsfex_rho_',str(i),'.txt']))
            gpsf=Table.read(gpsfn,format='ascii',header_start=1)
        except:
            gpsf=None
        try:
            piffn = os.path.join(file_path,''.join(['piff_rho_',str(i),'.txt']))
            piff=Table.read(piffn,format='ascii',header_start=1)
        except:
            piff=None

        savename = os.path.join(file_path,'rho_{}_comparisons.png'.format(i))

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=[12,6])
        ax.set_xscale('log')
        ax.set_yscale('log', nonpositive='clip')
        ax.set_ylabel(r'$\rho_{}(\theta)$'.format(i))
        plt.xlabel(r'$\theta$ (arcsec)')

        if pex is not None:

            r = pex['meanr'] * .144 / 60 # from pixels --> arcminutes
            pex_xip = np.abs(pex['xip'])
            pex_sig = pex['sigma_xip']

            plt.plot(r, pex_xip, color='blue',marker='.',ls='-',label=r'pex')
            plt.plot(r, -pex_xip, color='blue',  marker='.',ls=':',lw=1)
            plt.errorbar(r[pex_xip>0], pex_xip[pex_xip>0], yerr=pex_sig[pex_xip>0], color='blue', lw=0.3, ls='')
            plt.errorbar(r[pex_xip<0], -pex_xip[pex_xip<0], yerr=pex_sig[pex_xip<0], color='blue', lw=0.3, ls='')
            lp = plt.errorbar(-r, pex_xip, yerr=pex_sig, color='blue')

        if gpsf is not None:

            ### GPSF, rho3
            r = gpsf['meanr'] * .144 / 60 # from pixels --> arcminutes
            gpsf_xip = np.abs(gpsf['xip'])
            gpsf_sig = gpsf['sigma_xip']

            plt.plot(r, gpsf_xip, color='rebeccapurple',marker='.',ls='-',label=r'gpsf')
            plt.plot(r, -gpsf_xip, color='rebeccapurple',  marker='.',ls=':',lw=1)
            plt.errorbar(r[gpsf_xip>0], gpsf_xip[gpsf_xip>0], yerr=gpsf_sig[gpsf_xip>0], color='rebeccapurple', lw=0.3, ls='')
            plt.errorbar(r[gpsf_xip<0], -gpsf_xip[gpsf_xip<0], yerr=gpsf_sig[gpsf_xip<0], color='rebeccapurple', lw=0.3, ls='')
            lp2 = plt.errorbar(-r, gpsf_xip, yerr=gpsf_sig, color='rebeccapurple')

        if piff is not None:
            ### PIFF

            r = piff['meanr'] * .144 / 60 # from pixels --> arcminutes
            piff_xip = np.abs(piff['xip'])
            piff_sig = piff['sigma_xip']

            plt.plot(r, piff_xip, color='salmon',marker='.',ls='-',label=r'piff')
            plt.plot(r, -piff_xip, color='salmon',  marker='.',ls=':',lw=1)
            plt.errorbar(r[piff_xip>0], piff_xip[piff_xip>0], yerr=piff_sig[piff_xip>0], color='salmon', lw=0.3, ls='')
            plt.errorbar(r[piff_xip<0], -piff_xip[piff_xip<0], yerr=piff_sig[piff_xip<0], color='salmon', lw=0.3, ls='')
            lp3 = plt.errorbar(-r, piff_xip, yerr=piff_sig, color='salmon')

        plt.legend()
        plt.savefig(savename)
        plt.close(fig)

    print("plots done")

    return
