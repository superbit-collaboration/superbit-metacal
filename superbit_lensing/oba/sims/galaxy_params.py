import numpy as np
import pandas as pd
from astropy import units as u
import random
import matplotlib.pyplot as plt

def n_arcmin(catalog,
             iterations=100,
             plot=False):
    n = []
    for i in range(iterations):
        box = generate_box()
        ra_min_box = box[0]
        ra_max_box = box[1]
        dec_min_box = box[2]
        dec_max_box = box[3]

        df_temp = catalog[catalog['ALPHA_J2000']>=ra_min_box]
        df_temp = df_temp[df_temp['ALPHA_J2000']<=ra_max_box]
        df_temp = df_temp[df_temp['DELTA_J2000']>=dec_min_box]
        df_temp = df_temp[df_temp['DELTA_J2000']<=dec_max_box]
        n.append(len(df_temp))
        del df_temp
    if plot:
        plt.hist(n, histtype='step')
        plt.xlabel("$n$ arcmin$^{-2}$")
        plt.ylabel('Number of iterations')
        
    return n

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_shape_params(z_cosmos15,
                     mag_814w_cosmos15,
                     cosmos_fit2010_df,
                     z_bin_range=0.035,
                     mag_bin_range=0.035):
            
    # case 1
    if (z_cosmos15 < 5 and (mag_814w_cosmos15 < 25.2
                           and mag_814w_cosmos15 > 18)):
        filt_cosmos_fit2010 = cosmos_fit2010_df[cosmos_fit2010_df["c10_zphot"]>=(z_cosmos15-z_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010[filt_cosmos_fit2010["c10_zphot"]<=(z_cosmos15+z_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010.reset_index(drop=True)
        
        idx = find_nearest(filt_cosmos_fit2010['c10_mag_auto'], 
                                    mag_814w_cosmos15)
        
        case = 1

        c10_IDENT = filt_cosmos_fit2010['c10_IDENT'][idx]
        c10_mag_auto = filt_cosmos_fit2010['c10_mag_auto'][idx]
        c10_zphot = filt_cosmos_fit2010['c10_zphot'][idx]
        c10_flux_radius = filt_cosmos_fit2010['c10_flux_radius'][idx]
        
        c10_sersic_fit_intensity = filt_cosmos_fit2010['c10_sersic_fit_intensity'][idx]
        c10_sersic_fit_hlr = filt_cosmos_fit2010['c10_sersic_fit_hlr'][idx]
        c10_sersic_fit_n = filt_cosmos_fit2010['c10_sersic_fit_n'][idx]
        c10_sersic_fit_q = filt_cosmos_fit2010['c10_sersic_fit_q'][idx]
        c10_sersic_fit_boxiness = filt_cosmos_fit2010['c10_sersic_fit_boxiness'][idx]
        c10_sersic_fit_x0 = filt_cosmos_fit2010['c10_sersic_fit_x0'][idx]
        c10_sersic_fit_y0 = filt_cosmos_fit2010['c10_sersic_fit_y0'][idx]
        c10_sersic_fit_phi = filt_cosmos_fit2010['c10_sersic_fit_phi'][idx]
        
        c10_bulge_fit_disk_intensity = filt_cosmos_fit2010['c10_bulge_fit_disk_intensity'][idx]
        c10_bulge_fit_disk_hlr = filt_cosmos_fit2010['c10_bulge_fit_disk_hlr'][idx]
        c10_bulge_fit_disk_n = filt_cosmos_fit2010['c10_bulge_fit_disk_n'][idx]
        c10_bulge_fit_disk_q = filt_cosmos_fit2010['c10_bulge_fit_disk_q'][idx]
        c10_bulge_fit_disk_boxiness = filt_cosmos_fit2010['c10_bulge_fit_disk_boxiness'][idx]
        c10_bulge_fit_disk_x0 = filt_cosmos_fit2010['c10_bulge_fit_disk_x0'][idx]
        c10_bulge_fit_disk_y0 = filt_cosmos_fit2010['c10_bulge_fit_disk_y0'][idx]
        c10_bulge_fit_disk_phi = filt_cosmos_fit2010['c10_bulge_fit_disk_phi'][idx]
        
        c10_bulge_fit_bulge_intensity = filt_cosmos_fit2010['c10_bulge_fit_bulge_intensity'][idx]
        c10_bulge_fit_bulge_hlr = filt_cosmos_fit2010['c10_bulge_fit_bulge_hlr'][idx]
        c10_bulge_fit_bulge_n = filt_cosmos_fit2010['c10_bulge_fit_bulge_n'][idx]
        c10_bulge_fit_bulge_q = filt_cosmos_fit2010['c10_bulge_fit_bulge_q'][idx]
        c10_bulge_fit_bulge_boxiness = filt_cosmos_fit2010['c10_bulge_fit_bulge_boxiness'][idx]
        c10_bulge_fit_bulge_x0 = filt_cosmos_fit2010['c10_bulge_fit_bulge_x0'][idx]
        c10_bulge_fit_bulge_y0 = filt_cosmos_fit2010['c10_bulge_fit_bulge_y0'][idx]
        c10_bulge_fit_bulge_phi = filt_cosmos_fit2010['c10_bulge_fit_bulge_phi'][idx]

        c10_fitstatus_0 = filt_cosmos_fit2010['c10_fitstatus_0'][idx]
        c10_fitstatus_1 = filt_cosmos_fit2010['c10_fitstatus_1'][idx]
        c10_fitstatus_2 = filt_cosmos_fit2010['c10_fitstatus_2'][idx]
        c10_fitstatus_3 = filt_cosmos_fit2010['c10_fitstatus_3'][idx]
        c10_fitstatus_4 = filt_cosmos_fit2010['c10_fitstatus_4'][idx]

        c10_fit_mad_s = filt_cosmos_fit2010['c10_fit_mad_s'][idx]
        c10_fit_mad_b = filt_cosmos_fit2010['c10_fit_mad_b'][idx]
        c10_fit_dvc_btt = filt_cosmos_fit2010['c10_fit_dvc_btt'][idx]
        
        c10_use_bulgefit = filt_cosmos_fit2010['c10_use_bulgefit'][idx]
        c10_viable_sersic = filt_cosmos_fit2010['c10_viable_sersic'][idx]

        c10_hlr_sersic = filt_cosmos_fit2010['c10_hlr_sersic'][idx]
        c10_hlr_bulge = filt_cosmos_fit2010['c10_hlr_bulge'][idx]
        c10_hlr_disk = filt_cosmos_fit2010['c10_hlr_disk'][idx]
        
        c10_flux_sersic = filt_cosmos_fit2010['c10_flux_sersic'][idx]
        c10_flux_bulge = filt_cosmos_fit2010['c10_flux_bulge'][idx]
        c10_flux_disk = filt_cosmos_fit2010['c10_flux_disk'][idx]
        c10_flux_3 = filt_cosmos_fit2010['c10_flux_3'][idx]

    # case 2
    elif (z_cosmos15 < 5 and (mag_814w_cosmos15 >= 25.2
                         and mag_814w_cosmos15 < 30)):
        filt_cosmos_fit2010 = cosmos_fit2010_df[cosmos_fit2010_df["c10_zphot"]>=(z_cosmos15-z_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010[filt_cosmos_fit2010["c10_zphot"]<=(z_cosmos15+z_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010.reset_index(drop=True)
        
        idx = find_nearest(filt_cosmos_fit2010['c10_zphot'], 
                           z_cosmos15)
        
        case = 2
        
        c10_IDENT = filt_cosmos_fit2010['c10_IDENT'][idx]
        c10_mag_auto = filt_cosmos_fit2010['c10_mag_auto'][idx]
        c10_zphot = filt_cosmos_fit2010['c10_zphot'][idx]
        c10_flux_radius = filt_cosmos_fit2010['c10_flux_radius'][idx]
        
        c10_sersic_fit_intensity = 0
        c10_sersic_fit_hlr = filt_cosmos_fit2010['c10_sersic_fit_hlr'][idx]
        c10_sersic_fit_n = np.random.uniform(0, 4)
        c10_sersic_fit_q = np.random.uniform(0.1, 1)
        c10_sersic_fit_boxiness = 0
        c10_sersic_fit_x0 = 0
        c10_sersic_fit_y0 = 0
        c10_sersic_fit_phi = np.random.uniform(-2, 2)
        
        c10_bulge_fit_disk_intensity = 0
        c10_bulge_fit_disk_hlr = 0
        c10_bulge_fit_disk_n = 0
        c10_bulge_fit_disk_q = 0
        c10_bulge_fit_disk_boxiness = 0
        c10_bulge_fit_disk_x0 = 0
        c10_bulge_fit_disk_y0 = 0
        c10_bulge_fit_disk_phi = 0
        
        c10_bulge_fit_bulge_intensity = 0
        c10_bulge_fit_bulge_hlr = 0
        c10_bulge_fit_bulge_n = 0
        c10_bulge_fit_bulge_q = 0
        c10_bulge_fit_bulge_boxiness = 0
        c10_bulge_fit_bulge_x0 = 0
        c10_bulge_fit_bulge_y0 = 0
        c10_bulge_fit_bulge_phi = 0

        c10_fitstatus_0 = 0
        c10_fitstatus_1 = 0
        c10_fitstatus_2 = 0
        c10_fitstatus_3 = 0
        c10_fitstatus_4 = 0

        c10_fit_mad_s = 0
        c10_fit_mad_b = 0
        c10_fit_dvc_btt = 0
        
        c10_use_bulgefit = 0
        c10_viable_sersic = 0

        c10_hlr_sersic = 0
        c10_hlr_bulge = 0
        c10_hlr_disk = 0
        
        c10_flux_sersic = 0
        c10_flux_bulge = 0
        c10_flux_disk = 0
        c10_flux_3 = 0
                 
    # case 3
    elif (z_cosmos15 > 5 and (mag_814w_cosmos15 < 25.2
                             and mag_814w_cosmos15 > 18)):
        filt_cosmos_fit2010 = cosmos_fit2010_df[cosmos_fit2010_df["c10_mag_auto"]>=(mag_814w_cosmos15-mag_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010[filt_cosmos_fit2010["c10_mag_auto"]<=(mag_814w_cosmos15+mag_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010.reset_index(drop=True)

        idx = find_nearest(filt_cosmos_fit2010['c10_mag_auto'], 
                           mag_814w_cosmos15)

        case = 3
        
        c10_IDENT = filt_cosmos_fit2010['c10_IDENT'][idx]
        c10_mag_auto = filt_cosmos_fit2010['c10_mag_auto'][idx]
        c10_zphot = filt_cosmos_fit2010['c10_zphot'][idx]
        c10_flux_radius = filt_cosmos_fit2010['c10_flux_radius'][idx]
        
        c10_sersic_fit_intensity = 0
        c10_sersic_fit_hlr = np.random.uniform(5, 20)
        c10_sersic_fit_n = filt_cosmos_fit2010['c10_sersic_fit_n'][idx]
        c10_sersic_fit_q = filt_cosmos_fit2010['c10_sersic_fit_q'][idx]
        c10_sersic_fit_boxiness = 0
        c10_sersic_fit_x0 = 0
        c10_sersic_fit_y0 = 0
        c10_sersic_fit_phi = filt_cosmos_fit2010['c10_sersic_fit_phi'][idx]
        
        c10_bulge_fit_disk_intensity = 0
        c10_bulge_fit_disk_hlr = 0
        c10_bulge_fit_disk_n = 0
        c10_bulge_fit_disk_q = 0
        c10_bulge_fit_disk_boxiness = 0
        c10_bulge_fit_disk_x0 = 0
        c10_bulge_fit_disk_y0 = 0
        c10_bulge_fit_disk_phi = 0
        
        c10_bulge_fit_bulge_intensity = 0
        c10_bulge_fit_bulge_hlr = 0
        c10_bulge_fit_bulge_n = 0
        c10_bulge_fit_bulge_q = 0
        c10_bulge_fit_bulge_boxiness = 0
        c10_bulge_fit_bulge_x0 = 0
        c10_bulge_fit_bulge_y0 = 0
        c10_bulge_fit_bulge_phi = 0

        c10_fitstatus_0 = 0
        c10_fitstatus_1 = 0
        c10_fitstatus_2 = 0
        c10_fitstatus_3 = 0
        c10_fitstatus_4 = 0

        c10_fit_mad_s = 0
        c10_fit_mad_b = 0
        c10_fit_dvc_btt = 0
        
        c10_use_bulgefit = 0
        c10_viable_sersic = 0

        c10_hlr_sersic = 0
        c10_hlr_bulge = 0
        c10_hlr_disk = 0
        
        c10_flux_sersic = 0
        c10_flux_bulge = 0
        c10_flux_disk = 0
        c10_flux_3 = 0

    # case 4
    elif (z_cosmos15 > 5 and (mag_814w_cosmos15 >= 25.2
                             and mag_814w_cosmos15 < 30)):
        case = 4
        
        c10_IDENT = 0
        c10_mag_auto = 0
        c10_zphot = 0
        c10_flux_radius = 0
        
        c10_sersic_fit_intensity = 0
        c10_sersic_fit_hlr = np.random.uniform(5, 20)
        c10_sersic_fit_n = np.random.uniform(0, 4)
        c10_sersic_fit_q = np.random.uniform(0.1, 1)
        c10_sersic_fit_boxiness = 0
        c10_sersic_fit_x0 = 0
        c10_sersic_fit_y0 = 0
        c10_sersic_fit_phi = np.random.uniform(-2, 2)
        
        c10_bulge_fit_disk_intensity = 0
        c10_bulge_fit_disk_hlr = 0
        c10_bulge_fit_disk_n = 0
        c10_bulge_fit_disk_q = 0
        c10_bulge_fit_disk_boxiness = 0
        c10_bulge_fit_disk_x0 = 0
        c10_bulge_fit_disk_y0 = 0
        c10_bulge_fit_disk_phi = 0
        
        c10_bulge_fit_bulge_intensity = 0
        c10_bulge_fit_bulge_hlr = 0
        c10_bulge_fit_bulge_n = 0
        c10_bulge_fit_bulge_q = 0
        c10_bulge_fit_bulge_boxiness = 0
        c10_bulge_fit_bulge_x0 = 0
        c10_bulge_fit_bulge_y0 = 0
        c10_bulge_fit_bulge_phi = 0

        c10_fitstatus_0 = 0
        c10_fitstatus_1 = 0
        c10_fitstatus_2 = 0
        c10_fitstatus_3 = 0
        c10_fitstatus_4 = 0

        c10_fit_mad_s = 0
        c10_fit_mad_b = 0
        c10_fit_dvc_btt = 0
        
        c10_use_bulgefit = 0
        c10_viable_sersic = 0

        c10_hlr_sersic = 0
        c10_hlr_bulge = 0
        c10_hlr_disk = 0
        
        c10_flux_sersic = 0
        c10_flux_bulge = 0
        c10_flux_disk = 0
        c10_flux_3 = 0

    # case 5
    elif (z_cosmos15 <= 5 
          and (mag_814w_cosmos15 <= 18 or mag_814w_cosmos15 >= 30)): 
        filt_cosmos_fit2010 = cosmos_fit2010_df[cosmos_fit2010_df["c10_zphot"]>=(z_cosmos15-z_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010[filt_cosmos_fit2010["c10_zphot"]<=(z_cosmos15+z_bin_range)]
        filt_cosmos_fit2010 = filt_cosmos_fit2010.reset_index(drop=True)
 
        idx = find_nearest(filt_cosmos_fit2010['c10_zphot'], 
                           z_cosmos15)

        case = 5
        
        c10_IDENT = filt_cosmos_fit2010['c10_IDENT'][idx]
        c10_mag_auto = filt_cosmos_fit2010['c10_mag_auto'][idx]
        c10_zphot = filt_cosmos_fit2010['c10_zphot'][idx]
        c10_flux_radius = filt_cosmos_fit2010['c10_flux_radius'][idx]
        
        c10_sersic_fit_intensity = 0
        c10_sersic_fit_hlr = filt_cosmos_fit2010['c10_sersic_fit_hlr'][idx]
        c10_sersic_fit_n = np.random.uniform(0, 4)
        c10_sersic_fit_q = np.random.uniform(0.1, 1)
        c10_sersic_fit_boxiness = 0
        c10_sersic_fit_x0 = 0
        c10_sersic_fit_y0 = 0
        c10_sersic_fit_phi = np.random.uniform(-2, 2)
        
        c10_bulge_fit_disk_intensity = 0
        c10_bulge_fit_disk_hlr = 0
        c10_bulge_fit_disk_n = 0
        c10_bulge_fit_disk_q = 0
        c10_bulge_fit_disk_boxiness = 0
        c10_bulge_fit_disk_x0 = 0
        c10_bulge_fit_disk_y0 = 0
        c10_bulge_fit_disk_phi = 0
        
        c10_bulge_fit_bulge_intensity = 0
        c10_bulge_fit_bulge_hlr = 0
        c10_bulge_fit_bulge_n = 0
        c10_bulge_fit_bulge_q = 0
        c10_bulge_fit_bulge_boxiness = 0
        c10_bulge_fit_bulge_x0 = 0
        c10_bulge_fit_bulge_y0 = 0
        c10_bulge_fit_bulge_phi = 0

        c10_fitstatus_0 = 0
        c10_fitstatus_1 = 0
        c10_fitstatus_2 = 0
        c10_fitstatus_3 = 0
        c10_fitstatus_4 = 0

        c10_fit_mad_s = 0
        c10_fit_mad_b = 0
        c10_fit_dvc_btt = 0
        
        c10_use_bulgefit = 0
        c10_viable_sersic = 0

        c10_hlr_sersic = 0
        c10_hlr_bulge = 0
        c10_hlr_disk = 0
        
        c10_flux_sersic = 0
        c10_flux_bulge = 0
        c10_flux_disk = 0
        c10_flux_3 = 0
        
    # case 6
    elif (z_cosmos15 > 5 
          and (mag_814w_cosmos15 <= 18 or mag_814w_cosmos15 >= 30)):
        
        case = 6
        
        c10_IDENT = 0
        c10_mag_auto = 0
        c10_zphot = 0
        c10_flux_radius = 0
        
        c10_sersic_fit_intensity = 0
        c10_sersic_fit_hlr = np.random.uniform(5, 20)
        c10_sersic_fit_n = np.random.uniform(0, 4)
        c10_sersic_fit_q = np.random.uniform(0.1, 1)
        c10_sersic_fit_boxiness = 0
        c10_sersic_fit_x0 = 0
        c10_sersic_fit_y0 = 0
        c10_sersic_fit_phi = np.random.uniform(-2, 2)
        
        c10_bulge_fit_disk_intensity = 0
        c10_bulge_fit_disk_hlr = 0
        c10_bulge_fit_disk_n = 0
        c10_bulge_fit_disk_q = 0
        c10_bulge_fit_disk_boxiness = 0
        c10_bulge_fit_disk_x0 = 0
        c10_bulge_fit_disk_y0 = 0
        c10_bulge_fit_disk_phi = 0
        
        c10_bulge_fit_bulge_intensity = 0
        c10_bulge_fit_bulge_hlr = 0
        c10_bulge_fit_bulge_n = 0
        c10_bulge_fit_bulge_q = 0
        c10_bulge_fit_bulge_boxiness = 0
        c10_bulge_fit_bulge_x0 = 0
        c10_bulge_fit_bulge_y0 = 0
        c10_bulge_fit_bulge_phi = 0

        c10_fitstatus_0 = 0
        c10_fitstatus_1 = 0
        c10_fitstatus_2 = 0
        c10_fitstatus_3 = 0
        c10_fitstatus_4 = 0

        c10_fit_mad_s = 0
        c10_fit_mad_b = 0
        c10_fit_dvc_btt = 0
        
        c10_use_bulgefit = 0
        c10_viable_sersic = 0

        c10_hlr_sersic = 0
        c10_hlr_bulge = 0
        c10_hlr_disk = 0
        
        c10_flux_sersic = 0
        c10_flux_bulge = 0
        c10_flux_disk = 0
        c10_flux_3 = 0
 
    return (case,
            c10_IDENT,
            c10_mag_auto,
            c10_zphot,
            c10_flux_radius,
            c10_sersic_fit_intensity,
            c10_sersic_fit_hlr,
            c10_sersic_fit_n,
            c10_sersic_fit_q,
            c10_sersic_fit_boxiness,
            c10_sersic_fit_x0,
            c10_sersic_fit_y0,
            c10_sersic_fit_phi,
            c10_bulge_fit_disk_intensity,
            c10_bulge_fit_disk_hlr,
            c10_bulge_fit_disk_n,
            c10_bulge_fit_disk_q,
            c10_bulge_fit_disk_boxiness,
            c10_bulge_fit_disk_x0,
            c10_bulge_fit_disk_y0,
            c10_bulge_fit_disk_phi,
            c10_bulge_fit_bulge_intensity,
            c10_bulge_fit_bulge_hlr,
            c10_bulge_fit_bulge_n,
            c10_bulge_fit_bulge_q,
            c10_bulge_fit_bulge_boxiness,
            c10_bulge_fit_bulge_x0,
            c10_bulge_fit_bulge_y0,
            c10_bulge_fit_bulge_phi,
            c10_fitstatus_0,
            c10_fitstatus_1,
            c10_fitstatus_2,
            c10_fitstatus_3,
            c10_fitstatus_4,
            c10_fit_mad_s,
            c10_fit_mad_b,
            c10_fit_dvc_btt,
            c10_use_bulgefit,
            c10_viable_sersic,
            c10_hlr_sersic,
            c10_hlr_bulge,
            c10_hlr_disk,
            c10_flux_sersic,
            c10_flux_bulge,
            c10_flux_disk,
            c10_flux_3)

def generate_box(ra_min = (149.4*u.deg - 1*u.arcmin).to(u.deg),
                 ra_max = (150.8*u.deg - 1*u.arcmin).to(u.deg),
                 dec_min = (1.9*u.deg - 1*u.arcmin).to(u.deg),
                 dec_max = (2.8*u.deg - 1*u.arcmin).to(u.deg),
                 box_side_ra = 1*u.arcmin,
                 box_side_dec = 1*u.arcmin):
    # randomly generate box given ra, dec min and max and box side
    ra_rand = random.uniform(ra_min, ra_max)
    dec_rand = random.uniform(dec_min, dec_max)

    center_point = (ra_rand, dec_rand)

    ra_min_box = (center_point[0] - box_side_ra/2).to(u.deg)
    ra_max_box = (center_point[0] + box_side_ra/2).to(u.deg)
    
    dec_min_box = (center_point[1] - box_side_dec/2).to(u.deg)
    dec_max_box = (center_point[1] + box_side_dec/2).to(u.deg)
    
    return (ra_min_box.to(u.deg), 
            ra_max_box.to(u.deg),
            dec_min_box.to(u.deg),
            dec_max_box.to(u.deg))

def galaxy_params_box(catalog,
                      cosmos_fit2010):
    # generate random 1 arcmin^2 box in COSMOS field
    box = generate_box()
    ra_min_box = box[0]
    ra_max_box = box[1]
    dec_min_box = box[2]
    dec_max_box = box[3]
    
    # get COSMOS catalog within the box bounds
    df_temp = catalog[catalog['ALPHA_J2000']>=ra_min_box]
    df_temp = df_temp[df_temp['ALPHA_J2000']<=ra_max_box]
    df_temp = df_temp[df_temp['DELTA_J2000']>=dec_min_box]
    df_temp = df_temp[df_temp['DELTA_J2000']<=dec_max_box]
    
    # create new dataframe to store
    df_return = pd.DataFrame()
    df_return['RA'] = df_temp['ALPHA_J2000'] - ra_min_box
    df_return['DEC'] = df_temp['DELTA_J2000'] - dec_min_box
    df_return['NUMBER'] = df_temp['NUMBER'] 
    df_return['FLUX_814W'] = df_temp['FLUX_814W'] 
    df_return['PHOTOZ'] = df_temp['PHOTOZ'] 
    
    del df_temp
    
    df_return = df_return.reset_index(drop=True)
    
    # fill in galaxy shape params
    (hlr_cosmos10, 
    n_sersic_cosmos10, 
    q_cosmos10, 
    phi_cosmos10,
    z_cosmos10,
    mag_fit) = ([] for i in range(6))
    
    for i in range(len(df_return)):
        z_cosmos15 = df_return['PHOTOZ'][i]
        mag_814w_cosmos15 = df_return['FLUX_814W'][i]
        
        gal_param_dict = galaxy_parameters(z_cosmos15=z_cosmos15,
                                          mag_814w_cosmos15=mag_814w_cosmos15,
                                          cosmos_fit2010=cosmos_fit2010)
        
        mag_fit.append(gal_param_dict['mag_814w_cosmos10'])
        hlr_cosmos10.append(gal_param_dict['hlr_cosmos10'])
        n_sersic_cosmos10.append(gal_param_dict['n_sersic_cosmos10'])
        q_cosmos10.append(gal_param_dict['q_cosmos10'])
        phi_cosmos10.append(gal_param_dict['phi_cosmos10'])
        z_cosmos10.append(gal_param_dict['z_cosmos10'])
    
    df_return['z_cosmos10'] = z_cosmos10
    df_return['mag_814w_cosmos10'] = mag_fit
    df_return['hlr_cosmos10'] = hlr_cosmos10
    df_return['n_sersic_cosmos10'] = n_sersic_cosmos10
    df_return['q_cosmos10'] = q_cosmos10
    df_return['phi_cosmos10'] = phi_cosmos10
    
    return df_return

def convert_fits_to_df_cosmos2010(cosmos_fit2010):
    c10_IDENT = cosmos_fit2010[1].data['IDENT']
    c10_mag_auto = cosmos_fit2010[1].data['mag_auto']
    c10_flux_radius = cosmos_fit2010[1].data['flux_radius']
    c10_zphot = cosmos_fit2010[1].data['zphot']
    
    # SERSICFIT
    c10_sersic_fit_intensity = cosmos_fit2010[1].data['sersicfit'][:,0]
    c10_sersic_fit_hlr = cosmos_fit2010[1].data['sersicfit'][:,1]
    c10_sersic_fit_n = cosmos_fit2010[1].data['sersicfit'][:,2]
    c10_sersic_fit_q = cosmos_fit2010[1].data['sersicfit'][:,3]
    c10_sersic_fit_boxiness = cosmos_fit2010[1].data['sersicfit'][:,4]
    c10_sersic_fit_x0 = cosmos_fit2010[1].data['sersicfit'][:,5]
    c10_sersic_fit_y0 = cosmos_fit2010[1].data['sersicfit'][:,6]
    c10_sersic_fit_phi = cosmos_fit2010[1].data['sersicfit'][:,7]

    # BULGEFIT
    ## disk fit params
    c10_bulge_fit_disk_intensity = cosmos_fit2010[1].data['bulgefit'][:,0]
    c10_bulge_fit_disk_hlr = cosmos_fit2010[1].data['bulgefit'][:,1]
    c10_bulge_fit_disk_n = cosmos_fit2010[1].data['bulgefit'][:,2]
    c10_bulge_fit_disk_q = cosmos_fit2010[1].data['bulgefit'][:,3]
    c10_bulge_fit_disk_boxiness = cosmos_fit2010[1].data['bulgefit'][:,4]
    c10_bulge_fit_disk_x0 = cosmos_fit2010[1].data['bulgefit'][:,5]
    c10_bulge_fit_disk_y0 = cosmos_fit2010[1].data['bulgefit'][:,6]
    c10_bulge_fit_disk_phi = cosmos_fit2010[1].data['bulgefit'][:,7]

    ## bulge fit params
    c10_bulge_fit_bulge_intensity = cosmos_fit2010[1].data['bulgefit'][:,8]
    c10_bulge_fit_bulge_hlr = cosmos_fit2010[1].data['bulgefit'][:,9]
    c10_bulge_fit_bulge_n = cosmos_fit2010[1].data['bulgefit'][:,10]
    c10_bulge_fit_bulge_q = cosmos_fit2010[1].data['bulgefit'][:,11]
    c10_bulge_fit_bulge_boxiness = cosmos_fit2010[1].data['bulgefit'][:,12]
    c10_bulge_fit_bulge_x0 = cosmos_fit2010[1].data['bulgefit'][:,13]
    c10_bulge_fit_bulge_y0 = cosmos_fit2010[1].data['bulgefit'][:,14]
    c10_bulge_fit_bulge_phi = cosmos_fit2010[1].data['bulgefit'][:,15]

    # FIT_STATUS
    c10_fitstatus_0 = cosmos_fit2010[1].data['fit_status'][:,0]
    c10_fitstatus_1 = cosmos_fit2010[1].data['fit_status'][:,1]
    c10_fitstatus_2 = cosmos_fit2010[1].data['fit_status'][:,2]
    c10_fitstatus_3 = cosmos_fit2010[1].data['fit_status'][:,3]
    c10_fitstatus_4 = cosmos_fit2010[1].data['fit_status'][:,4]

    # FIT_MAD_S
    c10_fit_mad_s = cosmos_fit2010[1].data['fit_mad_s']

    # FIT_MAD_B
    c10_fit_mad_b = cosmos_fit2010[1].data['fit_mad_b']

    # FIT_DVC_BTT
    c10_fit_dvc_btt = cosmos_fit2010[1].data['fit_dvc_btt']

    # USE_BULGE_FIT
    c10_use_bulgefit = cosmos_fit2010[1].data['use_bulgefit']
  
    # VIABLE_SERSIC
    c10_viable_sersic = cosmos_fit2010[1].data['viable_sersic']

    # HLR
    c10_hlr_sersic = cosmos_fit2010[1].data['hlr'][:,0]
    c10_hlr_bulge = cosmos_fit2010[1].data['hlr'][:,1]
    c10_hlr_disk = cosmos_fit2010[1].data['hlr'][:,2]

    # FLUX
    c10_flux_sersic = cosmos_fit2010[1].data['flux'][:,0]
    c10_flux_bulge = cosmos_fit2010[1].data['flux'][:,1]
    c10_flux_disk = cosmos_fit2010[1].data['flux'][:,2]
    c10_flux_3 = cosmos_fit2010[1].data['flux'][:,3]

    
    # append the columns into pandas dataframe
    cosmos_fit2010_df = pd.DataFrame()

    cosmos_fit2010_df['c10_IDENT'] = c10_IDENT.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_mag_auto'] = c10_mag_auto.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_zphot'] = c10_zphot.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_flux_radius'] = c10_flux_radius.byteswap().newbyteorder()
    
    cosmos_fit2010_df['c10_sersic_fit_intensity'] = c10_sersic_fit_intensity.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_hlr'] = c10_sersic_fit_hlr.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_n'] = c10_sersic_fit_n.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_q'] = c10_sersic_fit_q.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_boxiness'] = c10_sersic_fit_boxiness.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_x0'] = c10_sersic_fit_x0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_y0'] = c10_sersic_fit_y0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_sersic_fit_phi'] = c10_sersic_fit_phi.byteswap().newbyteorder()
    
    cosmos_fit2010_df['c10_bulge_fit_disk_intensity'] = c10_bulge_fit_disk_intensity.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_hlr'] = c10_bulge_fit_disk_hlr.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_n'] = c10_bulge_fit_disk_n.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_q'] = c10_bulge_fit_disk_q.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_boxiness'] = c10_bulge_fit_disk_boxiness.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_x0'] = c10_bulge_fit_disk_x0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_y0'] = c10_bulge_fit_disk_y0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_disk_phi'] = c10_bulge_fit_disk_phi.byteswap().newbyteorder()
    
    cosmos_fit2010_df['c10_bulge_fit_bulge_intensity'] = c10_bulge_fit_bulge_intensity.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_hlr'] = c10_bulge_fit_bulge_hlr.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_n'] = c10_bulge_fit_bulge_n.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_q'] = c10_bulge_fit_bulge_q.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_boxiness'] = c10_bulge_fit_bulge_boxiness.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_x0'] = c10_bulge_fit_bulge_x0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_y0'] = c10_bulge_fit_bulge_y0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_bulge_fit_bulge_phi'] = c10_bulge_fit_bulge_phi.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_fitstatus_0'] = c10_fitstatus_0.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_fitstatus_1'] = c10_fitstatus_1.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_fitstatus_2'] = c10_fitstatus_2.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_fitstatus_3'] = c10_fitstatus_3.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_fitstatus_4'] = c10_fitstatus_4.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_fit_mad_s'] = c10_fit_mad_s.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_fit_mad_b'] = c10_fit_mad_b.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_fit_dvc_btt'] = c10_fit_dvc_btt.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_use_bulgefit'] = c10_use_bulgefit.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_viable_sersic'] = c10_viable_sersic.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_hlr_sersic'] = c10_hlr_sersic.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_hlr_bulge'] = c10_hlr_bulge.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_hlr_disk'] = c10_hlr_disk.byteswap().newbyteorder()

    cosmos_fit2010_df['c10_flux_sersic'] = c10_flux_sersic.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_flux_bulge'] = c10_flux_bulge.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_flux_disk'] = c10_flux_disk.byteswap().newbyteorder()
    cosmos_fit2010_df['c10_flux_3'] = c10_flux_3.byteswap().newbyteorder()

    return cosmos_fit2010_df

def cosmos_subsample(cosmos2015,
                     n_gal_sample,
                     ra_min=149.4*u.deg,
                     ra_max=150.8*u.deg,
                     dec_min=1.9*u.deg,
                     dec_max=2.8*u.deg):
    filt = cosmos2015[cosmos2015['ALPHA_J2000']>=ra_min.value]
    filt = filt[filt['ALPHA_J2000']<=ra_max.value]
    filt = filt[filt['DELTA_J2000']>=dec_min.value]
    filt = filt[filt['DELTA_J2000']<=dec_max.value]
    filt = filt.reset_index(drop=True)
    filt = filt.sample(n=n_gal_sample)
    filt = filt.reset_index(drop=True)
    return filt