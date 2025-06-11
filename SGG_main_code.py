#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:44:49 2025

Code to numerically solve and analyze the coupled PDE system of self-generated chemotaxis.
The theory and preliminary results are published in: https://www.biorxiv.org/content/10.1101/2024.12.19.628881v1

@author: Mehmet Can Ucar
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from matplotlib import rc
rc('text', usetex=True)
rc('font',size=14)
rc('font',family='serif')
rc('axes',labelsize=14)

#%% This function determines the 2nd derivative using finite-differences: 

def second_derivative(f,dx,bound):    
    f_2nd = np.zeros_like(f)
    f_2nd[0] = (2*f[1] - 2*f[0] - 2*dx*bound[0]) / dx**2
    f_2nd[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    f_2nd[-1] = (2*f[-2] - 2*f[-1] + 2*dx*bound[1]) / dx**2
    return f_2nd

#%% Define the 1st derivative via finite differences:

def first_derivative(f,dx,bound):    
    f_1st = np.zeros_like(f)
    f_1st[0] = bound[0]
    f_1st[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    f_1st[-1] = bound[1]
        
    return f_1st

#%% Function to model bounded logarithmic chemotactic sensing:
def logsense(attract, up, low, dx, bc=(0, 0)):

    sensing = np.log((1 + attract / low) / (1 + attract / up))

    # First derivative: centered difference, Neumann BC
    d1 = np.zeros_like(sensing)
    d1[1:-1] = (sensing[2:] - sensing[:-2]) / (2 * dx)
    d1[0] = bc[0]  # Left Neumann BC
    d1[-1] = bc[1]  # Right Neumann BC

    # Second derivative: centered difference
    d2 = np.zeros_like(sensing)
    d2[1:-1] = (sensing[2:] - 2 * sensing[1:-1] + sensing[:-2]) / dx**2

    # Ghost-point-based second derivative at boundaries
    d2[0] = (2 * sensing[1] - 2 * sensing[0] - 2 * dx * bc[0]) / dx**2
    d2[-1] = (2 * sensing[-2] - 2 * sensing[-1] + 2 * dx * bc[1]) / dx**2

    return {
        'fct': sensing,
        'd_fct': d1,
        'd2_fct': d2
    }

#%% Function to define coupled PDE of chemotaxis:

def coupled_diff_adv(grid_spec,params,sensing, system, Kup = 1, Klow = 0.1):
    
    (Lmax,N_grid,dx,N_t,dt,bound1,bound2) = grid_spec    
    (diff1,diff2,diffc,chi1,chi2,decay,influx) = params
    
    x = np.arange(0,Lmax,dx)

    # Initial Cell densities:
    # Determine the initial spatial range from a logistic decay:
    
    if system=='closed':
        f_prev = 1/(1 + np.exp(x - 1)) 
        g_prev = 1/(1 + np.exp(x - 1))
    elif system=='open':
        f_prev = 1/(1 + np.exp(x - 5)) 
        g_prev = 1/(1 + np.exp(x - 5))

    # Or Gaussian distributed initial densities:
 #   f_prev = np.exp(-0.5*(x/1)**2)    
 #   g_prev = np.exp(-0.5*(x/1)**2)
 
    # Initial Attractant density:
    c_prev = np.ones(N_grid)
    
    # Or pre-patterned attractant profiles can be used:
    #linear_ramp = np.linspace(0.1,10,N_grid)
    #gaussian_ramp = 10*np.exp(-(x-N_grid)**2/(30000))
    #c_prev = gaussian_ramp

    f_list = [f_prev]
    g_list = [g_prev]
    c_list = [c_prev]
    
    for t in range(N_t):
        
        f_1st = first_derivative(f_prev, dx, bound1)
        f_2nd = second_derivative(f_prev, dx, bound1)
        
        g_1st = first_derivative(g_prev, dx, bound1)
        g_2nd = second_derivative(g_prev, dx, bound1)
        
        c_1st = first_derivative(c_prev, dx, bound2)
        c_2nd = second_derivative(c_prev, dx, bound2)
        
        # Difference equation for the cell populations:
        # Here we can decide on different forms for the chemokine:
        if sensing == 'abs':  # Absolute gradient sensing
            f_next = f_prev + (diff1*f_2nd - chi1*(f_1st*c_1st+f_prev*c_2nd))*dt
            g_next = g_prev + (diff2*g_2nd - chi2*(g_1st*c_1st+g_prev*c_2nd))*dt
        elif sensing == 'rel': # Relative gradient sensing
            f_next = f_prev + (diff1*f_2nd - chi1*(f_1st*(c_1st/c_prev)+f_prev*(c_2nd/c_prev-(c_1st**2)/(c_prev**2))))*dt
            g_next = g_prev + (diff2*g_2nd - chi2*(g_1st*(c_1st/c_prev)+g_prev*(c_2nd/c_prev-(c_1st**2)/(c_prev**2))))*dt
        elif sensing == 'michaelis':  # Michaelis-Menten with upper sensing threshold:
            f_next = f_prev + (diff1*f_2nd - chi1*(f_1st*(c_1st/(c_prev+Klow))+f_prev*(c_2nd/(c_prev+Klow)-(c_1st**2)/((c_prev+Klow)**2))))*dt
            g_next = g_prev + (diff2*g_2nd - chi2*(g_1st*(c_1st/(c_prev+Klow))+g_prev*(c_2nd/(c_prev+Klow)-(c_1st**2)/((c_prev+Klow)**2))))*dt
        elif sensing == 'log':  # Bounded logarithmic sensing
            sense_dict = logsense(c_prev, Kup, Klow, dx, bound2)
            
            f_next = f_prev + (diff1*f_2nd - chi1*(f_1st*sense_dict['d_fct']+f_prev*sense_dict['d2_fct']))*dt
            g_next = g_prev + (diff2*g_2nd - chi2*(g_1st*sense_dict['d_fct']+g_prev*sense_dict['d2_fct']))*dt
            
        # Difference equation for the chemokine:
        if system=='closed': # Closed system
            c_next = c_prev + (diffc*c_2nd- decay*c_prev*f_prev)*dt
        elif system=='open':  # Open system
            c_next = c_prev + (diffc*c_2nd + 1 - c_prev - decay*c_prev*f_prev)*dt

        # Only save the list at certain time intervals (to reduce dataload)
        if t%100==0:
            f_list.append(f_next)
            g_list.append(g_next)
            c_list.append(c_next)
        
        f_next[0] += dt * influx[0]  # Inject cell mass directly at x = 0
        g_next[0] += dt * influx[1]

        f_prev = f_next
        g_prev = g_next
        c_prev = c_next
        
    d = dict()
    d['consumer'] = np.array(f_list)
    d['sensor'] = np.array(g_list)
    d['chem'] = np.array(c_list)
    
    return d

#%% Initialize and solve the coupled PDE:
# Spatial discretization (traveling waves):
N_grid = 900
Lmax = 900
dx = Lmax/N_grid

# Maximal time and time discretization:
N_t = 1000000
dt = 0.01

# Boundary conditions
# For cells (left and right end of x-axis)
boundary1 = np.array([0,0])
# For chemoattractant (left and right end of x-axis)
boundary2 = [0,0]

grid = (Lmax,N_grid,dx,N_t,dt,boundary1,boundary2)

diff1 = 0.07
diff2 = 0.2
diffc = 1
chi1 = 0.2
chi2 = chi1*1.2
decay = 1
influx = [0.001,0.001]   # Cell influx rate
params = (diff1,diff2,diffc,chi1,chi2,decay, influx)

K_low = 1e-6
solution = coupled_diff_adv(grid,params,'rel','closed', 1, K_low)

sol_load = solution

consume_sol = sol_load['consumer']
sensor_sol = sol_load['sensor']
chem_sol = sol_load['chem']

#%% Plot the profiles

x = np.arange(N_grid)*dx
times =  np.arange(len(consume_sol))[int(10000*dt):][::int(10/dt)]

fig, axs = plt.subplots(1,3, figsize=(14,4))

# Reduce horizontal space between axes
fig.subplots_adjust(hspace=0.1)

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, len(times))]

for j in np.arange(len(times)):
    
    axs[0].plot(x,consume_sol[int(times[j])],color=colors[::-1][j])  
    axs[1].plot(x,sensor_sol[int(times[j])],color=colors[::-1][j])
    axs[2].plot(x,chem_sol[int(times[j])],color=colors[::-1][j])

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
axs[0].grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)
axs[1].grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)
axs[2].grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)

axs[0].set_xlim([-5, Lmax/1.5])
axs[1].set_xlim([-5, Lmax/1.5])
axs[2].set_xlim([-5, Lmax/1.5])

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, labelleft=False)
plt.xlabel(r'Distance $x$',fontsize=16);
#plt.ylabel(r'Density',fontsize=16);

axs[0].tick_params(    
    which='both',labelsize=12) 
axs[1].tick_params(    
    which='both', labelsize=12) 
axs[2].tick_params(    
    which='both', labelsize=12) 

plt.grid(False)

# Add space at the bottom and left side of the exported pdf
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.2)

plt.title(f'$D_c =$ {np.round(diff1,3)}, $D_s =$ {np.round(diff2,3)}, $\chi_c =$ {np.round(chi1,4)}, $\chi_s =$ {np.round(chi2,4)}');

#%% Nondimensional version: Time evolution of densities in separate subfigures

fig, axs = plt.subplots(2,1, sharex=True, figsize=(5,5))

consume_sol = sol_load['consumer']
sensor_sol = sol_load['sensor']
chem_sol = sol_load['chem']

# Reduce horizontal space between axes
fig.subplots_adjust(hspace=0.1)

x = np.arange(N_grid)*dx

times =  np.arange(len(consume_sol))[int(80000*dt):int(1000000*dt)][::int(10/dt)]

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, len(times))]

for j in range(len(times)):
    DC_data = consume_sol[int(times[j])]
    T_data = sensor_sol[int(times[j])]
    
    axs[1].plot(x,T_data,color=colors[::-1][j])  
    axs[0].plot(x,DC_data,color=colors[::-1][j])  
    
    axs[1].set_ylabel(r'Sensor density',fontsize=16);
    axs[0].set_ylabel(r'Consumer density',fontsize=16);

# Axis limits
plt.xlim(10,Lmax/2);

axs[0].set_ylim([0, 0.03])
axs[1].set_ylim([0, 0.05])

axs[0].tick_params(    
    which='both',      
    labelbottom=False,labelsize=14) 
axs[1].tick_params(    
    which='both',labelsize=14) 

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
axs[0].grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)
axs[1].grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'Distance',fontsize=16);

# Add space at the bottom and left side of the exported pdf
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.2)


#%% Determine intensity front positions from the AVERAGED DATASETS
front_DC_pos = []
front_T_pos = []

x = np.arange(N_grid)*dx

for j in range(len(consume_sol)):
        
    # Get the densities at given time point
    DC_mean_data = consume_sol[j][10:]  # Ignore the closest spatial range to the boundary to exclude boundary peaks:
    T_mean_data = sensor_sol[j][10:]
    
    # Determine position of the "front" by looking at when the intensity value reaches approx half max:
    # Interpolate onto new x grid:
    x_interpolation = np.linspace(0, x[-1], N_grid)

    f_interp_T = np.interp(x_interpolation, x[10:], T_mean_data)

    f_interp_DC = np.interp(x_interpolation, x[10:], DC_mean_data)
    
    intens_range_DC = np.nanmax(f_interp_DC)-np.nanmin(f_interp_DC)
    int_threshold_DC = np.nanmax(f_interp_DC) - intens_range_DC/2
    last_mid_density_DC = np.where(f_interp_DC - int_threshold_DC>0)[0][-1]
    
    intens_range_T = np.nanmax(f_interp_T)-np.nanmin(f_interp_T)
    int_threshold_T = np.nanmax(f_interp_T) - intens_range_T/2
    last_mid_density_T = np.where(f_interp_T - int_threshold_T>0)[0][-1]

    front_DC_pos.append(x_interpolation[last_mid_density_DC])
    front_T_pos.append(x_interpolation[last_mid_density_T])
    
front_DC_pos = np.array(front_DC_pos)
front_T_pos = np.array(front_T_pos)


#%%
DC_frontpos_data = front_DC_pos.copy()
T_frontpos_data = front_T_pos.copy()


#%% Plot just front position over time
fig, ax = plt.subplots(figsize=(4,5))
t = np.arange(len(DC_frontpos_data))*dt*100

ax.plot(t,DC_frontpos_data,color='steelblue')
ax.plot(t,T_frontpos_data,color='firebrick')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)

# Axis labels
ax.set_xlabel(r'Time',fontsize=14);
ax.set_ylabel(r'Front speed',fontsize=14);

plt.tight_layout()

#%% Density profiles in the comoving frame
# Chemokine diffusion in um^2 per min:
chem_diffusion = 86*60
alpha = 0.1

# Space rescaling
rescale_x = np.sqrt(chem_diffusion*alpha)

# Time rescaling
start_time = 4500
end_time = start_time+int(150/alpha)
time_step = int(1/alpha)
times = np.arange(start_time,end_time,time_step)

fig, ax = plt.subplots(figsize=(4, 5))

shifted_profiles_T = []
shifted_profiles_DC = []

# Create common comoving x-grid
# CLOSED:
x_common_shifted = np.linspace(-1100, 1300, 100)

x = np.arange(N_grid)*dx

for t in times:
    intensity_vals_T = sensor_sol[int(t)]*1000  # Rescaling for comparison with exp intensity values
    intensity_vals_DC = consume_sol[int(t)]*1000
    
    # Compute shift
    DC_front = int(DC_frontpos_data[t])
    x_shifted = (x - x[np.argmin(np.abs(x-DC_front))])*rescale_x

    # Interpolate onto common grid
    f_interp_T = interp1d(x_shifted, intensity_vals_T, kind='linear', bounds_error=False, fill_value=np.nan)
    y_interp_T = f_interp_T(x_common_shifted)

    f_interp_DC = interp1d(x_shifted, intensity_vals_DC, kind='linear', bounds_error=False, fill_value=np.nan)
    y_interp_DC = f_interp_DC(x_common_shifted)

    shifted_profiles_T.append(y_interp_T)
    shifted_profiles_DC.append(y_interp_DC)

    # Plot each shifted curve
    ax.plot(x_common_shifted, y_interp_DC, color='steelblue', alpha=0.01)
    ax.plot(x_common_shifted, y_interp_T, color='firebrick', alpha=0.01)
    
# Convert list to array and compute mean ignoring NaNs
shifted_array_T = np.vstack(shifted_profiles_T)
mean_profile_T = np.nanmean(shifted_array_T, axis=0)
shifted_array_DC = np.vstack(shifted_profiles_DC)
mean_profile_DC = np.nanmean(shifted_array_DC, axis=0)

# Plot the mean profile on top
ax.plot(x_common_shifted, mean_profile_DC, color='steelblue', linewidth=2, label=r'Consumer')
ax.plot(x_common_shifted, mean_profile_T, color='firebrick', linewidth=2, label=r'Sensor')

ax.set_ylim(top=38)

# Plot formatting
ax.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.3)
ax.set_xlim(x_common_shifted[0], x_common_shifted[-1])
ax.set_xlabel(r'Comoving coordinates [$\mu$m]', fontsize=18)
ax.set_ylabel('Rescaled density [a.u.]', fontsize=18)
ax.legend()
plt.tight_layout()

plt.show()

#%% Define kymograph coordinates:

# Chemokine diffusion in um^2 per min:
chem_diffusion = 86*60
alpha = 0.1

# Space rescaling
rescale_x = np.sqrt(chem_diffusion*alpha)

x = np.arange(N_grid)*dx*rescale_x

# Time rescaling
start_time = 4500
end_time = start_time+int(150/alpha)
time_step = int(1/alpha)
times = np.arange(start_time,end_time,time_step)
x_shift = 3700

dens_data_cons = []
dens_data_sens = []

for j in times:
    for k in np.arange(x.shape[0]):
        
        cons_dens = consume_sol[int(j)]*1000
        cons_dens_new = np.zeros(len(cons_dens))
        sens_dens = sensor_sol[int(j)]*1000
        sens_dens_new = np.zeros(len(sens_dens))
        
        # Omit the initial boundary effects:
        cons_dens_new[2:] = cons_dens[2:]
        dens_data_cons.append([x[k]-x_shift,alpha*(j-times[0]),cons_dens_new[k]])

        sens_dens_new[2:] = sens_dens[2:]
        dens_data_sens.append([x[k]-x_shift,alpha*(j-times[0]),sens_dens_new[k]])

dens_data_cons = np.array(dens_data_cons)
dens_data_sens = np.array(dens_data_sens)

x_values,y_values,z_values = dens_data_cons[:,0],dens_data_cons[:,1],dens_data_cons[:,2]

cmap = plt.get_cmap('inferno')
colors = [cmap(i) for i in np.linspace(0, 1, len(times))]

#CLOSED SYSTEM: 
cons_frontpos_data = DC_frontpos_data*rescale_x-x_shift
sens_frontpos_data = T_frontpos_data*rescale_x-x_shift

#%% Plot kymographs with marked half-max positions FOR THEORY
import matplotlib.tri as tri

triang = tri.Triangulation(x_values, y_values)

fig1, ax1 = plt.subplots(figsize=(8,3.5))

# Add space at the bottom and left side of the exported pdf
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.2)

tcf = ax1.tricontourf(triang, z_values, 20, cmap='inferno')

cbar = fig1.colorbar(tcf)
cbar.ax.tick_params(labelsize=10)
cbar.set_label(r'Consumer density $\rho_c$',size=16)

for j in range(len(colors))[::5]:
    ax1.plot(sens_frontpos_data[times[j]],(times[j]-times[0])*alpha,'*',color='ivory',markersize=6)
    ax1.plot(cons_frontpos_data[times[j]],(times[j]-times[0])*alpha,'x',color='ivory',markersize=6)

    
ax1.plot(sens_frontpos_data[times[1]],(times[1]-times[0])*alpha,'*',color='ivory',markersize=6,label='Sensor front')
ax1.plot(cons_frontpos_data[times[1]],(times[1]-times[0])*alpha,'x',color='ivory',markersize=6,label='Consumer front')

ax1.legend(fontsize=12, loc=4)

# set axes limits and labels
ax1.set_ylim(0,150);

#CLOSED
ax1.set_xlim(200,1600);

#xticks(np.arange(1, 9, step=1));
ax1.set_ylabel(r'Time $t$ [min]',fontsize = 18);
ax1.set_xlabel(r'Distance $[\mu$m]',fontsize = 18);
