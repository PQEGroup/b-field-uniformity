import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, CylinderSegment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
"""
Plots flux stream in xz plane along y=0
magnets: collection of magnets used to determine B field
x_grid_bounds: 1x2 list to define x bounds for square grid, grid_bounds[0] is min bound, grid_bounds[1] is max bound
z_grid_bounds: 1x2 list to define z bounds for square grid, grid_bounds[0] is min bound, grid_bounds[1] is max bound

mag_draw_bounds: vertices to define magnet cross section polygon plot
             mag_draw_bounds[i][0] gives list of x-coords of ith polygon vertices
             mag_draw_bounds[i][1] gives list of z-coords of ith polygon vertices
"""
def make_flux_stream(magnets, x_grid_bounds, z_grid_bounds, mag_draw_bounds):
    ts_x = np.linspace(x_grid_bounds[0], x_grid_bounds[1], 101)
    ts_z = np.linspace(z_grid_bounds[0], z_grid_bounds[1], 101)
    grid = np.array([[(x,0,z) for x in ts_x] for z in ts_z])
    
    #get magnetic field
    B = magnets.getB(grid)
    
    #plot flux stream
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 20*np.sum(np.abs(z_grid_bounds)) / np.sum(np.abs(x_grid_bounds))))
    ax1.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2], density=2,
        color=np.log(np.linalg.norm(B, axis=2)), linewidth=1, cmap='autumn', broken_streamlines=False)
    
    #plot coil cross section
    for poly in mag_draw_bounds:
        ax1.plot(poly[0], poly[1],'k--')

        
"""
Generates plot of Bz strength in specified plane.
"""
def plot_plane_field_strength(magnets, x_grid_bounds, y_grid_bounds, z_elev, plot_nonuniformity = True):
    ts_x = np.linspace(x_grid_bounds[0], x_grid_bounds[1], 101)
    ts_y = np.linspace(y_grid_bounds[0], y_grid_bounds[1], 101)
    grid = np.array([[(x,y,z_elev) for x in ts_x] for y in ts_y])

    #get magnetic field
    mT_to_G = 10
    B = magnets.getB(grid)
    G = B * mT_to_G

    #matplotlib contour plot
    # ax = plt.figure().add_subplot(projection='3d')
    # surf = ax.plot_surface(grid[:,:,0], grid[:,:,1], B[:,:,2], linewidth=0)
    
    Gz = G[:,:,2]
    
    fig_base = make_subplots(rows=1, cols=2,
                             specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                             subplot_titles = [f'Field strength in plane z={z_elev}',
                                               f'Field nonuniformity in plane z={z_elev}'
                                              ],
                             shared_yaxes=False
                            )
    
#     fig1 = go.Figure(data=[go.Surface(x=grid[:,:,0], y=grid[:,:,1], z=Gz)])
#     fig1.update_layout(title=f'Field strength in plane z={z_elev}', autosize=False,
#                       width=500, height=500,
#                       margin=dict(l=65, r=50, b=65, t=90))
#     fig.show()
    fig_base.add_trace(
        go.Surface(x=grid[:,:,0], y=grid[:,:,1], z=Gz),
        row=1, col=1
    )
    
    
    if plot_nonuniformity:
        mid_idx = int(len(ts_x)/2)
        mid_idy = int(len(ts_y)/2)
        G_center = Gz[mid_idx][mid_idy]
        G_non = np.abs((Gz - G_center)/G_center) * 100
        
#         fig2 = go.Figure(data=[go.Surface(x=grid[:,:,0], y=grid[:,:,1], z=G_non)])
#         fig2.update_layout(title=f'Field nonuniformity in plane z={z_elev}', autosize=False,
#                           width=500, height=500,
#                           margin=dict(l=65, r=50, b=65, t=90))
#         fig2.show()
        fig_base.add_trace(
            go.Surface(x=grid[:,:,0], y=grid[:,:,1], z=G_non),
            row=1, col=2
        )
    plot_height = 450
    fig_base.update_layout(
        height=plot_height,
        width=2*plot_height,
    )
    fig_base.update_traces(showscale=False)
    fig_base.show()

"""
Set up axial and transverse sensors and plots the B-field measured
"""
def get_field_on_axes(magnets, axial_sensor_length, transverse_sensor_length, uniformity_bound_t, uniformity_bound_ax, is_logscale=False):
    # define sensor and axis
    senpos_transverse = np.linspace((-transverse_sensor_length/2, 0, 0), (transverse_sensor_length/2, 0, 0), 5000)
    sen_t = magpy.Sensor(position=senpos_transverse)

    senpos_axial = np.linspace((0, 0, -axial_sensor_length/2), (0, 0, axial_sensor_length/2), 5000)
    sen_ax = magpy.Sensor(position=senpos_axial)
    
    # read from transverse and axial sensors
    Bz_t = sen_t.getB(magnets).T[2]
    Bz_ax = sen_ax.getB(magnets).T[2]

    # convert from returned mT readings to Gauss
    mT_to_G = 10
    Gz_t = Bz_t * mT_to_G
    Gz_ax = Bz_ax * mT_to_G
    
    # plotting
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12)
    plt.rcParams.update({'font.size': 12})
    
    fig_width = 10
    fig = plt.figure(figsize=(fig_width, fig_width))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    
    # matplotlib 3d view
#     ax5 = fig.add_subplot(325, projection='3d', elev=24)
#     ax6 = fig.add_subplot(326, projection='3d', elev=24)
#     ax6.view_init(elev=0, azim=0)
    
    ax1.set_ylabel("Gauss")
    ax1.plot(senpos_transverse[:, 0], Gz_t)
    ax1.set_title('Transverse $B_z$ Profile')
    ax1.grid(color='.9', which='major', axis='both', linestyle='-')

    ax2.plot(senpos_axial[:, 2], Gz_ax)
    ax2.set_title('Axial $B_z$ Profile')
    ax2.grid(color='.9')
    
    nonun_ax = get_nonuniformity(Gz_ax, "+z")
    nonun_t = get_nonuniformity(Gz_t, "+x")
    
    uniformity_bounds = 2
    
    xt = senpos_transverse[:, 0]
    # Get indices between uniformity bound range
    indices_xt = np.where(np.logical_and(xt>=uniformity_bound_t[0], xt<=uniformity_bound_t[1]))[0]
    clipped_nonun_t = np.abs(np.take(nonun_t, indices_xt))
    min_nonun_t = np.min(clipped_nonun_t)
    max_nonun_t = np.max(clipped_nonun_t)
    print(f"Min transverse nonuniformity: {min_nonun_t}\nMax transverse nonuniformity:{max_nonun_t}")
    
    if (is_logscale):
        ax3.set_yscale("log")
        ax4.set_yscale("log")
    
    # Get clipped range for plot
    clipped_xt = np.take(xt, indices_xt)
    # Plot range corresponding to these indices
    ax3.plot(clipped_xt, clipped_nonun_t)
    ax3.grid(color='.9', which='major', axis='both', linestyle='-')
    ax3.set_ylim(min_nonun_t - 1e-7, max_nonun_t + 1e-7)
    ax3.set_title('Transverse $B_z$ Non-uniformity Profile')
    
    
    xa = senpos_axial[:, 2]
    # Get indices between uniformity bound range
    indices_xa = np.where(np.logical_and(xa>=uniformity_bound_ax[0], xa<=uniformity_bound_ax[1]))[0]
    clipped_nonun_ax = np.abs(np.take(nonun_ax, indices_xa))
    min_nonun_ax = np.min(clipped_nonun_ax)
    max_nonun_ax = np.max(clipped_nonun_ax)
    print(f"Min axial nonuniformity: {min_nonun_ax}\nMax axial nonuniformity:{max_nonun_ax}")
    
    # Get clipped range for plot
    clipped_xa = np.take(xa, indices_xa)
    # Plot range corresponding to these indices
    ax4.plot(clipped_xa, clipped_nonun_ax)
    ax4.grid(color='.9', which='major', axis='both', linestyle='-')
    ax4.set_ylim(min_nonun_ax - 1e-7, max_nonun_ax + 1e-7)
    ax4.set_title('Axial $B_z$ Non-uniformity Profile')

#     magpy.show(magnets, sen_ax, sen_t, canvas=ax5, style_magnetization_show=True)
#     magpy.show(magnets, sen_ax, sen_t, canvas=ax6, style_magnetization_show=True)
    
    magpy.show(magnets, sen_ax, sen_t, style_magnetization_show=True, backend='plotly')
    sq = 10
    plot_plane_field_strength(magnets, [-sq, sq], [-sq, sq], 0)

    plt.tight_layout()
    plt.show()

# return nonuniformity from single axis sensor about central point
def get_nonuniformity(G_measured, direction):
    maxG = max(G_measured)
    G_center = G_measured.item(int(len(G_measured)/2))
    
    print(f"Maximum B-field along {direction} direction: {round(maxG, 3)} G")
    print(f"Central B-field along {direction} axis: {round(G_center, 3)} G")
    nonuniformity = (G_measured - G_center) / G_center
    return nonuniformity

# create sweep range centered on center value with provided step size and number of samples
def centered_sweep_range(center, steps, dx):
    start = center + int(steps / 2) * dx
    stop = center - int(steps / 2) * dx
    sweep = np.linspace(start=start, stop=stop, num=(steps + 1 if (steps % 2 == 0) else steps))
    return sweep

# make numpy grid
def make_grid(x_grid_bounds, y_grid_bounds, grid_res):
    ts_x = np.linspace(x_grid_bounds[0], x_grid_bounds[1], grid_res)
    ts_y = np.linspace(y_grid_bounds[0], y_grid_bounds[1], grid_res)
    grid = np.array([[(x,y,0) for x in ts_x] for y in ts_y])
    return grid

# returns
# - sum total nonuniformity over grid
# - center b-field
# - average nonuniformity over grid area
def get_grid_nonuniformity(magnets, grid, grid_res):
    mT_to_G = 10
    B = magnets.getB(grid)
    G = B * mT_to_G
    # find magnitude of the b-field at all points over grid
#     Gmag = np.sqrt(G[:,:,0]**2 + G[:,:,1]**2 + G[:,:,2]**2)
    Gmag = G[:,:,2]
    # find magnitude of b-field at center of grid
    mid_id = int(grid_res/2)
    Gcenter = Gmag[mid_id][mid_id]
    # calculate nonuniformity with respect to center b-field
    Gnon = np.abs((Gmag - Gcenter)/Gcenter)
    max_nonuniformity = np.max(Gnon)
    av_nonuniformity = np.sum(Gnon) / grid_res**2
    return Gnon, Gcenter, av_nonuniformity, max_nonuniformity
