import numpy as np
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, CylinderSegment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

"""
Br must be given in mT
so 1.09 T is 1.09e3 mT!!!!!!
"""


# define geometry with cylinders along circumference
# used in Ruster and Malinowski
def cyl_ring(n_magnets, br, c_diameter, c_height, st_rad, dist, show=False):
    assert br > 1e2
    magnets = magpy.Collection()
    theta_range = np.linspace(0, 2*np.pi, n_magnets)
    for theta in theta_range:
        magnets = magnets.add(Cylinder(magnetization=br,
                                     dimension = (c_diameter, c_height),
                                     position = (st_rad * np.cos(theta), st_rad * np.sin(theta), dist)))
    if show:
        magpy.show(magnets, style_magnetization_show=True, backend='plotly')
    return magnets

# define halbach cylinder geometry
def halbach_cylinder(Br, c_d, c_h, D, n, alternate=False):
    assert Br > 1e2
    magnets = magpy.Collection()
    if alternate:
        n = 2*n
    for i in range(n):
        theta_i = np.radians(720/n * i)
        b_rem = (Br * np.cos(theta_i), Br * np.sin(theta_i), 0)
        pos_ang = np.radians(360/n * i)
        pos = (D/2 * np.cos(pos_ang), D/2 * np.sin(pos_ang), 0)
        src_i = Cylinder(magnetization=b_rem, position=pos, dimension=(c_d, c_h))
        if (i % 2 == 1):
            magnets.add(src_i)
        if (not alternate and i % 2 == 0):
            magnets.add(src_i)
            
    magnets.rotate_from_angax(90, 'x')
    magnets.rotate_from_angax(-90, 'y')
#     magnets.rotate_from_angax(-90, 'z')
    user_defined_style = {
        'show': True,
        "size": 0.1,
        'color': {
            'transition': 0,
            'mode': 'tricolor',
            'middle': 'white',
            'north': 'magenta',
            'south': 'turquoise',
        },
        "mode": "arrow+color",
    }
    magpy.defaults.display.style.magnet.magnetization = user_defined_style
#     magpy.show(magnets, style_magnetization_show=True, backend='plotly')
    return magnets

# parameters
# ring_specs: a 2D array containing dimensions to specify each ring
# each element of dims is [Br, mag_dir, innerrad, width, thickness, dist, mirror_z]
# Br given in mT
# mag_dir is a tuple representing a normalized vector specifying magnetization direction
# mirror is a boolean to create a mirrored ring pair about the z=0 axis
def n_rings(ring_specs, show=False):
    magnets = magpy.Collection()
    for ring_spec in ring_specs:
        Br, mag_dir, mirror_z, innerrad, width, thickness, dist = ring_spec
        magnetization = tuple(Br * axis for axis in mag_dir)
        ring = CylinderSegment(magnetization=magnetization,
                              dimension=(innerrad, innerrad + width, thickness,0, 360),
                              position=(0, 0, dist)
                              )
        magnets.add(ring)
        if mirror_z:
            ring2 = CylinderSegment(magnetization=magnetization,
                              dimension=(innerrad, innerrad + width, thickness,0, 360),
                              position=(0, 0, -dist)
                              )
            magnets.add(ring2)
    if show:
        magpy.show(magnets, style_magnetization_show=True, backend='plotly')
    return magnets

def four_rings(Br, innerrad1, innerrad2, width, thickness, dist1, dist2, show=False):
    assert Br > 1e2
    magnets = magpy.Collection()
    ring2_top = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad2, innerrad2 + width, thickness, 0, 360),
                               position = (0, 0, +dist2))
    ring1_top = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad1, innerrad1 + width, thickness, 0, 360),
                               position = (0, 0, +dist1))
    ring1_bot = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad1, innerrad1 + width, thickness, 0, 360),
                               position = (0, 0, -dist1))
    ring2_bot = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad2, innerrad2 + width, thickness, 0, 360),
                               position = (0, 0, -dist2))
    magnets.add(ring2_top).add(ring2_bot).add(ring1_top).add(ring1_bot)
    if show:
        magpy.show(magnets, backend='plotly')
    return magnets

def three_rings(Br, innerrad1, innerrad2, width, thickness, dist, show=False):
    assert Br > 1e2
    magnets = magpy.Collection()
    ring2_top = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad2, innerrad2 + width, thickness, 0, 360),
                               position = (0, 0, +dist))
    ring1 = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad1, innerrad1 + width, thickness, 0, 360),
                               position = (0, 0, 0))
    ring2_bot = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad2, innerrad2 + width, thickness, 0, 360),
                               position = (0, 0, -dist))
    magnets.add(ring2_top).add(ring2_bot).add(ring1)
    if show:
        magpy.show(magnets, backend='plotly')
    return magnets

def two_rings(Br, innerrad, width, thickness, dist, show=False):
    assert Br > 1e2
    magnets = magpy.Collection()
    ring_top = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad, innerrad + width, thickness, 0, 360),
                               position = (0, 0, +dist))
    ring_bot = CylinderSegment(magnetization=(0, 0, Br),
                               dimension=(innerrad, innerrad + width, thickness, 0, 360),
                               position = (0, 0, -dist))
    magnets.add(ring_top).add(ring_bot)
    if show:
        magpy.show(magnets, backend='plotly')
    return magnets

def noisy_rings(mean_Br, sd_Br, ring_set_position, N_seg, show=False):
    assert mean_Br > 1e2
    magnets = magpy.Collection()
    angle = 360/N_seg
    for pos in ring_set_position:
        ring_top = magpy.Collection()
        ring_bot = magpy.Collection()
        innerrad, width, thickness, dist = pos
        for i in range(N_seg):
            Br_top = np.random.default_rng().normal(mean_Br, sd_Br)
            ring_top += CylinderSegment(magnetization=(0, 0, Br_top),
                                       dimension=(innerrad, innerrad + width, thickness, i*angle, (i+1)*angle),
                                       position = (0, 0, +dist))
            Br_bot = np.random.default_rng().normal(mean_Br, sd_Br)
            ring_bot += CylinderSegment(magnetization=(0, 0, Br_bot),
                                       dimension=(innerrad, innerrad + width, thickness, i*angle, (i+1)*angle),
                                       position = (0, 0, -dist))
            
        magnets.add(ring_top).add(ring_bot)
#     .add(ring_bot)
#     user_defined_style = {
#         'show': True,
#         "size": 0.1,
#         'color': {
#             'transition': 0,
#             'mode': 'tricolor',
#             'middle': 'white',
#             'north': 'magenta',
#             'south': 'turquoise',
#         },
#         "mode": "arrow+color",
#     }
#     magpy.defaults.display.style.magnet.magnetization = user_defined_style
    if show:
        magpy.show(magnets, backend='plotly')
    return magnets

# define halbach cylinder geometry with noise in angle
def noisy_halbach_cylinder(mean_Br, sd_Br, c_d, c_h, D, n, mean_cyl_deg, sd_cyl_deg, mean_pos_deg, sd_pos_deg, alternate=False):
    assert mean_Br > 1e2
    magnets = magpy.Collection()
    if alternate:
        n = 2*n
    for i in range(n):
        d_theta = np.random.default_rng().normal(mean_cyl_deg, sd_cyl_deg)
        d_pos_ang = np.random.default_rng().normal(mean_pos_deg, sd_pos_deg)
        
        Br = np.random.default_rng().normal(mean_Br, sd_Br)
        
        theta_i = np.radians(720/n * i + d_theta)
        b_rem = (Br * np.cos(theta_i), Br * np.sin(theta_i), 0)
        pos_ang = np.radians(360/n * i + d_pos_ang)
        pos = (D/2 * np.cos(pos_ang), D/2 * np.sin(pos_ang), 0)
        src_i = Cylinder(magnetization=b_rem, position=pos, dimension=(c_d, c_h))
        if (i % 2 == 1):
            magnets.add(src_i)
        if (not alternate and i % 2 == 0):
            magnets.add(src_i)
            
    magnets.rotate_from_angax(90, 'x')
    magnets.rotate_from_angax(-90, 'y')
    user_defined_style = {
        'show': True,
        "size": 0.1,
        'color': {
            'transition': 0,
            'mode': 'tricolor',
            'middle': 'white',
            'north': 'magenta',
            'south': 'turquoise',
        },
        "mode": "arrow+color",
    }
    magpy.defaults.display.style.magnet.magnetization = user_defined_style
#     magpy.show(magnets, style_magnetization_show=True, backend='plotly')
    return magnets