import numpy as np
from .render import vis_of_vertices, render_texture
from scipy import ndimage

def get_visibility(vertices, triangles, h, w):
    triangles = triangles.T
    vertices_vis = vis_of_vertices(vertices.T, triangles, h, w)
    vertices_vis = vertices_vis.astype(bool)
    for k in range(2):
        tri_vis = vertices_vis[triangles[0,:]] | vertices_vis[triangles[1,:]] | vertices_vis[triangles[2,:]]
        ind = triangles[:, tri_vis]
        vertices_vis[ind] = True
    # for k in range(2):
    #     tri_vis = vertices_vis[triangles[0,:]] & vertices_vis[triangles[1,:]] & vertices_vis[triangles[2,:]]
    #     ind = triangles[:, tri_vis]
    #     vertices_vis[ind] = True
    vertices_vis = vertices_vis.astype(np.float32)  #1 for visible and 0 for non-visible
    return vertices_vis

def get_uv_mask(vertices_vis, triangles, uv_coords, h, w, resolution):
    triangles = triangles.T
    vertices_vis = vertices_vis.astype(np.float32)
    uv_mask = render_texture(uv_coords.T, vertices_vis[np.newaxis, :], triangles, resolution, resolution, 1)
    uv_mask = np.squeeze(uv_mask > 0)
    uv_mask = ndimage.binary_closing(uv_mask)
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = ndimage.binary_closing(uv_mask)
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = ndimage.binary_erosion(uv_mask, structure = np.ones((4,4)))  
    uv_mask = uv_mask.astype(np.float32)

    return np.squeeze(uv_mask)

def get_depth_image(vertices, triangles, h, w, isShow = False):
    z = vertices[:, 2:]
    if isShow:
        z = z/max(z)
    depth_image = render_texture(vertices.T, z.T, triangles.T, h, w, 1)
    return np.squeeze(depth_image)
