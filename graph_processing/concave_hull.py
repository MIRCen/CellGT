import torch
from shapely.ops import unary_union
from shapely import geometry
from shapely.geometry import Polygon, MultiLineString, LineString
from shapely.geometry import MultiPolygon
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as pathm
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Point


def plot_polygon(ax, poly, **kwargs):
    """
    Plots a Shapely Polygon (or MultiPolygon's part) on a Matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot the polygon on.
        poly (shapely.geometry.Polygon): The polygon to be plotted.
        **kwargs: Additional keyword arguments passed to the PathPatch (e.g., facecolor, edgecolor, alpha).

    Returns:
        matplotlib.collections.PatchCollection: The patch collection added to the plot.
    """
    path = pathm.make_compound_path(
        pathm(np.asarray(poly.exterior.coords)[:, :2]),
        *[pathm(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def calc_alpha2(alpha, tri,points):
    """
    Computes the alpha-shape (concave hull) based on Delaunay triangulation.

    Args:
        alpha (float or None): Filtering radius. Triangles with circumradius above this value are discarded.
        tri (scipy.spatial.Delaunay): Delaunay triangulation of input points.
        points (ndarray): 2D array of point coordinates.

    Returns:
        tuple:
            - list_polygones (list of shapely.geometry.Polygon): Polygons constructed from filtered triangles.
    """

    # loop over triangles:
    list_polygones=[]
    for ia, ib, ic in tri.simplices:
        # extraction des points de Delaunay
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Longueurs des cotés du triangle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimètre du triangle
        s = (a + b + c) / 2.0
        # Surface du triangle par la formule de Heron
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # rayon de filtrage
        circum_r = a * b * c / (4.0 * area)
        if (alpha is None ) or (circum_r < alpha):
            m = geometry.Polygon([geometry.Point(pa[0], pa[1]),geometry.Point(pb[0], pb[1]), geometry.Point(pc[0], pc[1])])
            list_polygones.append(m)

    return list_polygones


def remove_small_holes(poly, min_perimeter_holes=900, remove_all_holes=True):
    """
    Removes small holes from a polygon based on hole perimeter.

    Args:
        poly (shapely.geometry.Polygon): Input polygon.
        min_perimeter_holes (float): Minimum perimeter threshold for a hole to be preserved.
        remove_all_holes (bool): If True, all holes are removed regardless of size.

    Returns:
        shapely.geometry.Polygon: Cleaned polygon with selected holes removed.
    """
    list_interiors=poly.interiors
    index =np.where([interiors.length >min_perimeter_holes for interiors in list_interiors])
    if (len(index[0])>0) and not remove_all_holes:
        interi = [poly.interiors[int(i)] for i in index[0]]
        return Polygon(poly.exterior,interi )
    else:
        return Polygon(poly.exterior)

def remove_small_polygons(multipoly, min_area_polygon=100000):
    """
    Removes small polygons from a MultiPolygon based on area.

    Args:
        multipoly (shapely.geometry.MultiPolygon): Input MultiPolygon.
        min_area_polygon (float): Minimum area threshold for polygons to be kept.

    Returns:
        shapely.geometry.MultiPolygon: Filtered MultiPolygon.
    """

    index =np.where([poly.area > min_area_polygon for poly in multipoly.geoms])
    list_poly = [multipoly.geoms[int(i)] for i in index[0]]
    return MultiPolygon(list_poly)


def compute_mask(alpha,  points, erosion = -80 , dilation=160, min_perimeter_holes=9000, min_area_polygon=1000000):
    """
    Computes a polygon from a set of 2D points using alpha-shape, erosion, and diltation.

    Args:
        alpha (float or None): Alpha parameter for triangle filtering.
        points (ndarray): 2D coordinates of input points.
        erosion (float): Buffer size for morphological erosion (negative shrinks the shape).
        dilation (float): Buffer size for morphological dilation after erosion.
        min_perimeter_holes (float): Minimum perimeter of holes to retain.
        min_area_polygon (float): Minimum polygon area to retain.

    Returns:
        shapely.geometry.MultiPolygon: Final geometric mask.
    """
    #index_points = np.nonzero(cluster_ids_list)
    points_mask = points
    tri = Delaunay(points_mask)
    list_polygones = calc_alpha2(alpha, tri, points_mask)
    list_polygones = unary_union(list_polygones)
    erosed = list_polygones.buffer(erosion)
    dilated_polygons = erosed.buffer(dilation)
    if dilated_polygons.geom_type == "MultiPolygon":
        dilated_polygons = remove_small_polygons(dilated_polygons, min_area_polygon=min_area_polygon)
        mask_polygons=[]
        for geom in dilated_polygons.geoms:  # new_shape.geoms:
            geom = remove_small_holes(geom, min_perimeter_holes=min_perimeter_holes, remove_all_holes=True)
            mask_polygons.append(geom)
        return MultiPolygon(mask_polygons)
    elif  dilated_polygons.geom_type == "Polygon":
        return MultiPolygon([remove_small_holes(dilated_polygons, min_perimeter_holes=min_perimeter_holes, remove_all_holes=True)])


def compute_masks_label(labeled_nodes,alpha,points,erosion = -80 , dilation=160,nb_classe=5):
    """
    Computes one polygon mask per class label from labeled points.

    Args:
        labeled_nodes (ndarray): Array of integer class labels per point.
        alpha (float): Alpha-shape parameter.
        points (ndarray): 2D coordinates of points.
        erosion (float): Morphological erosion value.
        dilation (float): Morphological dilation value.
        nb_classe (int): Total number of labels/classes to evaluate.

    Returns:
        list of shapely.geometry.Polygon or MultiPolygon: One mask per label (empty if not enough points).
    """
    labels_present=np.unique(labeled_nodes)
    masks_label=[]
    for label in range(0,19):#nb_classe):
        if label in labels_present:

            filtered_index=np.where(labeled_nodes==label)[0]


            if len(filtered_index)>30:
                mask=compute_mask(alpha, points[filtered_index], erosion=erosion, dilation=dilation, min_perimeter_holes=0,
                     min_area_polygon=0 )
            else:
                mask=Polygon()
        else:
            mask = Polygon()
        masks_label.append(mask)
    return masks_label

def multipolygon_contains_point(point,multipolygon):
    if multipolygon.geom_type == "MultiPolygon":
        for g in list(multipolygon):
            if g.contains(point):
                return True
        return False
    else:
        return multipolygon.contains(point)

def compute_mask_list(cluster_ids_list, alpha_list, points, erosion=None, dilation=None,min_perimeter_holes=None,  min_area_polygon=None):
    masks_list = []
    for alpha in alpha_list:

        mask = compute_mask(alpha, erosion=erosion, dilation=dilation,min_perimeter_holes=min_perimeter_holes,  min_area_polygon=min_area_polygon)
        masks_list.append(mask)
    return masks_list



def plot_mask_list(mask_list, color_list=['#542E54','#069AF3','#13EAC9', 'tab:orange', '#f6688e','y','tab:red','#89a0b0', '#a4a2fe','#228b22', '#addffd','#c7faf2','#ffe1ab','#fcc5d3','#ffffb1','#ff7e7e','#d3dbe1','#d5d4ff','#bdefbd']):

    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')
    for mask, c in zip(mask_list, color_list):
        if not mask.is_empty:
            if mask.geom_type == "Polygon":
                plot_polygon(axs, mask, facecolor=c, edgecolor="black", alpha = 1)

            elif mask.geom_type == "MultiPolygon":
                for geom in mask.geoms:  # new_shape.geoms:
                    plot_polygon(axs, geom, facecolor=c, edgecolor="black", alpha = 1)
        else:
            print(f"Wrong mask type, it should be Shaper Polygon or Multipolygon")
    plt.gca().invert_yaxis()
    plt.show()

def  compute_distances_points(data):
    alpha, erosion, dilation, min_perimeter_holes, min_area_polygon = None, -0, 150, 15000, 0
    mask = compute_mask(alpha, points = np.array(data.pos), erosion=-80, dilation=160)
    points = np.array(data.pos)
    return [mask.geoms[0].exterior.distance(Point(x, y)) for x, y in zip(points[:, 0], points[:, 1])]









