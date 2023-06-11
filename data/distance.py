import os, sys
import numpy as np

from collections import defaultdict
from enum import Enum
from sklearn.metrics import pairwise_distances

from geomdl import BSpline


##########################  Primitive Class  ##########################

class Prim_type(Enum):
    Line = 1
    Circle = 2
    BSpline = 3
    Ellipse = 4
    Unknown = 5

class Primitive:
    
    def __init__(self, info, center, scale):
        
        self.flag_sharp = False
        self.type = None
        # Circle
        self.location = None # center
        self.radius = None   # radius
        # B-splines
        self.flag_close = False
        self.degree = None
        self.poles = []
        self.knots = []
        # Ellipse
        self.maj_radius = 0.
        self.min_radius = 0.
        self.focus_1 = None
        self.focus_2 = None
        
        for line in info:
            if 'sharp' in line:
                if line[-5:] == ' true':
                    self.flag_sharp = True
                
            if 'vert_indices' in line:
                self.vert_indices = np.array(list(map(int, line.lstrip('  vert_indices: [').rstrip(']').split(','))))
                if self.vert_indices[0] == self.vert_indices[-1]:
                    self.flag_close = True

            if 'type' in line:
                ptype = line.split(": ")[1]
                if ptype == 'Line':
                    self.type = Prim_type.Line
                elif ptype == 'Circle':
                    self.type = Prim_type.Circle
                elif ptype == 'BSpline':
                    self.type = Prim_type.BSpline
                elif ptype == 'Ellipse':
                    self.type = Prim_type.Ellipse
                else:
                    print("Unknown type: ", ptype)
                    self.type = Prim_type.Unknown
                    
            # Circle Info        
            if 'location' in line:
                self.location = np.array((line.split(": ")[1][1:-1]).split(", ")).astype(float)
                self.location = (self.location - center) / scale

            if 'radius' in line:
                self.radius = float(line.split(": ")[1]) / scale
                
            # BSpline Info
            if 'closed' in line:
                self.flag_close = True if line.split(": ")[1] == 'true' else False
            if 'degree' in line:
                self.degree = int(line.split(": ")[1])
            if '- [' in line:
                key_point = list(map(float, (line.split("[")[1][:-1]).split(", ")))
                key_point = [(key_point[i] - center[i]) / scale for i in range(3)]
                self.poles.append(key_point)
            if 'knots' in line:
                self.knots = np.array(list(map(float, (line.split(": [")[1][:-1]).split(", "))))
                
            # Ellipse Info
            if 'maj_radius' in line:
                self.maj_radius = float(line.split(": ")[1]) / scale
            if 'min_radius' in line:
                self.min_radius = float(line.split(": ")[1]) / scale
            if 'focus1' in line:
                self.focus_1 = np.array(list(map(float, ((line.split(": [")[1]).split("]")[0]).split(", "))))
                self.focus_1 = (self.focus_1 - center) / scale
            if 'focus2' in line:
                self.focus_2 = np.array(list(map(float, ((line.split(": [")[1]).split("]")[0]).split(", "))))
                self.focus_2 = (self.focus_2 - center) / scale
            
        if self.type == Prim_type.BSpline and self.degree == 1:
            self.type = Prim_type.Line
                
    def __str__(self):
        
        descript = ""
        descript += "Primitive type: %s\n" % (self.type.name)
        descript += "Flag sharp: %r\n" % (self.flag_sharp)
        descript += "Flag close: %r\n" % (self.flag_close)
        
        return descript


##########################  I/O Functions  ##########################

def read_yml(filepath, center, scale):
    
    with open(filepath, 'r') as fp:
        fp_line = fp.read().split("\n")[:-1]
        
    primitives = []
    info = []
    line = ''

    for i in range(1, len(fp_line)):

        if fp_line[i][:8] == 'surfaces':
            prim = Primitive(info, center, scale)
            if prim.flag_sharp:
                primitives.append(prim)
            break

        if len(info) > 0 and fp_line[i][0] == '-':
            prim = Primitive(info, center, scale)
            if prim.flag_sharp:
                primitives.append(prim)
            info = []
            line = ''

        if len(line) == 0 and fp_line[i][-1] != ',':
            info.append(fp_line[i])
        elif len(line) == 0 and fp_line[i][-1] == ',':
            line = fp_line[i]
        elif len(line) != 0 and fp_line[i][-1] == ',':
            line = line + fp_line[i]
        else:
            line = line + fp_line[i]
            info.append(line)
            line = ''

    return primitives


def read_obj(filepath):
    
    with open(filepath, 'r') as fp:
        fp_line = fp.read().split("\n")[:-1]
        
    verts = [line for line in fp_line if line[:2] == 'v ']
    np_verts = np.array([vert.split()[1:] for vert in verts]).astype(float)
    
    return np_verts


def read_ply(filepath):
    with open(filepath, 'r') as fp:
        fp_line = fp.read().split("\n")[:-1]
    read = False
    infos = []
    for line in fp_line:
        if read == True:
            infos.append(line)
        elif read == False and line[:5] == 'end_h':
            read = True
    np_infos=np.array([info.split()[1:] for info in infos]).astype(float)
    np_verts = np_infos[:,0:3]
    np_offsets = np_infos[:,3:6]
    np_labels = np_infos[:,6:7]

    return np_verts, np_offsets, np_labels

def read_obj_and_normalize(filepath):
    
    m_verts = read_obj(filepath)
    # compute center
    center = np.average(m_verts, axis = 0)
    m_verts_center = m_verts - center.reshape((1, -1))
    # compute scale
    scale = np.max(np.abs(m_verts_center)) # scale to normalize the point cloud to [-1, 1] * [-1, 1] * [-1, 1]
    m_verts_normalized = m_verts_center / scale
    
    return m_verts_normalized, center, scale


def read_xyz(filepath, center, scale, with_normal):
    
    with open(filepath, 'r') as fp:
        fp_line = fp.read().split("\n")[:-1]
        
    fp_line = np.array([line.split() for line in fp_line]).astype(float)

    # normalization
    pts = fp_line[:, :3]
    pts = (pts - center.reshape((1, -1))) / scale

    # delete normal
    if with_normal:
        assert(fp_line.shape[1] > 3)
        # normalization
        normal = fp_line[:, 3:]
        normal = normal / np.linalg.norm(normal, axis = 1)
        return pts, normal
    
    return pts


def read_xyz_with_scaling(filepath):
    
    with open(filepath, 'r') as fp:
        fp_line = fp.read().split("\n")[:-1]
        
    fp_line = np.array([line.split() for line in fp_line]).astype(float)
    
    # delete normal
    m_verts = fp_line[:, :3] if fp_line.shape[1] > 3 else fp_line
    
    # normalization
    center = np.average(m_verts, axis = 0)
    m_verts_center = m_verts - center.reshape((1, -1))
    scale = np.max(np.abs(m_verts_center))
    m_verts_normalized = m_verts_center / scale
    
    return m_verts_normalized, center, scale

##########################  Distance Utils  ##########################

def read_xyz_no_normalization(filepath):

    with open(filepath, 'r') as fp:
        fp_line = fp.read().split("\n")[:-1]

    fp_line = np.array([line.split() for line in fp_line]).astype(float)

    # delete normal
    m_verts = fp_line[:, :3] if fp_line.shape[1] > 3 else fp_line

    return m_verts

def find_basis_from_vertices(p1, p2, p3):
    
    center = p1
    base_1 = p2 - center
    base_1 = base_1 / np.linalg.norm(base_1)
    
    normal = np.cross(base_1, p3 - center)
    normal = normal / np.linalg.norm(normal)
    
    base_2 = np.cross(base_1, normal)
    base_2 = base_2 / np.linalg.norm(base_2)
    
    if np.dot(base_2, p3 - center) < 0:
        base_2 = -base_2
    
    return base_1, base_2, normal

def proj_points_on_plane(base_1, base_2, center, point):
    # point: N x 3
    # return: N x 2
    
    pc = point - center.reshape((1, -1))
    px = np.dot(pc, base_1.reshape((-1, 1)))
    py = np.dot(pc, base_2.reshape((-1, 1)))
    
    return np.hstack((px, py))


##########################  Distance Functions  ##########################

def dist_from_points_to_line(primitive, points, mesh_verts):
    
    start = mesh_verts[primitive.vert_indices[0]]
    end = mesh_verts[primitive.vert_indices[-1]]
    direction = end - start
    
    vec_start = points - start
    vec_end = points - end

    ind_start = (vec_start.dot(direction) < 0)
    ind_end = (vec_end.dot(direction) > 0)

    dist_start = np.linalg.norm(vec_start, axis = 1)
    dist_end = np.linalg.norm(vec_end, axis = 1)
    # dist = ||PA x BA|| / ||BA||
    dist_seg = np.linalg.norm(np.cross(vec_start, direction), axis = 1) / np.linalg.norm(direction)
    # proj = A + (dot(PA, BA) / dot(BA, BA)) BA
    projs = start.reshape((1, -1)) + (vec_start.dot(direction) / np.power(np.linalg.norm(direction), 2)).reshape((-1, 1)) * direction.reshape((1, -1))
    
    dist_seg[ind_start] = dist_start[ind_start]
    dist_seg[ind_end] = dist_end[ind_end]
    projs[ind_start] = start
    projs[ind_end] = end
    
    
    return dist_seg, projs


def dist_from_points_to_ellipse(primitive, points, mesh_verts, num_sampling = 1e4):
    # find plane defined by primitive
    center = 0.5 * (primitive.focus_1 + primitive.focus_2)
    # avoid when center, focus_1 and vert be collinear
    is_parellel = (np.linalg.norm(np.cross(primitive.focus_1 - center, mesh_verts[primitive.vert_indices[0]] - center)) < 1e-5)
    base_1, base_2, _ = find_basis_from_vertices(center, primitive.focus_1, mesh_verts[primitive.vert_indices[1]]) if is_parellel \
                        else find_basis_from_vertices(center, primitive.focus_1, mesh_verts[primitive.vert_indices[0]])
    
    proj_points = proj_points_on_plane(base_1, base_2, center, mesh_verts[primitive.vert_indices])
    # note: projected center is (0, 0)
    theta = None  # 2d points
    
    if primitive.flag_close:
        step = 2. * np.pi / num_sampling
        theta = np.arange(0, 2 * np.pi + 1e-8, step)
    else:
        # find bisector and max angle
        v1 = proj_points[0] # - center (zero)
        vmid = proj_points[len(proj_points) // 2] # - center (zero)
        vn = proj_points[-1] # - center (zero)
        bisector = 0.5 * (v1 / np.linalg.norm(v1) + vn / np.linalg.norm(vn))
        if np.linalg.norm(bisector) < 1e-5: # v1 and vn collinear
            #print('Collinear v1 and vn!')
            bisector[0] = -vn[1]
            bisector[1] = vn[0]
        bisector = bisector / np.linalg.norm(bisector)
        if bisector.dot(vmid) < 0:
            bisector = -bisector
        max_angle = np.arccos(np.clip(v1.dot(bisector) / np.linalg.norm(v1), -1., 1.))
        bisect_angle = np.arccos(np.clip(bisector[0], -1., 1.))
        if bisector[1] < 0:
            bisect_angle = -bisect_angle
        step = 2 * max_angle / num_sampling
        theta = np.arange(bisect_angle - max_angle, bisect_angle + max_angle + 1e-4, step)
        
    # 2d coordinates of sampled points
    x = primitive.maj_radius * np.cos(theta)
    y = primitive.min_radius * np.sin(theta)
    # 2d coordinates of sampled points
    sample_3d = np.repeat(center.reshape((1, -1)), len(x), axis = 0) + \
                                    x.reshape((-1, 1)) * base_1.reshape((1, -1)) + \
                                    y.reshape((-1, 1)) * base_2.reshape((1, -1))
    # pairwise dists
    pairwise_dists = pairwise_distances(points, sample_3d)
    min_indices = np.argmin(pairwise_dists, axis = 1)
    min_dists = pairwise_dists[np.arange(len(pairwise_dists)), min_indices]
    min_points = sample_3d[min_indices]
    
    return min_dists, min_points


def dist_from_points_to_circle_arc(primitive, points, mesh_verts):
    
    center = primitive.location
    radius = primitive.radius
    # find plane
    base_1, base_2, plane_normal = find_basis_from_vertices(center, mesh_verts[primitive.vert_indices[0]], mesh_verts[primitive.vert_indices[1]])
    proj_inliers = proj_points_on_plane(base_1, base_2, center, mesh_verts[primitive.vert_indices])
    # find nearest to full circle
    # proj_points = proj_points_on_plane(base_1, base_2, center, points)
    # proj_points = proj_points / np.linalg.norm(proj_points, axis = 1).reshape((-1, 1))
    proj_points = proj_points_on_plane(base_1, base_2, center, points)
    proj_points_norm = np.linalg.norm(proj_points, axis=1)
    proj_points_zero = (proj_points_norm < 1e-10)
    proj_points_norm[proj_points_zero] = 1.
    proj_points = proj_points / proj_points_norm.reshape((-1, 1))
    
    if not primitive.flag_close:
        # find bisector and max angle
        v1 = proj_inliers[0] # - center
        vmid = proj_inliers[len(proj_inliers) // 2] # - center
        vn = proj_inliers[-1] #- center
        bisector = 0.5 * (v1 + vn)
        if np.linalg.norm(bisector) < 1e-5: # v1 and vn collinear
            #print('Collinear v1 and vn!')
            bisector[0] = -vn[1]
            bisector[1] = vn[0]
        bisector = bisector / np.linalg.norm(bisector)
        if bisector.dot(vmid) < 0:
            bisector = -bisector
        max_angle = np.arccos(np.clip(v1.dot(bisector) / radius, -1., 1.))
    
        # find indices of points connected with two extremes
        angles = np.arccos(np.clip(np.dot(proj_points, bisector.reshape((-1, 1))).reshape((-1,)), -1., 1.))
        flag_extreme = (angles > max_angle)
        flag_head = np.arccos(np.clip(np.dot(proj_points, v1.reshape((-1, 1))).reshape((-1,)) / radius, -1., 1.)) < \
                    np.arccos(np.clip(np.dot(proj_points, vn.reshape((-1, 1))).reshape((-1,)) / radius, -1., 1.))
        ind_v1 = np.bitwise_and(flag_extreme, flag_head)
        ind_vn = np.bitwise_and(flag_extreme, np.logical_not(flag_head))
        proj_points[ind_v1] = proj_inliers[0] / radius
        proj_points[ind_vn] = proj_inliers[-1] / radius
        
    # from 2d to 3d
    proj_points_3d = np.repeat(center.reshape((1, -1)), len(proj_points), axis = 0) + \
                               radius * proj_points[:, 0].reshape((-1, 1)) * base_1.reshape((1, -1)) + \
                               radius * proj_points[:, 1].reshape((-1, 1)) * base_2.reshape((1, -1))
    dists = np.linalg.norm(points - proj_points_3d, axis = 1)
    
    return dists, proj_points_3d


def dist_from_points_to_bspline(primitive, points, eva_rate = 0.05):
    
    # Set up curve
    curve = BSpline.Curve()
    curve.degree = primitive.degree
    curve.ctrlpts = primitive.poles
    curve.knotvector = primitive.knots
    
    # Set evaluation delta
    curve.delta = eva_rate
    curve.evaluate(start = 0., stop = 1.)
    curve_points = np.array(curve.evalpts)
    
    # Compute distance
    pairwise_dists = pairwise_distances(points, curve_points)
    min_indices = np.argmin(pairwise_dists, axis = 1)
    min_dists = pairwise_dists[np.arange(len(pairwise_dists)), min_indices]
    min_points = curve_points[min_indices]
    
    return min_dists, min_points


def dist_from_points_to_primitive(primitive, points, mesh_verts):
    
    if primitive.type == Prim_type.Line:
        return dist_from_points_to_line(primitive, points, mesh_verts)
    if primitive.type == Prim_type.Circle:
        return dist_from_points_to_circle_arc(primitive, points, mesh_verts)
    if primitive.type == Prim_type.BSpline:
        return dist_from_points_to_bspline(primitive, points)
    if primitive.type == Prim_type.Ellipse:
        return dist_from_points_to_ellipse(primitive, points, mesh_verts, num_sampling = 1e4)
    
    return None, None


def sample_line(primitive, mesh_verts, step):

    start = mesh_verts[primitive.vert_indices[0]]
    end = mesh_verts[primitive.vert_indices[-1]]
    direction = end - start

    steps = np.arange(0, np.linalg.norm(direction) + 1e-8, step)
    direction/= np.linalg.norm(direction)

    sample_3d = np.repeat(start.reshape((1, -1)), len(steps), axis=0) + \
                steps.reshape((-1, 1)) * direction.reshape((1, -1))
    return sample_3d

def sample_ellipse(primitive, mesh_verts, step):
    center = 0.5 * (primitive.focus_1 + primitive.focus_2)
    # avoid when center, focus_1 and vert be collinear
    is_parellel = (np.linalg.norm(
        np.cross(primitive.focus_1 - center, mesh_verts[primitive.vert_indices[0]] - center)) < 1e-5)
    base_1, base_2, _ = find_basis_from_vertices(center, primitive.focus_1,
                                                 mesh_verts[primitive.vert_indices[1]]) if is_parellel \
        else find_basis_from_vertices(center, primitive.focus_1, mesh_verts[primitive.vert_indices[0]])

    proj_points = proj_points_on_plane(base_1, base_2, center, mesh_verts[primitive.vert_indices])
    # note: projected center is (0, 0)
    theta = None  # 2d points

    step_angle=np.arcsin(np.clip(step/primitive.maj_radius,0,1))
    if primitive.flag_close:

        theta = np.arange(0, 2 * np.pi + 1e-8, step_angle)
    else:
        # find bisector and max angle
        v1 = proj_points[0]  # - center (zero)
        vmid = proj_points[len(proj_points) // 2]  # - center (zero)
        vn = proj_points[-1]  # - center (zero)
        bisector = 0.5 * (v1 / np.linalg.norm(v1) + vn / np.linalg.norm(vn))
        if np.linalg.norm(bisector) < 1e-5:  # v1 and vn collinear
            # print('Collinear v1 and vn!')
            bisector[0] = -vn[1]
            bisector[1] = vn[0]
        bisector = bisector / np.linalg.norm(bisector)
        if bisector.dot(vmid) < 0:
            bisector = -bisector
        max_angle = np.arccos(np.clip(v1.dot(bisector) / np.linalg.norm(v1), -1., 1.))
        bisect_angle = np.arccos(np.clip(bisector[0], -1., 1.))
        if bisector[1] < 0:
            bisect_angle = -bisect_angle

        theta = np.arange(bisect_angle - max_angle, bisect_angle + max_angle + 1e-4, step_angle)

    # 2d coordinates of sampled points
    x = primitive.maj_radius * np.cos(theta)
    y = primitive.min_radius * np.sin(theta)
    # 2d coordinates of sampled points
    sample_3d = np.repeat(center.reshape((1, -1)), len(x), axis=0) + \
                x.reshape((-1, 1)) * base_1.reshape((1, -1)) + \
                y.reshape((-1, 1)) * base_2.reshape((1, -1))
    return sample_3d

def sample_circle_arc(primitive, mesh_verts, step):
    center = primitive.location
    radius = primitive.radius
    base_1, base_2, plane_normal = find_basis_from_vertices(center, mesh_verts[primitive.vert_indices[0]],
                                                            mesh_verts[primitive.vert_indices[1]])
    # avoid when center, focus_1 and vert be collinear


    proj_points = proj_points_on_plane(base_1, base_2, center, mesh_verts[primitive.vert_indices])
    # note: projected center is (0, 0)
    theta = None  # 2d points
    step_angle = np.arcsin(np.clip(step / radius, 0, 1))
    if primitive.flag_close:

        theta = np.arange(0, 2 * np.pi + 1e-8, step_angle)
    else:
        # find bisector and max angle
        v1 = proj_points[0]  # - center (zero)
        vmid = proj_points[len(proj_points) // 2]  # - center (zero)
        vn = proj_points[-1]  # - center (zero)
        bisector = 0.5 * (v1 / np.linalg.norm(v1) + vn / np.linalg.norm(vn))
        if np.linalg.norm(bisector) < 1e-5:  # v1 and vn collinear
            # print('Collinear v1 and vn!')
            bisector[0] = -vn[1]
            bisector[1] = vn[0]
        bisector = bisector / np.linalg.norm(bisector)
        if bisector.dot(vmid) < 0:
            bisector = -bisector
        max_angle = np.arccos(np.clip(v1.dot(bisector) / np.linalg.norm(v1), -1., 1.))
        bisect_angle = np.arccos(np.clip(bisector[0], -1., 1.))
        if bisector[1] < 0:
            bisect_angle = -bisect_angle

        theta = np.arange(bisect_angle - max_angle, bisect_angle + max_angle + 1e-4, step_angle)

    # 2d coordinates of sampled points
    x = radius * np.cos(theta)
    y =radius * np.sin(theta)
    # 2d coordinates of sampled points
    sample_3d = np.repeat(center.reshape((1, -1)), len(x), axis=0) + \
                x.reshape((-1, 1)) * base_1.reshape((1, -1)) + \
                y.reshape((-1, 1)) * base_2.reshape((1, -1))
    return sample_3d

def sample_bspline(primitive, step):
    curve = BSpline.Curve()
    curve.degree = primitive.degree
    curve.ctrlpts = primitive.poles
    curve.knotvector = primitive.knots

    # Set evaluation delta
    curve.delta = step
    curve.evaluate(start=0., stop=1.)
    curve_points = np.array(curve.evalpts)

    return curve_points


def sample_primitive(primitive, mesh_verts,step):
    if primitive.type == Prim_type.Line:
        return sample_line(primitive, mesh_verts, step)
    if primitive.type == Prim_type.Circle:
        return sample_circle_arc(primitive, mesh_verts, step)
    if primitive.type == Prim_type.BSpline:
        return sample_bspline(primitive, step)
    if primitive.type == Prim_type.Ellipse:
        return sample_ellipse(primitive, mesh_verts, step)

    return None
def process_sampling(primitives, mesh_vertices,step):
    samples=[]
    for i in range(len(primitives)):
        sample = sample_primitive(primitives[i], mesh_vertices,step)
        samples.extend(sample)
    samples = np.array(samples)

    return samples

def process_dists_to_primitives(samples, primitives, mesh_vertices, threshold = 0.05):
    
    ldists = []
    lprojs = []
    
    for i in range(len(primitives)):
        dists, projs = dist_from_points_to_primitive(primitives[i], samples, mesh_vertices)
        ldists.append(dists)
        lprojs.append(projs)

    ldists = np.array(ldists).T
    lprojs = np.array(lprojs)

    indices = np.argmin(ldists, axis = 1)
    dists = ldists[np.arange(len(samples)), indices]
    offsets = lprojs[indices, np.arange(len(samples))] - samples
    
    labels = (dists <= threshold)
    #offsets[np.logical_not(labels)] = 0.
    
    return labels.astype(int), offsets