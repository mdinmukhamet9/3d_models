import copy
import numpy as np 
import open3d as o3d


axes = {
    'xy': np.array([1, -1, 0]) / np.sqrt(2),
    '-xy':np.array([1, 1, 0]) / np.sqrt(2)

}


# laod data
def load_obj(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    # pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    return mesh

# so all objects will require same threshold
def normalize(obj):
    mesh = copy.deepcopy(obj)
    vertices = np.asarray(mesh.vertices)
    center = np.mean(vertices, axis = 0)
    vertices -= center
    scale = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis =0))
    vertices /= scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

# how well they alligned
def compute_min_dist(n_obj, mirrored):
    v1 = np.asarray(n_obj.vertices)
    v2 = np.asarray(mirrored.vertices)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(v1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(v2)

    dists1 = pcd1.compute_point_cloud_distance(pcd2)
    dists2 = pcd2.compute_point_cloud_distance(pcd1)

    return (np.mean(dists1) + np.mean(dists2)) / 2

# symmetry line for visualization
def create_symmetry_line(center, direction, length=2.0, color=[0.1, 0.7, 0.1]):
    direction = direction / np.linalg.norm(direction)
    p1 = center - direction * length / 2
    p2 = center + direction * length / 2

    points = [p1, p2]
    lines = [[0, 1]]
    colors = [color]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# rotate around needed line
def get_rotation_matrix(T):
    axis = T / np.linalg.norm(T)
    angle = np.pi
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    T = np.eye(4)
    T[:3, :3] = R
    return T

# symmetry checker
def check_symmetry(obj, threshold=0.01, visualize=False):
    n_obj = normalize(obj)
    center = n_obj.get_center()
    for name, T in axes.items():
        mirrored = copy.deepcopy(n_obj)
        T_n = get_rotation_matrix(T)
        mirrored.transform(T_n)

        mean_distance = compute_min_dist(n_obj, mirrored)

        if visualize:
            original_vis = copy.deepcopy(n_obj)
            mirrored_vis = copy.deepcopy(mirrored)
            
            original_vis.compute_vertex_normals()
            mirrored_vis.compute_vertex_normals()
            sym_line = create_symmetry_line(center, T, length=3.0)
            original_vis.paint_uniform_color([0.6, 0.2, 0.2])
            mirrored_vis.paint_uniform_color([0.1, 0.1, 0.6])
            mirrored_vis.translate((0.0001, 0.0001, 0.0001))

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            coord.translate([0.5, 0.0, 0])
            o3d.visualization.draw_geometries([original_vis, mirrored_vis,sym_line, coord])
            print(f"axes:{name}, threshold: {mean_distance}")
        if mean_distance < threshold:
            return True, mean_distance, name, T
    return False, 9000, None, None


if __name__== '__main__':
    obj_mesh = load_obj('obj_files/467499.obj')

    status, dist, axis, transformation = check_symmetry(obj_mesh)
    print(f"::Status: {status}, Axis: {axis}, Mean dist :{dist}\n Transformation: \n{transformation}")