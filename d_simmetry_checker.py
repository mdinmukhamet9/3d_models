import numpy as np 
import open3d as o3d
import copy

def load_obj_as_pcd(path, number_of_points = 10000):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    return pcd


def check_symmetry(obj, threshold=0.01, visualize=True):
    axes = {
        'xy': np.diag([-1, -1, 1, 1]),
        'x': np.diag([-1, 1, 1, 1]),
        'y': np.diag([1, -1, 1, 1]),
        'z': np.diag([1, 1, -1, 1]),
    }

    original_points = np.asarray(obj.points)

    for name, T in axes.items():
        mirrored = copy.deepcopy(obj)
        mirrored.transform(T.copy())
        mirrored_points = np.asarray(mirrored.points)

        dists = np.linalg.norm(np.sort(original_points, axis=0) - np.sort(mirrored_points, axis=0), axis=1)
        mean_dist = np.mean(dists)

        if visualize:
            original_vis = copy.deepcopy(obj)
            mirrored_vis = copy.deepcopy(mirrored)

            original_vis.paint_uniform_color([1, 0, 0])
            mirrored_vis.paint_uniform_color([0, 1, 0])

            o3d.visualization.draw_geometries([original_vis, mirrored_vis])

        if mean_dist < threshold:
            return True, name, T

    return False, None, None



if __name__== '__main__':
    obj_cloud = load_obj_as_pcd('obj_files/468011.obj')

    status, axis, transformation = check_symmetry(obj_cloud)
    print(f"Status: {status}, Axis: {axis}\n Transformation: {transformation}")