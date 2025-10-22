import sys
import numpy as np
import polyscope as ps
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
import random

def read_xyz_file(file_path: str) -> np.ndarray:
    """ read the file """
    data = np.loadtxt(file_path)
    return data

def estimate_normals(points):
    """ Estimate normals from k=50 nearest neighbors """
    tree = KDTree(points)
    k = 50

    _, knn_indices = tree.query(points, k)  # 50 nn

    normals = np.zeros((points.shape[0], 3))  # create empty array

    for i in range(points.shape[0]):
        '''for each point in the point cloud'''
        neighbors = points[knn_indices[i]]  # fetch its neighbors

        centroid = np.mean(neighbors, axis=0)
        centered_neighbors = neighbors - centroid

        # covariance matrix eigenvalues -> normals
        cov_matrix = np.cov(centered_neighbors, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        normals[i] = normal  # add normal to the array of all normals

    return normals


def orient_normals(normals):
    """ just orient them in z > 0 direction :D Enough to make all roof datasets work """
    for i in range(len(normals)):
        if normals[i, 2] < 0:
            normals[i] = -normals[i]
    return normals


def largest_connected_component(points, k=5):
    tree = KDTree(points)
    _, neighbors = tree.query(points, k=k + 1)
    neighbors = neighbors[:, 1:]  # Ignore self (first neighbor)

    def bfs(start_idx, visited):
        """ Perform BFS (Breadth First Search) and return all connected indices starting from start_idx """
        queue = [start_idx]  # Use a list as the queue
        component = []
        visited[start_idx] = True

        while queue:
            current = queue.pop(0)  # Pop the first element to mimic a queue
            component.append(current)
            for neighbor in neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)  # Add to the end of the list

        return component

    visited = np.zeros(len(points), dtype=bool)
    components = []

    # Identify all connected components
    for i in range(len(points)):
        if not visited[i]:
            component = bfs(i, visited)
            components.append(component)

    try:
        # Find the largest connected component
        largest_component_indices = max(components, key=len)

        # Return points belonging to the largest connected component
        return points[largest_component_indices]
    except:
        return points


def boundary_detection(points):
    tree = KDTree(points)
    boundary = []

    for i in range(len(points)):
        distances, neighbors = tree.query(points[i], 15)  # 15 neighbors
        neighbors = neighbors[1:]

        barycenter = np.mean(points[neighbors], axis=0)
        offset = np.mean(abs(barycenter - points[i]), axis=0)
        if offset > 0.08:
            # manually selected offset value to find points at the edges
            # print(f'offset value {offset}')
            boundary.append(i)

    if len(boundary) != 0:
        return points[boundary]
    return points


def ransac_plane(points, normals, N=30, K=200, tau=0.1, eta=0.7):
    primitives = []
    for _ in range(N):
        # print(f"Number of points to look through: {len(points)}")
        # indices = np.random.choice(len(points), 3, replace=False)
        # p1, p2, p3 = points[indices]

        tree = KDTree(points)
        if points.shape[0] > 0:
            index = np.random.randint(0, points.shape[0])  # choose one random point
        else:
            break

        _, knni = tree.query(points[index], 20)  # nearest 20 neighbors
        indices = np.random.choice(knni, 3, replace=False)  # select random 3 out of all neighbors
        try:
            p1, p2, p3 = points[indices]  # triplet
        except IndexError:
            # in case less than 3 points remain in points
            break

        # get the normal from triplet, orient and normalize
        v1 = p2 - p1
        v2 = p3 - p1
        triplet_normal = np.cross(v1, v2)
        triplet_normal = -triplet_normal if triplet_normal[2] < 0 else triplet_normal
        triplet_normal = triplet_normal / np.linalg.norm(triplet_normal)

        inliers = []

        for j in range(len(points)):
            # first inliers identification
            pq = points[j] - p1
            distance = np.abs(np.dot(pq, triplet_normal))
            normal_alignment = np.abs(np.dot(triplet_normal, normals[j]))
            if distance < tau and normal_alignment > eta:
                #  normal_alignment > eta and distance < tau
                inliers.append(j)

        if len(inliers) > K:
            # to keep the largest continuous piece if inlier treshold surpassed
            inliers_points = largest_connected_component(points[inliers])

            # primitives.append([inliers_points, []])

            # new plane from all inliers and find inliers again for more accurate result
            centroid = np.mean(inliers_points, axis=0)
            centered_inlier_points = inliers_points - centroid
            _, _, vh = np.linalg.svd(centered_inlier_points)
            new_plane_normal = vh[-1]  # last one in z-direction (normal)
            new_inliers = []

            for j in range(len(points)):
                # second inliers identification
                pq = points[j] - centroid
                distance = np.abs(np.dot(pq, new_plane_normal))
                normal_alignment = np.abs(np.dot(new_plane_normal, normals[j]))
                if distance < tau and normal_alignment > eta:
                    new_inliers.append(j)

            new_inlier_points = largest_connected_component(points[new_inliers])  # instead of new_inliers
            # print(f'new inliers {len(new_inlier_points)}')
            boundary_points = boundary_detection(new_inlier_points)
            # print(f'boundary points {len(boundary_points)}')
            primitives.append([boundary_points, new_plane_normal])

            # to remove saved points from pointcloud
            mask = np.ones(points.shape[0], dtype=bool)
            mask[new_inliers] = False  # new_inliers
            normals = normals[mask]
            points = points[mask]


    # return [max(primitives, key=len)]
    return primitives

def triangulate(points, pca=False):
    """ PCA included (optional) """
    if pca:
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        principal_directions = vh[:2]  # first two x, y-direction
        projected = centered_points @ principal_directions.T  # project into 2d
        minims = np.min(projected, axis=0)
        maxims = np.max(projected, axis=0)
        corners = np.array([
            [minims[0], minims[1]],
            [minims[0], maxims[1]],
            [maxims[0], maxims[1]],
            [maxims[0], minims[1]]
        ])
        corners_3d = (corners @ principal_directions) + centroid  # project back to 3d
        delaunay = Delaunay(corners_3d[:, :2])
        triangles = delaunay.simplices
        return corners_3d, triangles
    else:
        delaunay = Delaunay(points[:, :2])  # Project to 2d
        triangles = delaunay.simplices
        return points, triangles


########## Main script
if __name__ == "__main__":
    input_xyz_file = sys.argv[1]
    # input_xyz_file = 'simple_roof.xyz'
    points = read_xyz_file(input_xyz_file)

    ps.init()
    ps.set_ground_plane_mode("none")
    ps_points = ps.register_point_cloud("raw points", points, radius=0.001, enabled=True)

    normals = estimate_normals(points)
    oriented_normals = orient_normals(normals)
    # ps_points.add_vector_quantity("normals", oriented_normals, enabled=True)

    # triplet = select_triplet(points)
    # ps_points = ps.register_point_cloud("random triplet", triplet, enabled=True)

    primitives = ransac_plane(points, oriented_normals, N=30, K=100, tau=0.5, eta=0.9)

    for i, primitive in enumerate(primitives):
        primitive_points, primitive_normal = primitive
        ps.register_point_cloud(f"primitive {i}", primitive_points, radius=0.003, enabled=True)
        # plane_points = ps.register_point_cloud(f"plane random points {i}", building_plane, enabled=True)

        corners, triangles = triangulate(primitive_points, pca=True)
        ps.register_surface_mesh(f"surface mesh {i}", corners, triangles)
        # ps.register_point_cloud(f"corners {i}", corners, enabled=True)


    ps.show()




