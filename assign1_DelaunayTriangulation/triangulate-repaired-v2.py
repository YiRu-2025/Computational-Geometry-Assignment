import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from geompreds import orient2d, incircle
from itertools import permutations
import argparse
from functools import partial


class HilbertSort3D(object):

    def __init__(self, origin=(0,0,0), radius=1.0, bins=32):
        '''
        '''
        self.origin = np.array(origin)
        self.radius = radius
        self.bins = bins
        order = np.log2(32)
        if order%1.0 > 0.0: raise ValueError("HilbertSort: Bins should be a power of 2.")
        self.curve = self._hilbert_3d(int(np.log2(bins)))

    def _hilbert_3d(self, order):
            '''
            Method generates 3D hilbert curve of desired order.
            Param:
                order - int ; order of curve
            Returns:
                np.array ; list of (x, y, z) coordinates of curve
            '''

            def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
                if order == 0:
                    xx = x + (xi + yi + zi)/3
                    yy = y + (xj + yj + zj)/3
                    zz = z + (xk + yk + zk)/3
                    array.append((xx, yy, zz))
                else:
                    gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

                    gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
                               yi/2, yj/2, yk/2, array)
                    gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
                               xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
                    gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
                               -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                    gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
                               -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                    gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
                               -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                    gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
                               -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                    gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
                               -zk/2, -xi/2, -xj/2, -xk/2, array)

            n = pow(2, order)
            hilbert_curve = []
            gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

            return np.array(hilbert_curve).astype('int')

    # def sort(self, data):
    #     '''
    #     Method bins points according to parameters and sorts by traversing binning
    #     matrix using hilbert space-filling curve.
    #     Param:
    #         data - np.array; list of 3D points; (Nx3)
    #     Returns:
    #         sorted_data - np.array; list of sorted 3D points; (Nx3)
    #
    #     '''
    #     # Center data around origin
    #     data_ = data - self.origin
    #
    #     # Bin points
    #     binned = [[[[] for k in range(self.bins)] for j in range(self.bins)] for i in range(self.bins)]
    #     bin_interval = ((self.radius*2) / self.bins)
    #     offset = int(self.radius/bin_interval)
    #     for i, _ in enumerate(data):
    #         x = int(_[-3]/bin_interval) + offset
    #         y = int(_[-2]/bin_interval) + offset
    #         z = int(_[-1]/bin_interval) + offset
    #         if (x > self.bins-1) or (x < 0): continue
    #         if (y > self.bins-1) or (y < 0): continue
    #         if (z > self.bins-1) or (z < 0): continue
    #         binned[x][y][z].append(_)
    #
    #     # Traverse and Assemble
    #     sorted_data = []
    #     for _ in self.curve:
    #         x = binned[_[0]][_[1]][_[2]]
    #         if len(x) > 0:
    #             sorted_data.append(np.array(x))
    #     sorted_data = np.concatenate(sorted_data, axis=0)
    #
    #     return sorted_data

    def sort(self, data):
        # Dynamically adjust origin and radius
        self.origin = np.mean(data, axis=0)
        self.radius = np.max(np.linalg.norm(data - self.origin, axis=1)) * 20

        # Rescale data to fit within the binning range
        scale = np.max(np.abs(data - self.origin)) / self.radius
        data_ = (data - self.origin) / scale

        # Bin points
        binned = [[[[] for _ in range(self.bins)] for _ in range(self.bins)] for _ in range(self.bins)]
        bin_interval = ((self.radius * 2) / self.bins)
        offset = int(self.bins / 2)

        for _ in data_:
            x = max(0, min(self.bins - 1, int(_[0] / bin_interval) + offset))
            y = max(0, min(self.bins - 1, int(_[1] / bin_interval) + offset))
            z = max(0, min(self.bins - 1, int(_[2] / bin_interval) + offset))
            binned[x][y][z].append(_)

        # Verify all points are binned
        binned_count = sum(len(pt) for plane in binned for row in plane for pt in row)
        if binned_count != len(data):
            raise ValueError(f"Binning error: {len(data) - binned_count} points were lost.")

        # Traverse and Assemble
        sorted_data = []
        for _ in self.curve:
            x = binned[_[0]][_[1]][_[2]]
            if len(x) > 0:
                sorted_data.append(np.array(x))

        if len(sorted_data) == 0:
            raise ValueError("No points fall within the binning range.")

        sorted_data = np.concatenate(sorted_data, axis=0)

        # Final validation
        if len(sorted_data) != len(data):
            raise ValueError(f"Sorting error: {len(data) - len(sorted_data)} points were lost.")

        # print(f"data len: {len(data)}")
        # print(f"sorted data len: {len(sorted_data)}")
        return sorted_data * scale + self.origin  # Rescale back to original size




class TriangularMesh:

    def __init__(self, vertices, triangles):
        self.vertices = []
        j = 0
        for v in vertices:
            self.vertices.append(Vertex(v[0], v[1], j, None))
            j = j + 1

        self.faces = []
        self.halfedges = []
        edges = {}
        j = 0
        index = 0
        for t in triangles:
            indices = [index, index + 1, index + 2]
            index = index + 3
            for i in range(3):
                self.vertices[t[i]].halfedge = indices[i]
                self.halfedges.append(Halfedge(indices[(i + 1) % 3], None, indices[(i + 2) % 3], t[i], j, indices[i]))
                edges[(t[i], t[(i + 1) % 3])] = indices[i]
            self.faces.append(Face(j, indices[0]))
            j = j + 1
        for e, ind1 in edges.items():
            if (e[1], e[0]) in edges:
                ind2 = edges[(e[1], e[0])]
                self.halfedges[ind1].opposite = ind2
                self.halfedges[ind2].opposite = ind1


class Vertex:

    def __init__(self, x=0, y=0, index=None, halfedge=None):
        self.x = x
        self.y = y
        self.index = index
        self.halfedge = halfedge

class Face:

    def __init__(self, index=None, halfedge=None):
        self.index = index
        # halfedge going ccw around this facet.
        self.halfedge = halfedge

class Halfedge:

    def __init__(self, next=None, opposite=None, prev=None, vertex=None,
                 facet=None, index=None):
        self.opposite = opposite
        self.next = next
        self.prev = prev
        self.vertex = vertex
        self.facet = facet
        self.index = index


def space_filling_sort(point_cloud):
    # Calculate the centroid (center of gravity)
    # centroid = np.mean(point_cloud, axis=0)
    # print(f"Centroid: {centroid}")
    # # Calculate the distance of each point to the centroid adn find the maximum distance (radius)
    # radius = max(np.linalg.norm(point_cloud - centroid, axis=1))
    #
    # print(f"radius: {radius}")
    #
    # origin = (centroid[0], centroid[1], 0)

    dummy_origin = (0,0,0)
    sorter = HilbertSort3D(origin=dummy_origin, radius=1,
                           bins=32)  # use 2x radius, this needs adjusting based on dataset maybe
    points_3d = [point + [0] for point in point_cloud]  # add 0 to make 3d for sorter
    # print(points_3d)
    sorted_3d = sorter.sort(points_3d)
    sorted_2d = [point[:2] for point in sorted_3d]  # sorted points in 2D array

    # print(f'point cloud: {len(point_cloud)}')
    # print(f'sorted 2d: {len(sorted_2d)}')
    return sorted_2d

def read_input(filename, plot=False):
  # function that reads input file and plots the point cloud for visualisation
  with open(filename, 'r') as f:
    n = int(f.readline())
    # print(f'Number of points: {n}')
    # print(f'Expected triangles: {2*n + 2}')
    points = []

    for _ in range(n):
        x, y = map(float, f.readline().split())
        points.append([x, y])

  # print(f'number of read points: {len(points)}')

  x_coords = [point[0] for point in points]
  y_coords = [point[1] for point in points]

  # these are the infinite points forming convex hull
  # maybe adapt the number based on the distribution of the point cloud
  eta = 0.02 * np.sqrt((max(x_coords) - min(x_coords))**2 + (max(y_coords) - min(y_coords))**2)
  xlim = min(x_coords) - eta, max(x_coords) + eta
  ylim = min(y_coords) - eta, max(y_coords) + eta

  if plot:
    plt.scatter(x_coords, y_coords, color='black', marker='.')

    plt.plot(xlim[0], ylim[0], color='red', marker='.')
    plt.plot(xlim[0], ylim[1], color='red', marker='.')
    plt.plot(xlim[1], ylim[0], color='red', marker='.')
    plt.plot(xlim[1], ylim[1], color='red', marker='.')
    plt.show()

  return points, xlim, ylim

def draw_polygons(vertices, connections, color='black', point=None, t=None, highlight_he=None, cavity_border=None):
    global fig, ax, poly_collection, point_marker
    fig, ax = plt.subplots()
    ax.set_title("Mesh. Click to add new point...")
    polygons = [[vertices[index] for index in connection] for connection in connections]

    # Create a PolyCollection object
    # poly_collection = PolyCollection(polygons, facecolors='cyan', edgecolors='black', linewidths=1)
    poly_collection = PolyCollection(polygons, facecolors='white', edgecolors=color, linewidths=1)

    # Create the plot
    ax.add_collection(poly_collection)
    fig.canvas.draw()

    try:
        if point.any():
            point_marker = ax.scatter(*point, color='red', zorder=5)
    except:
        if point:
            point_marker = ax.scatter(*point, color='red', zorder=5)

    if highlight_he:
      con = [(t.halfedges[he].vertex, t.halfedges[t.halfedges[he].next].vertex) for he in highlight_he]
      pts = [(v.x, v.y) for v in t.vertices]
      lines = [(pts[start], pts[end]) for start, end in con]
      line_collection = LineCollection(lines, colors='blue', linewidths=2)

      ax.add_collection(line_collection)

    if cavity_border:
      con = [(t.halfedges[he].vertex, t.halfedges[t.halfedges[he].next].vertex) for he in cavity_border]
      pts = [(v.x, v.y) for v in t.vertices]
      lines = [(pts[start], pts[end]) for start, end in con]
      line_collection2 = LineCollection(lines, colors='red', linewidths=2)

      ax.add_collection(line_collection2)


    # Set limits for the plot based on the vertices
    ax.autoscale()
    ax.margins(0.1)
    ax.set_aspect('equal')


    # Display the plot
    # plt.show()

def draw_lines(pts, con):
    """
    Draws lines between points based on a list of points and their connections.

    Parameters:
    pts (list of lists): List of [x, y] coordinates for the points.
    con (list of lists): Each sublist contains two indices of points that form a line.
    """
    global fig, ax, line_collection
    # Create the list of lines using the point coordinates and connections
    lines = [(pts[start], pts[end]) for start, end in con]

    # Create a LineCollection object
    line_collection = LineCollection(lines, colors='blue', linewidths=2)

    # Create the plot

    ax.add_collection(line_collection)


    # Display the plot
    # plt.show()

def draw_halfedges(t, hedges):
  connections = [(t.halfedges[he].vertex, t.halfedges[t.halfedges[he].next].vertex) for he in hedges]
  # print(connections)
  points = [(v.x, v.y) for v in t.vertices]
  # print(points)
  draw_lines(points, connections)

def verts_polys(t):
  # extracts vertices and their connectivity in the triangulation
  verts = [[ver.x, ver.y] for ver in t.vertices]
  polys = [[t.halfedges[face.halfedge].vertex, t.halfedges[t.halfedges[face.halfedge].prev].vertex, t.halfedges[t.halfedges[face.halfedge].next].vertex] for face in t.faces]
  return verts, polys

def get_vertex_coords(halfedge_index, T):
    """Helper function to get vertex coordinates (x, y) from a halfedge index."""
    vertex_index = T.halfedges[halfedge_index].vertex
    vertex = T.vertices[vertex_index]
    return (vertex.x, vertex.y)

def my_incircle(a,b,c,p):
  value = incircle(a,b,c,p)
  if value < 0:
    return False
  elif value > 0:
    return True
  else:
    # print('point exactly on the circumscribec circle, need to patch it somehow')
    return True

def intriangle(triangle, point):
  a1 = orient2d(triangle[0], triangle[1], point)  # all going CCW around facet
  a2 = orient2d(point, triangle[1], triangle[2])
  a3 = orient2d(triangle[2], triangle[0], point)
  return (a1 >= 0 and a2 >= 0 and a3 >= 0) or (a1 <= 0 and a2 <= 0 and a3 <= 0)

def remove_opposites(t, halfedges):
  for he in halfedges[:]:  #itereates over a copy of list because of index shifting
    # print(he, T.halfedges[he].opposite)
    if t.halfedges[he].opposite in halfedges:
        halfedges.remove(t.halfedges[he].opposite)
        halfedges.remove(he)
  return halfedges

def intersect(a,b,c,d) :
    abc = orient2d(a,b,c)
    bad = orient2d(b,a,d)
    if abc*bad < 0 :
        return False
    cdb = orient2d(c,d,b)
    dca = orient2d(d,c,a)
    if cdb*dca < 0 :
        return False
    return True

def find_initria(T, point):
  initria = []
  for face in T.faces:
    # loop through ALL triangles in mesh
    start = face.halfedge  # halfedge index
    face_he_indexes = [start, T.halfedges[start].next, T.halfedges[start].prev]
    face_vert_indexes = [T.halfedges[hhe].vertex for hhe in face_he_indexes]
    v1 = get_vertex_coords(face_he_indexes[0], T)
    v2 = get_vertex_coords(face_he_indexes[1], T)
    v3 = get_vertex_coords(face_he_indexes[2], T)
    # print(v1, v2, v3, point)
    if intriangle((v1, v2, v3), point):
      initria.append(face_he_indexes[0])
      initria.append(face_he_indexes[1])
      initria.append(face_he_indexes[2])
      # print(f'Initria: {face_vert_indexes}')
      return initria

def find_nextria(T, start_face, point):
  initria = []
  trials = 0
  max_trials = 1000
  # print(f'looking for point: {point}')

  start = T.faces[start_face].halfedge  # halfedge index
  face_he_indexes = [start, T.halfedges[start].next, T.halfedges[start].prev]
  face_vert_indexes = [T.halfedges[hhe].vertex for hhe in face_he_indexes]

  while not initria:
      if trials > max_trials:  # Check if trials exceed the limit
          #this is problem for some structured meshes and I don't know why
          print(f'exceeded trials')

          start = T.faces[start_face].halfedge  # halfedge index
          face_he_indexes = [start, T.halfedges[start].next, T.halfedges[start].prev]

          verts, polys = verts_polys(T)
          draw_polygons(verts, polys, 'black', point, T, face_he_indexes)

          # print(f'but i dont understand this')
          break
      a = get_vertex_coords(face_he_indexes[0], T)
      b = get_vertex_coords(face_he_indexes[1], T)
      c = get_vertex_coords(face_he_indexes[2], T)
      # print(v1, v2, v3, point)
      if intriangle((a, b, c), point):
        initria.append(face_he_indexes[0])
        initria.append(face_he_indexes[1])
        initria.append(face_he_indexes[2])
        # print(f'Initria: {face_vert_indexes}')
        # print(f'This took {trials} takes')
        return initria
      else:
        trials += 1
        cog_x = (a[0] + b[0] + c[0]) / 3
        cog_y = (a[1] + b[1] + c[1]) / 3
        cog = (cog_x, cog_y)
        for he in face_he_indexes:
          start = get_vertex_coords(he, T)
          end = get_vertex_coords(T.halfedges[he].next, T)
          if intersect(cog, point, start, end):
            opp = T.halfedges[he].opposite
            facet = T.halfedges[opp].facet
            start = T.faces[facet].halfedge # halfedge index of the opposite face that crosses the path to the point
            face_he_indexes = [start, T.halfedges[start].next, T.halfedges[start].prev]
            break

def contains_permutation(tris_to_delete, target_list):
    target_permutations = set(permutations(target_list))

    for tri in tris_to_delete:
        if tuple(tri) in target_permutations:
            return True
    return False

def he_points(t, he):
  a = t.halfedges[he].vertex
  b = t.halfedges[t.halfedges[he].next].vertex
  return a, b

def orient_cavity(T, cavity):
    # Start with the halfedge that has the minimum index
    start = min(cavity)
    ordered = [start]  # Initialize the list with the starting halfedge
    current = start

    # Keep track of which half-edges we have already ordered
    used = {start}

    # Iterate to follow the halfedge chain
    while len(ordered) < len(cavity):
        next_he = T.halfedges[current].next

        # Find the next halfedge in the cavity that starts where the current halfedge ends
        for he in cavity:
            if he not in used and T.halfedges[he].vertex == T.halfedges[next_he].vertex:
                ordered.append(he)
                used.add(he)
                current = he
                break
        else:
            # If we can't find the next halfedge, the cavity is not fully connected
            break

    return ordered

def find_cavity(T, cavity, point):
    # cavity is a list of halfedges forming the cavity
    added_new = True
    he_to_delete = []
    cavity_tris = []

    # tris_to_delete.append([T.halfedges[cavity[i]].vertex for i in range(len(cavity))])  # add first triangle to be deleted

    while added_new:
        added_new = False
        current_cavity_size = len(cavity)  # Track size of the cavity at the start of the iteration
        # print(f'Cavity size: {current_cavity_size}')

        for he in list(cavity):  # Create a copy of the cavity to safely iterate over it
            opp = T.halfedges[he].opposite
            if opp is None:
                # print(f'boundary edge {he} and {opp}')
                continue

            # ifnd the first halfedge of the face
            # print(f'opp before: {opp}, {he_points(T, opp)}')
            facet = T.halfedges[opp].facet
            opp = T.faces[facet].halfedge
            # print(f'opp after: {opp}, {he_points(T, opp)}')

            face_he_indexes = [opp, T.halfedges[opp].next, T.halfedges[opp].prev]
            # face_vert_indexes = [T.halfedges[hhe].vertex for hhe in face_he_indexes]
            # print(f'face_vert_indexes: {face_vert_indexes}')
            v1 = get_vertex_coords(face_he_indexes[0], T)
            v2 = get_vertex_coords(face_he_indexes[1], T)
            v3 = get_vertex_coords(face_he_indexes[2], T)

            # Check if the point lies inside the incircle of the triangle
            if my_incircle(v1, v2, v3, point):
                added_new = True
                # print(f'New tria: {face_vert_indexes}')
                # Append new half-edges only if they're not already in the cavity
                for index in face_he_indexes:
                    if index not in cavity:
                        cavity.append(index)

                # print(face_vert_indexes)
                # if face_vert_indexes not in tris_to_delete:
                #   tris_to_delete.append(face_vert_indexes)

        # print(f'Cavity before remove opposites: {cavity}')
        for he in cavity[:]:  # itereates over a copy of cavity because of index shifting
            # print(he, T.halfedges[he].opposite)
            if T.halfedges[he].opposite in cavity and he not in he_to_delete:
                cavity.remove(T.halfedges[he].opposite)
                cavity.remove(he)
                he_to_delete.append(T.halfedges[he].opposite)
                he_to_delete.append(he)
            if T.halfedges[he].facet not in cavity_tris:
                cavity_tris.append(T.halfedges[he].facet)

        # print(f'Cavity after remove opposites and before orient: {cavity}')
        cavity = orient_cavity(T, cavity)
        # print(f'Cavity after orient: {cavity}')

        # If no new half-edges were added in this iteration, break out of the loop
        if len(cavity) == current_cavity_size:
            break

    return cavity, he_to_delete, cavity_tris

def triangulate(T, point_cloud):

    for j, point in enumerate(point_cloud):
        # print(f'{10 * "="}   POINT {j}   {10 * "="}')
        if j == 0:
            initria = find_initria(T, point)
            # print(initria)
        else:
            initria = find_nextria(T, starting_face, point)

        cavity, reuse_he, reuse_tris = find_cavity(T, initria, point)

        # print(f'cavity: {cavity}')
        # verts, polys = verts_polys(T)
        # draw_polygons(verts, polys, 'black', point, T, cavity)

        new_point_index = j + 4
        T.vertices.append(Vertex(point[0], point[1], new_point_index, None))

        new_he_index = 6 * (j + 1)
        new_face_index = 2 * (j + 1)
        starting_face = new_face_index + 1

        for i, he in enumerate(cavity):
            try:
                # try to reuse triange index
                T.faces[reuse_tris[i]].halfedge = he
                T.halfedges[he].facet = reuse_tris[i]
                # print(f'Alpha {i} --- face: {reuse_tris[i]} has halfedge {he} now')
                try:
                    # try to reuse halfedges index
                    # print(f'try to reuse halfedges index {reuse_he[2*i]} and {reuse_he[2*i + 1]}')
                    T.halfedges[reuse_he[2 * i]].opposite = None
                    T.halfedges[reuse_he[2 * i]].prev = he
                    T.halfedges[reuse_he[2 * i]].next = reuse_he[2 * i + 1]
                    T.halfedges[reuse_he[2 * i]].vertex = T.halfedges[cavity[i + 1]].vertex
                    T.halfedges[reuse_he[2 * i]].facet = reuse_tris[i]

                    T.halfedges[reuse_he[2 * i + 1]].opposite = None
                    T.halfedges[reuse_he[2 * i + 1]].prev = reuse_he[2 * i]
                    T.halfedges[reuse_he[2 * i + 1]].next = he
                    T.halfedges[reuse_he[2 * i + 1]].vertex = new_point_index
                    T.halfedges[reuse_he[2 * i + 1]].facet = reuse_tris[i]

                    T.halfedges[he].next = reuse_he[2 * i]
                    T.halfedges[he].prev = reuse_he[2 * i + 1]

                    if i > 0:
                        # skips the first triangle
                        T.halfedges[reuse_he[2 * i + 1]].opposite = reuse_he[2 * i - 2]
                        T.halfedges[reuse_he[2 * i - 2]].opposite = reuse_he[2 * i + 1]

                except:
                    # if no hedge indexel left to recycle
                    # print(f'T-3 adding halfedges {new_he_index} and {new_he_index + 1}')

                    T.halfedges[he].next = new_he_index
                    T.halfedges[he].prev = new_he_index + 1

                    T.halfedges.append(Halfedge(
                        next=new_he_index + 1,
                        opposite=new_he_index + 3,  # was None
                        prev=he,
                        vertex=T.halfedges[cavity[i + 1]].vertex,
                        facet=reuse_tris[i],
                        index=new_he_index
                    ))

                    if len(reuse_he) == 0:
                        # print(f'just one triangle forming the cavity')
                        T.halfedges.append(Halfedge(
                            next=he,
                            opposite=new_he_index + 4,
                            prev=new_he_index,
                            vertex=new_point_index,
                            facet=reuse_tris[i],
                            index=new_he_index + 1
                        ))
                        T.halfedges[new_he_index + 4].opposite = new_he_index + 1
                    else:
                        # print(f'many triangles forming the cavity')
                        T.halfedges.append(Halfedge(
                            next=he,
                            opposite=reuse_he[2 * i - 2],
                            prev=new_he_index,
                            vertex=new_point_index,
                            facet=reuse_tris[i],
                            index=new_he_index + 1
                        ))
                        T.halfedges[reuse_he[2 * i - 2]].opposite = new_he_index + 1



            except:
                # add last two facets
                if i == len(cavity) - 2:
                    # if it's second to last triangle
                    # print(f'T-2 adding halfedges {new_he_index+2} and {new_he_index + 3} ')
                    T.faces.append(Face(
                        halfedge=he,
                        index=new_face_index
                    ))
                    T.halfedges[he].facet = new_face_index
                    T.halfedges.append(Halfedge(
                        next=new_he_index + 3,
                        opposite=None,
                        prev=he,
                        vertex=T.halfedges[cavity[i + 1]].vertex,
                        facet=new_face_index,
                        index=new_he_index + 2
                    ))
                    T.halfedges.append(Halfedge(
                        next=he,
                        opposite=new_he_index,
                        prev=new_he_index + 2,
                        vertex=new_point_index,
                        facet=new_face_index,
                        index=new_he_index + 3
                    ))
                    T.halfedges[new_he_index].opposite = new_he_index + 3

                    T.halfedges[he].next = new_he_index + 2
                    T.halfedges[he].prev = new_he_index + 3
                if i == len(cavity) - 1:
                    # if it's the last triangle
                    # print(f'T-1 last triangle adding halfedges {new_he_index + 4} and {new_he_index + 5} ')
                    new_face_index += 1
                    T.faces.append(Face(
                        halfedge=he,
                        index=new_face_index
                    ))
                    T.halfedges[he].facet = new_face_index

                    T.halfedges[new_he_index + 2].opposite = new_he_index + 5
                    T.halfedges[he].next = new_he_index + 4
                    T.halfedges[he].prev = new_he_index + 5

                    if len(reuse_tris) == 1:
                        # print(f'only one reuse tri')
                        T.halfedges.append(Halfedge(
                            next=new_he_index + 5,
                            opposite=new_he_index + 1,
                            prev=he,
                            vertex=T.halfedges[cavity[0]].vertex,
                            facet=new_face_index,
                            index=new_he_index + 4
                        ))
                        T.halfedges[new_he_index + 1].opposite = new_he_index + 4
                    else:
                        # print(f'more reuse tri')
                        T.halfedges.append(Halfedge(
                            next=new_he_index + 5,
                            opposite=reuse_he[1],
                            prev=he,
                            vertex=T.halfedges[cavity[0]].vertex,
                            facet=new_face_index,
                            index=new_he_index + 4
                        ))
                        T.halfedges[reuse_he[1]].opposite = new_he_index + 4

                    T.halfedges.append(Halfedge(
                        next=he,
                        opposite=new_he_index + 2,
                        prev=new_he_index + 4,
                        vertex=new_point_index,
                        facet=new_face_index,
                        index=new_he_index + 5
                    ))

def triangulate_single(T, point, starting_face):
    initria = find_nextria(T, starting_face, point)
    cavity, reuse_he, reuse_tris = find_cavity(T, initria, point)

    last_he_index = max(edge.index for edge in T.halfedges if edge.index is not None)
    last_face_index = max(face.index for face in T.faces if face.index is not None)
    last_point_index = max(point.index for point in T.vertices if point.index is not None)

    new_point_index = last_point_index + 1
    T.vertices.append(Vertex(point[0], point[1], new_point_index, None))
    new_he_index = last_he_index + 1
    new_face_index = last_face_index + 1


    for i, he in enumerate(cavity):
        try:
            # try to reuse triange index
            T.faces[reuse_tris[i]].halfedge = he
            T.halfedges[he].facet = reuse_tris[i]
            # print(f'Alpha {i} --- face: {reuse_tris[i]} has halfedge {he} now')
            try:
                # try to reuse halfedges index
                # print(f'try to reuse halfedges index {reuse_he[2*i]} and {reuse_he[2*i + 1]}')
                T.halfedges[reuse_he[2 * i]].opposite = None
                T.halfedges[reuse_he[2 * i]].prev = he
                T.halfedges[reuse_he[2 * i]].next = reuse_he[2 * i + 1]
                T.halfedges[reuse_he[2 * i]].vertex = T.halfedges[cavity[i + 1]].vertex
                T.halfedges[reuse_he[2 * i]].facet = reuse_tris[i]

                T.halfedges[reuse_he[2 * i + 1]].opposite = None
                T.halfedges[reuse_he[2 * i + 1]].prev = reuse_he[2 * i]
                T.halfedges[reuse_he[2 * i + 1]].next = he
                T.halfedges[reuse_he[2 * i + 1]].vertex = new_point_index
                T.halfedges[reuse_he[2 * i + 1]].facet = reuse_tris[i]

                T.halfedges[he].next = reuse_he[2 * i]
                T.halfedges[he].prev = reuse_he[2 * i + 1]

                if i > 0:
                    # skips the first triangle
                    T.halfedges[reuse_he[2 * i + 1]].opposite = reuse_he[2 * i - 2]
                    T.halfedges[reuse_he[2 * i - 2]].opposite = reuse_he[2 * i + 1]

            except:
                # if no hedge indexel left to recycle
                # print(f'T-3 adding halfedges {new_he_index} and {new_he_index + 1}')

                T.halfedges[he].next = new_he_index
                T.halfedges[he].prev = new_he_index + 1

                T.halfedges.append(Halfedge(
                    next=new_he_index + 1,
                    opposite=new_he_index + 3,  # was None
                    prev=he,
                    vertex=T.halfedges[cavity[i + 1]].vertex,
                    facet=reuse_tris[i],
                    index=new_he_index
                ))

                if len(reuse_he) == 0:
                    # print(f'just one triangle forming the cavity')
                    T.halfedges.append(Halfedge(
                        next=he,
                        opposite=new_he_index + 4,
                        prev=new_he_index,
                        vertex=new_point_index,
                        facet=reuse_tris[i],
                        index=new_he_index + 1
                    ))
                    T.halfedges[new_he_index + 4].opposite = new_he_index + 1
                else:
                    # print(f'many triangles forming the cavity')
                    T.halfedges.append(Halfedge(
                        next=he,
                        opposite=reuse_he[2 * i - 2],
                        prev=new_he_index,
                        vertex=new_point_index,
                        facet=reuse_tris[i],
                        index=new_he_index + 1
                    ))
                    T.halfedges[reuse_he[2 * i - 2]].opposite = new_he_index + 1



        except:
            # add last two facets
            if i == len(cavity) - 2:
                # if it's second to last triangle
                # print(f'T-2 adding halfedges {new_he_index+2} and {new_he_index + 3} ')
                T.faces.append(Face(
                    halfedge=he,
                    index=new_face_index
                ))
                T.halfedges[he].facet = new_face_index
                T.halfedges.append(Halfedge(
                    next=new_he_index + 3,
                    opposite=None,
                    prev=he,
                    vertex=T.halfedges[cavity[i + 1]].vertex,
                    facet=new_face_index,
                    index=new_he_index + 2
                ))
                T.halfedges.append(Halfedge(
                    next=he,
                    opposite=new_he_index,
                    prev=new_he_index + 2,
                    vertex=new_point_index,
                    facet=new_face_index,
                    index=new_he_index + 3
                ))
                T.halfedges[new_he_index].opposite = new_he_index + 3

                T.halfedges[he].next = new_he_index + 2
                T.halfedges[he].prev = new_he_index + 3
            if i == len(cavity) - 1:
                # if it's the last triangle
                # print(f'T-1 last triangle adding halfedges {new_he_index + 4} and {new_he_index + 5} ')
                new_face_index += 1
                T.faces.append(Face(
                    halfedge=he,
                    index=new_face_index
                ))
                T.halfedges[he].facet = new_face_index

                T.halfedges[new_he_index + 2].opposite = new_he_index + 5
                T.halfedges[he].next = new_he_index + 4
                T.halfedges[he].prev = new_he_index + 5

                if len(reuse_tris) == 1:
                    # print(f'only one reuse tri')
                    T.halfedges.append(Halfedge(
                        next=new_he_index + 5,
                        opposite=new_he_index + 1,
                        prev=he,
                        vertex=T.halfedges[cavity[0]].vertex,
                        facet=new_face_index,
                        index=new_he_index + 4
                    ))
                    T.halfedges[new_he_index + 1].opposite = new_he_index + 4
                else:
                    # print(f'more reuse tri')
                    T.halfedges.append(Halfedge(
                        next=new_he_index + 5,
                        opposite=reuse_he[1],
                        prev=he,
                        vertex=T.halfedges[cavity[0]].vertex,
                        facet=new_face_index,
                        index=new_he_index + 4
                    ))
                    T.halfedges[reuse_he[1]].opposite = new_he_index + 4

                T.halfedges.append(Halfedge(
                    next=he,
                    opposite=new_he_index + 2,
                    prev=new_he_index + 4,
                    vertex=new_point_index,
                    facet=new_face_index,
                    index=new_he_index + 5
                ))

    return cavity

def save_file(T, filename):
    triangles = []
    for face in T.faces:
        start = face.halfedge  # halfedge index
        face_he_indexes = [start, T.halfedges[start].next, T.halfedges[start].prev]
        face_vert_indexes = [T.halfedges[he].vertex for he in face_he_indexes]
        triangles.append(face_vert_indexes)

    if filename is None:
        filename = 'triangles.dat'

    with open(filename, 'w') as f:
        f.write(f'{len(triangles)}\n')
        # print(f'Saved triangles: {len(triangles)}')
        for tria in triangles:
            f.write(f'{tria[0]} {tria[1]} {tria[2]}\n')

def on_click(event, T):
    global line_collection
    try:
        line_collection.remove()
        point_marker.remove()
    except:
        pass

    if event.inaxes:
        x, y = event.xdata, event.ydata

        new_marker = ax.plot(x, y, 'ro')[0]  # 'ro' is a red circle marker, get the artist object

        cavity = triangulate_single(T, (x, y), 0)

        draw_halfedges(T, cavity)

        vertices, connections = verts_polys(T)

        polygons = [[vertices[index] for index in connection] for connection in connections]
        poly_collection.set_verts(polygons)


        # draw_polygons(verts, polys, 'black')
        fig.canvas.draw()


def main():
    parser = argparse.ArgumentParser(
        description="Homework 1: Delaunay Triangulation [Yi Ru, Enjy Katary, Hancikovsky Matus]")

    parser.add_argument(
        '-i', '--input',
        type=str,
        # required=True,
        help='Input file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file. triangles.dat by default.'
    )

    parser.add_argument('-si', '--showinteractive', help='Show complete interactive mesh.'
                                                         ' Click anywhere on the mest to add a new point. '
                                                         'Algorithm shows cavity border and remeshes.',
                        action='store_true')

    args = parser.parse_args()
    # args.input, args.output

    points, xlim, ylim = read_input(str(args.input), False)

    verts = [[xlim[0], ylim[0]], [xlim[1], ylim[0]], [xlim[1], ylim[1]], [xlim[0], ylim[1]]]
    tris = [[0, 1, 2], [2, 3, 0]]

    T = TriangularMesh(verts, tris)

    sorted_points = space_filling_sort(points)

    triangulate(T, sorted_points)

    save_file(T, args.output)

    if args.showinteractive:
        verts, polys = verts_polys(T)
        draw_polygons(verts, polys, 'black')

        add_point_and_triangulate = partial(on_click, T=T)

        fig.canvas.mpl_connect('button_press_event', add_point_and_triangulate)
        plt.show()


if __name__ == "__main__":
    main()




