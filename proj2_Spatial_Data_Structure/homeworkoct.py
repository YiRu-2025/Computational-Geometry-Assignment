from contact import Contact
import numpy as np
from typing import List

MAX_OBJECTS = 8  # max obj
MIN_SIZE = 2.0 # min size of before stopping subdivision
MAX_REBUILD = 60 # rebuild frequency
class OctreeNode:#items of the tree to optimize memory usage
    __slots__ = ['center', 'size', 'children', 'objects'] #define class inputs

    def __init__(self, center: np.ndarray, size: float):
        self.center = center
        self.size = size
        self.children = None  # subtrees
        self.objects = []


class Octree:

    def __init__(self, domain_size: float = 20.0):
        self.root = OctreeNode(np.zeros(3), domain_size) # creating the oct tree
# bit-wise or

    def insert(self, position: np.ndarray, index: int, node: OctreeNode = None):
        if node is None:
            node = self.root # start at the root

        if node.children is None: # for final search with no subtree
            if len(node.objects) < MAX_OBJECTS or node.size <= MIN_SIZE:
                node.objects.append(index) # stop here
                return

            self.subdivide(node)

            old_objects = node.objects
            node.objects = []
            for old in old_objects:
                self.insert_subdivided(position, old, node)

            self.insert_subdivided(position, index, node) # insert new object

        else:
            self.insert_subdivided(position, index, node) # continue to child node



    def insert_subdivided(self, position: np.ndarray, index: int, node: OctreeNode): #bottom-up method, inserts in leaf node with no subtrees

        if node.children is None: # if there is no children
            node.objects.append(index)
            return

        octant = 0
        if position[0] > node.center[0]: octant |= 1
        if position[1] > node.center[1]: octant |= 2
        if position[2] > node.center[2]: octant |= 4

        self.insert(position, index, node.children[octant]) # insert into correct child


    def subdivide(self, node: OctreeNode):
        half = node.size / 2 # split the node into children
        node.children = []

        for i in range(MAX_OBJECTS):
            offset = np.array([
                half / 2 if i & 1 else -half / 2,
                half / 2 if i & 2 else -half / 2,
                half / 2 if i & 4 else -half / 2
            ])
            child = OctreeNode(node.center + offset, half) #create child node
            node.children.append(child)



class Homeworkoct:
    def __init__(self, simulator):
        self.octree = None  # initialize
        self.rebuild_counter = 0


    def object_added(self, simulator, i):

        if self.octree is None:
            self.octree = Octree()  # create octree if there is non

        self.octree.insert(simulator.positions[i], i) # add


    def find_intersections(self, simulator) -> List[Contact]:


        if len(simulator.radii) == 0:
            return [] # no intersecctions if no objects

        self.rebuild_counter += 1
        if self.rebuild_counter >= MAX_REBUILD:
            self.octree = Octree()
            for i in range(len(simulator.positions)):
                self.octree.insert(simulator.positions[i], i)
            self.rebuild_counter = 0

        return self._find_collisions(simulator, self.octree.root)

    def _find_collisions(self, simulator, node: OctreeNode) -> List[Contact]:
        contacts = []  # store detected collisions

        if node.objects: # check for collisions within this node
            n = len(node.objects)
            for i in range(n):
                for j in range(i + 1, n):
                    contact = simulator.intersect(node.objects[i], node.objects[j])
                    if contact is not None:
                        contacts.append(contact)

        if node.children:
            for child in node.children:
                contacts.extend(self._find_collisions(simulator, child))

        return contacts



