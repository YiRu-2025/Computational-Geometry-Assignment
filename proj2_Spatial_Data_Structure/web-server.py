import numpy as np
import asyncio
from websockets.asyncio.server import serve
import json
import math
from numpy.typing import ArrayLike
from typing import Optional
from contact import Contact
from dataclasses import dataclass

from homeworkoct import Homeworkoct
from homework import Homework
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from threading import Thread

TIMESTEP = 1.0 / 60.0

@dataclass
#creating instances
class Ball:
    """Information about a sphere."""
    #fields

    pos: ArrayLike #x,y,z
    velocity: ArrayLike # x,y,z vector
    radius: float # mogrd rakam
    rotation: ArrayLike # a +b i + c j + d k (rotation values a b c d)
    angular_velocity: ArrayLike
    index: Optional[int]
    #functions
    def intersect_wall(self, pos: ArrayLike):
        """
        Check intersection with a wall.

        pos is the projection of the ball onto the plane containing the wall.
        """
        other = Ball(pos, np.zeros(3, dtype='float32'), 1.0,   #defining wall as a ball
                     np.array([0.0, 0.0, 0.0, 1.0], dtype='float32'),
                     np.zeros(3, dtype='float32'), None)
        return self.intersect(other)

    def intersect(self, other) -> Optional[Contact]:  #intersection of the ball with another ball, self is the main ball
        """Check if two balls intersect."""
        combined_radii = self.radius + other.radius # addition of two radii
        delta_pos = self.pos - other.pos
        squared_dist = np.dot(delta_pos, delta_pos)

        if squared_dist >= combined_radii * combined_radii: # if ( sqaured_dist < comb*combi ) then intersection
            return None

        dist = math.sqrt(squared_dist)
        fc = 1e4 * (combined_radii / dist - 1.0)

        delta_vel = self.velocity - other.velocity
        h = np.dot(delta_pos, delta_vel) / squared_dist

        fc = max(fc - 5.0 * h, 0.0)
        p = (self.radius * self.angular_velocity + \
             other.radius * other.angular_velocity) / \
             combined_radii

        delta_vel -= h * delta_pos + np.cross(p, delta_pos)
        ft = min(0.5, 0.05 * abs(fc) * dist / max(1e-3, np.linalg.norm(delta_vel)))

        torque = (ft / dist) * np.cross(delta_pos, delta_vel)

        return Contact(self.index, other.index,
                       fc * delta_pos - ft * delta_vel,
                       torque)

class RigidbodySimulation:
    """Simulation of spheres."""

    def __init__(self): # constructor 
        """Create an empty simulation."""
        self.positions: ArrayLike = np.zeros((0, 3), order='F')
        self.rotations: ArrayLike = np.zeros((0, 4), order='F')
        self.velocities: ArrayLike =  np.zeros((0, 3), order='F')
        self.angular_velocities: ArrayLike = np.zeros((0, 3), order='F')
        self.radii: ArrayLike = np.array([], order='F', dtype="float32")

        self.elapsed_time = 0

        #self.user_data = Homework(self)
        self.user_data = Homeworkoct(self)

    def add_objects(self, positions, rotations, scales):  # Adds new spheres to the simulation:
        """Add more spheres to the simulation."""
        pos_array = np.array(positions, order='F', dtype='float32')
        rot_array = np.array(rotations, order='F', dtype='float32')

        velocities = np.zeros((len(scales), 3), dtype='float32', order='F')
        angular_velocities = np.zeros((len(scales), 3), dtype='float32', order='F')

        scales_array = np.array(scales, order='F', dtype='float32')

        if len(self.radii) == 0:
            self.positions = pos_array
            self.rotations = rot_array
            self.velocities = velocities
            self.angular_velocities = angular_velocities
            self.radii = scales_array
        else:
            self.positions = np.concat((self.positions, pos_array))
            self.rotations = np.concat((self.rotations, rot_array))
            self.velocities = np.concat((self.velocities, velocities))
            self.angular_velocities = np.concat((self.angular_velocities, angular_velocities))
            self.radii = np.concat((self.radii, scales_array))

        for i in reversed(range(len(scales))):
            self.user_data.object_added(self, len(self.radii) - 1 - i)

    def update(self, dt):   #Advances the simulation by a time step
        """Simulate rigidbody objects for dt seconds. Actual updates only occur at fixed interval."""
        self.elapsed_time += dt
        i = 0

        MAX_STEPS = 4
        while self.elapsed_time >= TIMESTEP:
            self.step()
            self.elapsed_time -= TIMESTEP
            i += 1
            if i >= MAX_STEPS: break

    def find_intersections(self):  # Relies on user_data to find points where spheres collide or intersect.
        """Return all contact points within the simulation."""
        return self.user_data.find_intersections(self)

    def ball(self, i: int) -> Ball:
        """Return all data relative to ball at index i."""
        return Ball(self.positions[i], self.velocities[i],
                    self.radii[i], self.rotations[i], self.angular_velocities[i],
                    i)

    def intersect(self, i: int, j: int):
        """Return a Contact object if two spheres intersect. None otherwise."""
        return self.ball(i).intersect(self.ball(j))

    def apply_contact(self, contact: Contact): # Resolves a collision by applying forces and torques to the velocities and angular velocities of the spheres based on their masses and radii.
        """Apply contact forces to resolve a contact."""  
        i, j = contact.obj_a, contact.obj_b
        m_a = self.radii[i] * self.radii[i]
        self.velocities[i] += TIMESTEP * contact.force / m_a
        self.angular_velocities[i] += TIMESTEP * contact.torque / (0.4 * m_a * self.radii[i])

        if j:
            m_b = self.radii[j] * self.radii[j]
            self.velocities[j] -= TIMESTEP * contact.force / m_b
            self.angular_velocities[j] -= TIMESTEP * contact.torque / (0.4 * m_b * self.radii[j])

    def step(self):
        """Simulate objects for TIMESTEP seconds."""
        if len(self.radii) == 0:
            return

        for i in range(len(self.radii)):
            ball = self.ball(i)
            domain_size = 10.0
            walls = np.array([
                [-domain_size - 1.0, ball.pos[1], ball.pos[2]],
                [+domain_size + 1.0, ball.pos[1], ball.pos[2]],
                [ball.pos[0], -domain_size - 1.0, ball.pos[2]],
                [ball.pos[0], +domain_size + 1.0, ball.pos[2]],
                [ball.pos[0], ball.pos[1], -domain_size - 1.0],
                [ball.pos[0], ball.pos[1], +domain_size + 1.0],
            ], dtype='float32')

            for wall in walls:
                contact = ball.intersect_wall(wall)
                if contact:
                    self.apply_contact(contact)

        for contact in self.find_intersections():
            self.apply_contact(contact)

        gravity = np.array([0, -9.81, 0])

        self.positions += TIMESTEP * self.velocities
        self.velocities += TIMESTEP * gravity

        drotation = 0.5 * np.einsum('ij,oj->oi',
                                    np.array([[0.0, 0.0, 0.0],
                                              [1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0],
                                              [0.0, 0.0, 1.0]]),
                                    self.angular_velocities)

        weights_a = np.array([[[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0]],
                              [[+1.0, 0.0, 0.0, 0.0], [0.0, +1.0, 0.0, 0.0], [0.0, 0.0, +1.0, 0.0], [0.0, 0.0, 0.0, -1.0]],
                              [[+1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, +1.0, 0.0], [0.0, 0.0, 0.0, +1.0]],
                              [[+1.0, 0.0, 0.0, 0.0], [0.0, +1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0]]])

        weights_b = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                              [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
                              [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])

        self.rotations += TIMESTEP * \
            np.einsum('qij,qij->qi',
                      np.einsum('ijk,qk->qij', weights_a, drotation),
                      np.einsum('ijk,qk->qij', weights_b, self.rotations))

        inv_norms = 1.0 / np.linalg.norm(self.rotations, axis=1)
        self.rotations = np.einsum('i,ij->ij', inv_norms, self.rotations)

async def serve_simulation(socket):
    """Start a simulation and communicate updates to the client."""
    colliders = {}
    registered_objects = {}

    simulation = RigidbodySimulation()
    prev_time = None

    async for message in socket:
        obj = json.loads(message)
        match obj["command"]:
            case "add-objects":
                simulation.add_objects(obj["positions"], obj["rotations"], obj["scales"])

            case "step":
                if prev_time is not None:
                    simulation.update(obj["timestamp"] - prev_time)
                prev_time = obj["timestamp"]

                await socket.send(simulation.positions.tobytes('C'))
                await socket.send(simulation.rotations.tobytes('C'))

async def main():
    """Wait for client to connect using WebSockets, and let them interact with a rigidbody simulation."""
    async with serve(serve_simulation, "localhost", 9345):
        await asyncio.get_running_loop().create_future()

def run_web_server(port):
    """Spawn an HTTP server for the browser to connect to."""
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)
    httpd.serve_forever()

thread = Thread(target=run_web_server, args=[8000])
thread.start()
asyncio.run(main())
thread.join()
