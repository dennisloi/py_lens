import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class ray:
    def __init__(self, strenght : float, origin : tuple, dir : tuple, color = 'g'):
        self.strength = strenght
        self.origin = origin
        self.dir = dir
        self.color = color
        self.end_point = None
        self.length = None

    @classmethod
    def from_ray(cls, r, dir, attenuation = 0):
        if r.end_point is None:
            print("Previous ray must be ended before creating a new one")
            return None
        return cls(r.strength * attenuation, r.end_point, dir, r.color)

    def end(self, end_point):
        self.end_point = end_point
        self.length = np.linalg.norm(np.array(self.origin) - np.array(self.end_point))

    def plot(self, plt, color = 'k', alpha = 1, arrow_type = 'small'):
        if self.end_point is not None:
            plt.plot([self.origin[0], self.end_point[0]], [self.origin[1], self.end_point[1]], color=color, alpha=alpha)
        #draw a small arrow on the origin

        if self.length is None or arrow_type == 'small':
            length = 0.1
        else:
            length = self.length / 2
        
        plt.arrow(self.origin[0], self.origin[1], length*self.dir[0], length*self.dir[1], head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=alpha)
        return
        
    
    def __str__(self):
        return f"Strenght: {self.strength}, Origin: {self.origin}, Direction: {self.dir}, End: {self.end_point}, Length: {self.length}"
    
class lens:
    def __init__(self, x : list, y : list, dx : list, refidx_l : float, refidx_r : float, refl_coeff : float, trans_coeff : float):
        
        self.x = x
        self.dx = dx
        self.y = y

        self.refidx_l = refidx_l
        self.refidx_r = refidx_r

        self.refl_coeff = refl_coeff
        self.trans_coeff = trans_coeff

        self.segments = self.calculate_segments()
        self.points = self.calculate_points()

    def calculate_points(self):
        points = []
        for i in range(len(self.x)):
            points.append((self.x[i], self.y[i]))
        return points

    def calculate_segments(self):
        segments = []
        for i in range(len(self.x)-1):
            segments.append(((self.x[i], self.y[i]), (self.x[i+1], self.y[i+1])))
        return segments

    def intersect (self, r):

        def intersect_ray_segment(ray, segment):
            # Ray
            o1 = np.array(ray.origin)
            d1 = np.array(ray.dir)
            # Segment
            o2 = np.array(segment[0])
            d2 = np.array(segment[1]) - np.array(segment[0])
            
            # Calculate intersection point
            cross = np.cross(d1, d2)
            if cross == 0:
                return None
            t1 = np.cross(o2 - o1, d2) / cross
            t2 = np.cross(o2 - o1, d1) / cross
            if t1 >= 0 and t2 >= 0 and t2 <= 1:
                return o1 + t1 * d1
            return None
            
        def find_ray_polygon_intersection(r):

            intersection = None
            intersected_segment = None

            # Nudge the ray origin to avoid self-intersection
            r.origin = (r.origin[0] + 1e-6 * r.dir[0], r.origin[1] + 1e-6 * r.dir[1])

            for segment in self.segments:
                intersection = intersect_ray_segment(r, segment)
                if intersection is not None:
                    intersected_segment = segment
                    break
            
            return segment, intersection
        
        def get_dx(segment, intersection):
            # Calculates the distance between the two points of the intersected segment
            # Then interpolates the value of dx at the intersection point
            p1 = np.array(segment[0])
            p2 = np.array(segment[1])

            # Distances between the intersection point and the segment points
            d1 = np.linalg.norm(p1 - intersection)
            d2 = np.linalg.norm(p2 - intersection)

            # Interpolated value of dx over the segment
            dx = self.dx[self.segments.index(segment)] * (d2 / (d1 + d2)) + self.dx[self.segments.index(segment) + 1] * (d1 / (d1 + d2))

            return dx
        
        segment, intersection = find_ray_polygon_intersection(r)
        
        if intersection is not None:
        
            # Incoming ray
            t0 = np.arctan2(r.dir[1], r.dir[0])

            # Segment normal
            t1 = np.arctan2(get_dx(segment, intersection), 1)

            # snell angle
            t2 = np.arcsin(self.refidx_l * np.sin(t1) / self.refidx_r)

            # transmitted ray
            t3 = t0 - t1 + t2

            # Transmitted ray direction
            transm_dir = (np.cos(t3), np.sin(t3))

            # Reflected ray direction
            refl_dir = (-np.cos(t0 - 2 * t1), -np.sin(t0 - 2 * t1))

            r.end(intersection)

            transmitted_ray = ray.from_ray(r, transm_dir, self.trans_coeff)
            reflected_ray = ray.from_ray(r, refl_dir, self.refl_coeff)

            return reflected_ray, transmitted_ray
        else:
            return None, None
        

y_values = np.linspace(-1, 1, 100)

x_values = +0.4*y_values**2

dx = +2 * 0.4* y_values

x_values1 = -0.6*y_values**2

dx1 = -2 * 0.6* y_values

startlense = lens(x = [-2, -2], y = [-2, 2], dx = [0, 0], refidx_l = 1, refidx_r = 1, refl_coeff = 0, trans_coeff=1)

lens0 = lens(x = x_values - 0.5, y = y_values, dx = dx, refidx_l = 1, refidx_r = 1.5, refl_coeff = 0.25, trans_coeff=0.6)

lens1 = lens(x = x_values1 + 1, y = y_values, dx = dx1, refidx_l = 1.5, refidx_r = 1, refl_coeff = 0, trans_coeff=0.6)

endlense = lens(x = [2, 2], y = [-2, 2], dx = [0, 0], refidx_l = 1, refidx_r = 1, refl_coeff = 0, trans_coeff=1)

rays = [[], [], [], []]

for i in np.linspace(-1,1 , 19):
    # Random matplotlib color

    color = np.random.rand(3,)

    rays[0].append(ray(strenght=1, origin=(-1.5, i), dir=(1, 0), color=color))

plt.figure()

for el in startlense.segments:
    plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], alpha=0.5, color='black')

for el in lens0.segments:
    plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], alpha=0.5, color='black')

for el in lens1.segments:
    plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], alpha=0.5, color='black')

for el in endlense.segments:
    plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], alpha=0.5, color='black')

#for point in lens0.points:
#    plt.plot(point[0], point[1], 'ro', alpha=0.5)

for r in rays[0]:
    r1, r2 = lens0.intersect(r)
    rays[1].append(r1)
    rays[2].append(r2)

# Back reflections termination
for r in rays[1]:
    if r is not None:
        r1, r2 = startlense.intersect(r)

# Transmitted rays, to the second lense
for r in rays[2]:
    if r is not None:
        r1, r2 = lens1.intersect(r)
        rays[3].append(r2)

# Transmitted rays termination
for r in rays[3]:
    if r is not None:
        r1, r2 = endlense.intersect(r)

for r in rays[0]:
    if r is not None:
        r.plot(plt, color=r.color, alpha=np.clip(r.strength, 0, 1))

for r in rays[1]:
    if r is not None:
        r.plot(plt, color=r.color, alpha=np.clip(r.strength, 0, 1))

for r in rays[2]:
    if r is not None:
        r.plot(plt, color=r.color, alpha=np.clip(r.strength, 0, 1))

for r in rays[3]:
    if r is not None:
        r.plot(plt, color=r.color, alpha=np.clip(r.strength, 0, 1))

plt.ylim(-1, 1)
plt.xlim(-1.5, 2)

plt.show()