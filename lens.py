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
        else:
        

            if self.length is None or arrow_type == 'small':
                length = 0.1
            else:
                length = self.length / 2
            
            #draw a small arrow on the origin
            plt.arrow(self.origin[0], self.origin[1], length*self.dir[0], length*self.dir[1], head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=alpha)
        return
        
    
    def __str__(self):
        return f"Strenght: {self.strength}, Origin: {self.origin}, Direction: {self.dir}, End: {self.end_point}, Length: {self.length}"
    
class lens:
    def __init__(self, x : list, y : list, dx : list, refidx_l : float, refidx_r : float, refl_coeff : float, trans_coeff : float):
        
        self.x = np.array(x)
        self.dx = np.array(dx)
        self.y = np.array(y)

        self.refidx_l = refidx_l
        self.refidx_r = refidx_r

        self.refl_coeff = refl_coeff
        self.trans_coeff = trans_coeff

        self.update()

    def update(self):
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
    
    def plot(self, plt):
        for el in self.segments:
            plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], alpha=0.5, color='black')
        return

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

            # 
            t1 = np.arctan2(get_dx(segment, intersection), 1)

            # snell angle
            t2 = np.arcsin(self.refidx_l * np.sin(t0 + t1) / self.refidx_r)

            # transmitted ray
            t3 =  t2 - t1

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
        
class lens_system:
    def __init__(self, position):
        self.position = position
        self.lenses = []
        self.boundaries = []

    def add_lens(self, lens, offset = (0, 0)):
        lens.x += offset[0]
        lens.y += offset[1]
        lens.update()
        self.lenses.append(lens)

    def add_boundary(self, refl_coeff = 0):
        # Get the miminum x value of the first lens
        x_min = np.min(self.lenses[0].x)

        # Get the maximum x value of the last lens
        x_max = np.max(self.lenses[-1].x)

        l_top = lens(x=[x_min, x_max], y=[1,1], dx=[np.inf, np.inf], refidx_l=1, refidx_r=1, refl_coeff = refl_coeff, trans_coeff=0)
        l_bottom = lens(x=[x_min, x_max], y=[-1,-1], dx=[-np.inf, -np.inf], refidx_l=1, refidx_r=1, refl_coeff = refl_coeff, trans_coeff=0)
                        
        self.boundaries.append(l_top)
        self.boundaries.append(l_bottom)

    def intersect(self, rays, reflection_limit = 1):

        rays_list = [rays]

        for lens in self.lenses:
            new_rays = []
            for r in rays_list[-1]:
                reflected, transmitted = lens.intersect(r)
                if transmitted is not None:
                    new_rays.append(transmitted)
                else:
                    for boundary in self.boundaries:
                        _, _ = boundary.intersect(r)

            rays_list.append(new_rays)

        return rays_list
        
    
    def plot(self, plt):
        for lens in self.lenses:
            lens.plot(plt)
        for boundary in self.boundaries:
            boundary.plot(plt)
        return

def lens_gen_linear(resolution, type):

    y_values = np.linspace(-1, 1, resolution)

    if type == 'linear+':
        return {
            'x_values' : y_values,
            'dx' : [1]*resolution,
            'y_values' : y_values
        }
    
    if type == 'linear-':
        return {
            'x_values' : -y_values,
            'dx' : [-1]*resolution,
            'y_values' : y_values
        }

def lens_gen_quadratic(resolution, a):

    y_values = np.linspace(-1, 1, resolution)

    return {
        'x_values' : a*y_values**2,
        'dx' : 2 * a * y_values,
        'y_values' : y_values
    }

def lens_gen_sphere(resolution, r):

    y_values = np.linspace(-1, 1, resolution)

    return {
        'x_values' : np.sqrt(r**2 - y_values**2),
        'dx' : -y_values / np.sqrt(r**2 - y_values**2),
        'y_values' : y_values
    }

def lens_asphere(resolution, R, k, a):

    y_values = np.linspace(-1, 1, resolution)

    x = y_values**2 / (R * (1 + np.sqrt(1 - (1 + k) * y_values**2 / R**2)))

    for i, alpha in enumerate(a):
        x += alpha * y_values**(i+4)

    #https://www.wolframalpha.com/input?i=derivative+of+y%28x%29%3D%28x%5E2%29%2F%28R*%281%2B%28sqrt%281-%281%2Bk%29*x%5E2%2FR%5E2%29%29%29

    dx = (R * x * np.sqrt(1 - ((k + 1) * x**2) / R**2)) / (R**2 - (k + 1) * x**2)

    for i, alpha in enumerate(a):
        dx += (i+4) * alpha * y_values**(i+3)

    return {
        'x_values' : x,
        'dx' : dx,
        'y_values' : y_values
    }

def lens_asphere(resolution, R, k, a):

    y = np.linspace(-1, 1, resolution)

    sqrt_el = np.sqrt(1 - ((1 + k)*y**2)/R**2)

    x = y**2 / (R * (1 + sqrt_el))

    for i in range(len(a)):
        x += a[i] * y**(4 + i*2)

    dx = 2*y / (R*(sqrt_el + 1)) + ((k + 1) * y**3) / (R**3*(sqrt_el)*(sqrt_el + 1)**2)

    for i in range(len(a)):
        dx += (4 + i*2) * a[i] * (4 + i*2 - 1)

    return {
        'x_values' : x,
        'dx' : dx,
        'y_values' : y
    }

def rays_gen_parallel(start, end, n, direction):
    
        dir_x = np.linspace(start[0], end[0], n)
        dir_y = np.linspace(start[1], end[1], n)

        rays = []

        for i in range(n):
            rays.append(ray(strenght=1, origin=(dir_x[i], dir_y[i]), dir=direction))

        return rays

def rays_gen_point(start, n, angle_start = 0, angle_end = 2*np.pi, color = 'g'):
    
        rays = []
    
        angle = np.linspace(angle_start, angle_end, n)
        for i in range(n):
            rays.append(ray(strenght=1, origin=start, dir=(np.cos(angle[i]), np.sin(angle[i])), color=color))
    
        return rays

lente_gippo = lens(
    x = [-0.263 , 0.263 ],
    y = [-1, 1],
    dx = [0.263, 0.263],
    refidx_l = 1,
    refidx_r = 1.5,
    refl_coeff = 0.25,
    trans_coeff=0.6
)

lens0 = lens(
    x =  lens_asphere(100, 1, -1, [-0.7])['x_values'],
    y =  lens_asphere(100, 1, -1, [-0.7])['y_values'],
    dx = lens_asphere(100, 1, -1, [-0.7])['dx'],
    refidx_l = 1,
    refidx_r = 1.5,
    refl_coeff = 0.25,
    trans_coeff=0.6
)

sensor = lens(
    x=[5, 5],
    y=[-1, 1],
    dx=[0, 0],
    refidx_l=1,
    refidx_r=1,
    refl_coeff=0,
    trans_coeff=0
)

lens_sys0 = lens_system((0, 0))
lens_sys0.add_lens(lens0)
#lens_sys0.add_lens(lens1, offset=(2, 0))
#lens_sys0.add_lens(lens2, offset=(1, 0))

lens_sys0.add_boundary(refl_coeff=0.5)
lens_sys0.add_boundary(refl_coeff=0.5)

plt.figure()
lens_sys0.plot(plt)

#rays = rays_gen_parallel(start=(-1.5, -0.5), end=(-1.5, 0), n=1, direction=(0.939, 0.342))
rays = rays_gen_point(start=(-3, 0), n=10, angle_start=-0.25, angle_end=0.25, color='red')
#rays.extend(rays_gen_point(start=(-3, -0.25), n=30, angle_start=-0.1, angle_end=0.2, color='blue'))

result = lens_sys0.intersect(rays)

sensor.plot(plt)
image = []
for r in result[-1]:
    _, a = sensor.intersect(r)
    if a is not None:
        image.append((a.origin, a.color))

for dot in image:
    plt.plot(dot[0][0], dot[0][1], dot[1][0]+'o', alpha=0.5)

cmap = plt.get_cmap('hsv')
for i, rays in enumerate(result):

    color = cmap(i / len(result))

    for r in rays:
        r.plot(plt, color=color, alpha=np.clip(r.strength, 0, 1))
        r.plot(plt, color=r.color, alpha=np.clip(r.strength, 0, 1))
        print(r)

plt.ylim(-2, 2)
plt.xlim(-2, 2)

plt.show()


import sys
sys.exit()
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