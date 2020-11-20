# %%
from IPython.display import Latex
from functools import cmp_to_key
from itertools import product
from timeit import default_timer as timer
from IPython.display import Markdown, HTML
from ipywidgets import interactive

import ipywidgets as widgets
import matplotlib
import matplotlib.widgets as wig
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib widget
sns.set_theme()
sns_style = {"edgecolor": "none", "s": 1}
plt.rcParams["animation.html"] = "jshtml"

SAVE_FILES = False
# %%
EPSILON = 1e-9


def det(a, b, c):
    det = a[0] * b[1] + \
        a[1] * c[0] + \
        b[0] * c[1] - \
        b[1] * c[0] - \
        c[1] * a[0] - \
        b[0] * a[1]
    return det


CCW, COL, CW = 1, 0, -1


def orient(a, b, c):
    d = det(a, b, c)

    if -EPSILON < d < EPSILON:
        return 0
    elif d > 0:
        return 1
    elif d < 0:
        return -1


# %%
SETS = {
    "A": [[0.0, 100.0],
          [-10.0, 75.0],
          [-5.0, 50.0],
          [0.0, 0.0],
          [5.0, 20],
          [10.0, 30.0],
          [50.0, 50.0],
          [50.0, 90.0]],

    "B": [[50, 75],
          [-25, 50],
          [25, 25],
          [-25, 0],
          [50, -25]],

    "nonB": [[0.0, 100.0],
             [-5.0, 50.0],
             [10.0, 75.0],
             [0.0, 0.0],
             [50.0, 50.0],
             [50.0, 90.0]],

    "C": [[29.36572356215214, 75.51020408163268],
          [-94.19642857142857, 6.493506493506516],
          [62.01878478664196, -88.49721706864563],
          [89.84809833024121, 16.883116883116912],
          [18.605055658627094, 30.983302411873865]],

    "D": [[0.12755102040816269, 74.3970315398887],
          [-74.45500927643783, -1.2987012987012747],
          [2.353896103896119, -76.99443413729125],
          [76.56539888682747, -1.2987012987012747],
          [1.6117810760667908, 24.304267161410053],
          [29.441094619666075, 68.46011131725422],
          [-3.2119666048237434, 96.66048237476812]],

    "H": [[0.12755102040816269, 68.46011131725422],
          [-57.015306122448976, 44.71243042671617],
          [-18.05426716141001, 24.675324675324703],
          [-47.36781076066789, 3.8961038961039094],
          [-12.117346938775512, -6.493506493506473],
          [-60.725881261595546, -30.612244897959158],
          [-18.796382189239324, -44.712430426716125],
          [-62.952226345083474, -56.9573283858998],
          [4.951298701298725, -66.2337662337662],
          [59.49675324675326, -51.7625231910946],
          [19.051484230055678, -38.40445269016695],
          [62.094155844155864, -19.85157699443411],
          [22.019944341372934, -9.090909090909065],
          [70.2574211502783, 7.977736549165144],
          [18.30936920222635, 23.933209647495403],
          [60.238868274582586, 33.20964749536182],
          [0.8696660482374909, 50.27829313543603]],
}
# %%


def split_chains(points, axis=1):
    p_max = max(range(len(points)), key=lambda i: points[i][axis])
    p_min = min(range(len(points)), key=lambda i: points[i][axis])

    chain_a, chain_b = [], []

    i = min(p_min, p_max)
    while i != max(p_min, p_max):
        chain_a.append(points[i])
        i += 1
        i %= len(points)

    chain_a.append(points[i])

    while i != min(p_min, p_max):
        chain_b.append(points[i])
        i += 1
        i %= len(points)

    chain_b.append(points[i])

    if p_max > p_min:
        chain_a, chain_b = chain_b, chain_a

    chain_b.reverse()
    return chain_a, chain_b


def plot_points(ax, points, **kw_args):
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], **kw_args)


points = SETS["B"]
fig, ax = plt.subplots()
left, right = split_chains(points)
fig.show()

plot_points(ax, left, color="b")
plot_points(ax, right, color="r")

# %%


def check_monotone(points, axis=1):
    left, right = split_chains(points, axis=axis)

    for i in range(1, len(left)):
        if left[i][axis] < left[i - 1][axis]:
            return False

    for i in range(1, len(right)):
        if right[i][axis] > right[i - 1][axis]:
            return False

    return True

# %%


print(check_monotone(SETS["B"]))
print(check_monotone(SETS["nonB"]))

# %%


def monotone_triangulate(points, axis=1):
    left, right = split_chains(points, axis=axis)
    LEFT, RIGHT, BOTH = 0b01, 0b10, 0b11

    li, ri = 1, 1
    points = [(left[0][0], left[0][1], BOTH)]
    while li < len(left) - 1 or ri < len(right) - 1:
        if li >= len(left) - 1:
            points.append((right[ri][0], right[ri][1], RIGHT))
            ri += 1
        elif ri >= len(right) - 1:
            points.append((left[li][0], left[li][1], LEFT))
            li += 1
        elif left[li][1] >= right[ri][1]:
            points.append((left[li][0], left[li][1], LEFT))
            li += 1
        elif left[li][1] < right[ri][1]:
            points.append((right[ri][0], right[ri][1], RIGHT))
            ri += 1
        else:
            assert False

    points.append((left[-1][0], left[-1][1], BOTH))
    queue = points[:2]
    triangles = []

    del left, right

    for i in range(2, len(points)):
        p = points[i]

        diff_chains = (queue[-1][2] & LEFT and p[2] & RIGHT) or \
            (queue[-1][2] & RIGHT and p[2] & LEFT)

        if diff_chains:
            while len(queue) >= 2:
                a = queue.pop()
                triangles.append((a[:2], queue[-1][:2], p[:2]))

            queue = [points[i - 1], points[i]]
        else:
            skipped = []
            while len(queue) >= 2:
                ori = orient(queue[-2], p, queue[-1])
                if ((p[2] & LEFT and ori == -1) or (p[2] & RIGHT and ori == 1)):
                    triangles.append((queue[-2][:2], queue[-1][:2], p[:2]))
                    queue.pop()
                else:
                    break
            queue.append(p)

    return triangles


# %%
points = SETS["A"]
fig, ax = plt.subplots()
left, right = split_chains(points)

plot_points(ax, left)
plot_points(ax, right)

triangles = monotone_triangulate(points)
print(triangles)

for a, b, c in triangles:
    x = [a[0], b[0], c[0]]
    y = [a[1], b[1], c[1]]
    ax.fill(x, y, alpha=0.3)

# %%


class Application:
    def __init__(self):
        self.fig = plt.figure(figsize=(9, 7))

        SPECS = {"width_ratios": [7, 2],
                 "hspace": 0.0,
                 "wspace": 0.0}

        gs1 = plt.GridSpec(1, 2, **SPECS)
        gs1.tight_layout(self.fig)
        gs2 = plt.GridSpec(10, 2, **SPECS)

        self.ax = self.fig.add_subplot(gs1[0])

        self.ax.set_aspect("equal")
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)

        self.text = self.ax.text(-100, 120, "BRRR")

        self.line_l, = self.ax.plot([], [], marker="o", color="b")
        self.line_r, = self.ax.plot([], [], marker="o", color="r")

        btn_ax = self.fig.add_subplot(gs2[1])
        self.check_btn = wig.Button(btn_ax, "Sprawdź \nmonotoniczność")
        self.check_btn.on_clicked(self.check_monotone)

        btn_ax = self.fig.add_subplot(gs2[3])
        self.triangulate_btn = wig.Button(btn_ax, "Trianguluj")
        self.triangulate_btn.on_clicked(self.triangulate)

        btn_ax = self.fig.add_subplot(gs2[5])
        self.reset_btn = wig.Button(btn_ax, "Reset")
        self.reset_btn.on_clicked(self.reset)

        self.points = []
        self.text_axes = []
        self.tri_axes = []

        self.line_l.figure.canvas.mpl_connect(
            "button_press_event", self.add_point)

        self.line_r.figure.canvas.mpl_connect(
            "button_press_event", self.add_point)

        self.fig.show()

    def add_point(self, event):
        if event.inaxes != self.line_l.axes and \
                event.inaxes != self.line_r.axes:
            return

        self.text_axes.append(
            self.ax.text(event.xdata, event.ydata, str(len(self.points))))

        self.points.append([event.xdata, event.ydata])
        self.draw()

    def set_points(self, points):
        self.reset(None)
        self.points = points

        for i, p in enumerate(points):
            self.text_axes.append(self.ax.text(p[0], p[1], str(i)))

        self.draw()

    def draw(self):
        try:
            self.check_monotone(None)
            self.triangulate(None)
        finally:
            pass

        left, right = split_chains(self.points)

        self.line_l.set_data(np.transpose(left))
        self.line_r.set_data(np.transpose(right))

        self.fig.canvas.draw()

    def check_monotone(self, event):
        if check_monotone(self.points):
            self.text.set_text("Y-Monotoniczny")
        else:
            self.text.set_text("Nie Y-Monotoniczny")

    def triangulate(self, event):
        self.clear_triangles()
        if not check_monotone(self.points):
            self.text.set_text("Nie Y-Monotoniczny !!!")
            return

        triangles = monotone_triangulate(self.points)
        for a, b, c in triangles:
            x = [a[0], b[0], c[0]]
            y = [a[1], b[1], c[1]]
            tri, = self.ax.fill(x, y, alpha=0.3)
            self.tri_axes.append(tri)

    def clear_triangles(self):
        while self.tri_axes:
            self.tri_axes.pop().remove()

    def reset(self, event):
        self.text.set_text("Reset")
        self.points = []
        self.line_l.set_data([], [])
        self.line_r.set_data([], [])

        self.clear_triangles()

        while self.text_axes:
            self.text_axes.pop().remove()

        self.fig.canvas.draw()


app = Application()

# %%
alpha = np.linspace(0.5 * np.pi, 2.5 * np.pi, 100)
r = 100.0
points = [(np.cos(a) * r, np.sin(a) * r) for a in alpha]
display(points)
app = Application()
app.set_points(points)
