# %%
from enum import Enum, Flag

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


class Orient(int, Enum):
    CCW, COL, CW = 1, 0, -1


def orient(a, b, c):
    d = det(a, b, c)

    if -EPSILON < d < EPSILON:
        return Orient.COL
    elif d > 0:
        return Orient.CCW
    elif d < 0:
        return Orient.CW


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

    "W": [[-45.14146567717995, 72.54174397031542],
          [-81.13404452690166, 36.17810760667908],
          [-23.62012987012986, -7.235621521335787],
          [12.372448979591837, 22.448979591836775],
          [-36.60714285714285, 30.983302411873865],
          [29.441094619666075, 51.76252319109466],
          [54.673005565862724, -12.430426716140971],
          [-13.60157699443414, -36.92022263450832],
          [-80.762987012987, -18.367346938775484],
          [-95.60528756957328, -58.81261595547308],
          [-15.456864564007404, -94.80519480519479],
          [27.956864564007446, -56.586270871985135],
          [76.19434137291282, -95.5473098330241],
          [95.48933209647498, -10.32282003710573],
          [70.2574211502783, -33.209647495361764],
          [91.40769944341375, 62.894248608534355],
          [31.296382189239353, 92.20779220779224],
          [-4.6961966604823715, 69.94434137291285]],
}
# %%


def split_chains(points, axis=1):
    p_max, _ = max(enumerate(points), key=lambda ip: ip[1][axis])
    p_min, _ = min(enumerate(points), key=lambda ip: ip[1][axis])

    chain_a, chain_b = [], []

    i = p_max
    while i != p_min:
        chain_a.append(points[i])
        i += 1
        i %= len(points)

    chain_a.append(points[i])

    while i != p_max:
        chain_b.append(points[i])
        i += 1
        i %= len(points)

    chain_b.append(points[i])

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
        if left[i][axis] > left[i - 1][axis]:
            return False

    for i in range(1, len(right)):
        if right[i][axis] > right[i - 1][axis]:
            return False

    return True

# %%


print(check_monotone(SETS["B"]))
print(check_monotone(SETS["nonB"]))

# %%


class Chain(int, Flag):
    LEFT, RIGHT, BOTH = 0b01, 0b10, 0b11


def chain_sort(left, right):
    li, ri = 1, 1
    points = [(left[0][0], left[0][1], Chain.BOTH)]
    while li < len(left) - 1 or ri < len(right) - 1:
        if li >= len(left) - 1:
            points.append((right[ri][0], right[ri][1], Chain.RIGHT))
            ri += 1
        elif ri >= len(right) - 1:
            points.append((left[li][0], left[li][1], Chain.LEFT))
            li += 1
        elif left[li][1] >= right[ri][1]:
            points.append((left[li][0], left[li][1], Chain.LEFT))
            li += 1
        elif left[li][1] < right[ri][1]:
            points.append((right[ri][0], right[ri][1], Chain.RIGHT))
            ri += 1
        else:
            assert False

    points.append((left[-1][0], left[-1][1], Chain.BOTH))
    return points

# %%


def monotone_triangulate(points, axis=1):
    left, right = split_chains(points, axis=axis)
    points = chain_sort(left, right)
    queue = points[:2]
    triangles = []

    def _is(x, a):
        return (x & a) == a

    del left, right

    for i in range(2, len(points)):
        p = points[i]

        if (queue[-1][2] | p[2]) == Chain.BOTH:
            while len(queue) >= 2:
                a = queue.pop()
                triangles.append((a[:2], queue[-1][:2], p[:2]))

            queue = [points[i - 1], points[i]]
        else:
            skipped = []
            while len(queue) >= 2:
                ori = orient(queue[-2], p, queue[-1])
                if (_is(p[2], Chain.LEFT) and ori == Orient.CW) \
                        or (_is(p[2], Chain.RIGHT) and ori == Orient.CCW):

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

plot_points(ax, left, color="b")
plot_points(ax, right, color="r")

triangles = monotone_triangulate(points)
print(triangles)

for a, b, c in triangles:
    x = [a[0], b[0], c[0]]
    y = [a[1], b[1], c[1]]
    ax.fill(x, y, alpha=0.3)

# %%


class Classification(Enum):
    STARTING = 0
    FINISHING = 1
    JOINING = 2
    DIVIDING = 3
    REGULAR = 4


def classify_points(points, axis=1):
    result = [None] * len(points)

    for i in range(len(points)):
        prev = points[i - 1]
        curr = points[i]
        next = points[(i + 1) % len(points)]

        # curr above neighbors
        above = curr[axis] > prev[axis] and curr[axis] > next[axis]
        below = curr[axis] < prev[axis] and curr[axis] < next[axis]
        ori = orient(prev, curr, next)

        cl = None
        if above and ori == Orient.CCW:
            cl = Classification.STARTING
        elif below and ori == Orient.CCW:
            cl = Classification.FINISHING
        elif below and ori == Orient.CW:
            cl = Classification.JOINING
        elif above and ori == Orient.CW:
            cl = Classification.DIVIDING
        else:
            cl = Classification.REGULAR

        result[i] = (points[i][0], points[i][1], cl)

    return result

# %%


CLASSIFICATION_COLORS = {
    Classification.STARTING: "#5CC83B",
    Classification.FINISHING: "#EB3223",
    Classification.JOINING: "#323693",
    Classification.DIVIDING: "#C2DFE2",
    Classification.REGULAR: "#60350F",
}


def plot_classification(ax, points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    c = [CLASSIFICATION_COLORS[p[2]] for p in points]

    sc1 = ax.scatter(x, y, color="black", marker="o", linewidth=3, zorder=2)
    sc2 = ax.scatter(x, y, c=c, marker="o", linewidth=1, zorder=3)

    return sc1, sc2


points = classify_points(SETS["W"])
fig, ax = plt.subplots()
plot_points(ax, points + [points[0]], color="b", marker="")
plot_classification(ax, points)
fig.show()
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

        btn_ax = self.fig.add_subplot(gs2[7])
        self.circle_btn = wig.Button(btn_ax, "Koło")
        self.circle_btn.on_clicked(self.set_circle)

        btn_ax = self.fig.add_subplot(gs2[9])
        self.harmania_btn = wig.Button(btn_ax, "Harmonijka")
        self.harmania_btn.on_clicked(self.set_harmania)

        btn_ax = self.fig.add_subplot(gs2[11])
        self.classify_btn = wig.Button(btn_ax, "Zklasyfikuj")
        self.classify_btn.on_clicked(self.classify)

        self.points = []
        self.text_axes = []
        self.tri_axes = []
        self.class_axes = []

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

    def set_circle(self, event):
        alpha = np.linspace(0.5 * np.pi, 2.5 * np.pi, 64)
        r = 95.0
        points = [(np.cos(a) * r, np.sin(a) * r) for a in alpha]

        self.set_points(points)

    def set_harmania(self, event):
        points = []
        alpha = np.linspace(95, -90, 32)
        r = 50.0
        for i, a in enumerate(alpha):
            rr = r if i % 2 == 0 else r / 2
            points.append((-rr, a))

        alpha = np.linspace(-95, 90, 32)
        for i, a in enumerate(alpha):
            rr = r if i % 2 == 0 else r / 2
            points.append((rr, a))

        self.set_points(points)

    def reset(self, event):
        self.text.set_text("Reset")
        self.points = []
        self.line_l.set_data([], [])
        self.line_r.set_data([], [])

        self.clear_triangles()
        self.clear_classify()

        while self.text_axes:
            self.text_axes.pop().remove()

        self.fig.canvas.draw()

    def classify(self, event):
        self.clear_classify()
        points = classify_points(self.points)
        self.class_axes = plot_classification(self.ax, points)

    def clear_classify(self):
        if self.class_axes:
            self.class_axes[0].remove()
            self.class_axes[1].remove()
            self.class_axes = None


app = Application()

# %%
