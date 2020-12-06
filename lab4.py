
# %% [markdown]
# ### Importy
# %%
from sortedcollections import SortedDict
from enum import Enum, Flag
from queue import PriorityQueue
from collections import OrderedDict

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

SAVE_FILES = True
# %% [markdown]
# ### Implementacja wyznacznika oraz funkcji orientacji
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


def does_intersecs(l1, l2):
    a, b = l1
    c, d = l2

    return orient(a, b, c) != orient(a, b, d) and \
        orient(a, b, c) != orient(a, b, d)


def get_linfun(p1, p2):
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = -a * p1[0] + p1[1]
    return a, b


def line_intersection(l1, l2):
    a1, b1 = get_linfun(*l1)
    a2, b2 = get_linfun(*l2)

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return x, y

# %%


class Heap(object):
    def __init__(self):
        self.heap = []

    def insert(self, val):
        i = len(self.heap)
        self.heap.append(val)

        while i > 0 and self.heap[i] < self.heap[(i - 1) // 2]:
            p = (i - 1) // 2
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            i = p

    def pop(self):
        self.heap[-1], self.heap[0] = self.heap[0], self.heap[-1]
        val = self.heap.pop()
        self.heapify(0)

        return val

    def peek(self):
        return self.heap[0]

    def heapify(self, idx):
        l = 2 * idx + 1
        r = 2 * idx + 2

        i = idx

        if l < len(self.heap) and self.heap[i] > self.heap[l]:
            i = l

        if r < len(self.heap) and self.heap[i] > self.heap[r]:
            i = r

        if i != idx:
            self.heap[i], self.heap[idx] = self.heap[idx], self.heap[i]
            self.heapify(i)

    def __len__(self):
        return len(self.heap)


class Node(object):
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.parent: Node = None
        self.left: Node = None
        self.right: Node = None
        self.color = None

    def succ(self):
        if self.right is not None:
            return self.right.min()

        x = self
        y = self.parent

        while y is not None and y.right is x:
            x = y
            y = y.parent

        return y

    def pred(self):
        if self.left is not None:
            return self.left.max()

        x = self
        y = self.parent

        while y is not None and y.left is x:
            x = y
            y = y.parent

        return y

    def max(self):
        node = self
        while node.right is not None:
            node = node.right
        return node

    def min(self):
        node = self
        while node.left is not None:
            node = node.left
        return node


class Tree(object):
    def __init__(self):
        self.root: Node = None

    def add(self, key, value):
        node = Node(key, value)
        self.add_node(node)
        return node

    def add_node(self, z: Node):
        y = None
        x = self.root

        while x is not None:
            y = x

            if z.key < x.key:
                x = x.left
            else:
                x = x.right

        z.parent = y
        if y is None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

    def find(self, key):
        node = self.root

        while node is not None and node.key != key:
            if node.key < key:
                node = node.left
            else:
                node = node.right

        return node

    def remove(self, key):
        node = self.find(key)
        self.remove_node(node)
        return node

    def remove_node(self, z: Node):
        if z.left is None:
            self.put_in_place(z, z.right)
        elif z.right is None:
            self.put_in_place(z, z.left)
        else:
            y = z.right.min()
            if y.parent is not z:
                self.put_in_place(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.put_in_place(z, y)
            y.left = z.left
            y.left.parent = y

    def put_in_place(self, u, v):
        if u is self.root:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v

        if v is not None:
            v.parent = u.parent

    def __str__(self):
        def to_str(node):
            if node is None:
                return ""

            return f"{to_str(node.left)}{node.key} = {node.value}, {to_str(node.right)}"

        return f"Tree[{to_str(self.root)}]"

# %%


class Event(int, Enum):
    START = 0
    INTERSECTION = 1
    END = 2


def find_intersections(segments: list):
    queue = Heap()
    tree = Tree()
    result = []
    nodes = [None] * len(segments)
    sweep = Tree()

    for i, seg in enumerate(segments):
        start, end = seg

        if start[0] > end[0]:
            start, end = end, start
            segments[i] = start, end

        queue.insert((start[0], i, None, Event.START))
        queue.insert((end[0], i, None, Event.END))

    def update_events(cursor, i, j):
        nonlocal nodes, sweep, segments

        if i != j and does_intersecs(segments[i], segments[j]):
            x, _ = line_intersection(segments[i], segments[j])

            if cursor > x:
                return

            queue.insert((x, i, j, Event.INTERSECTION))

    while queue:
        cursor, i, j, event = queue.pop()

        above = None
        below = None

        if event == Event.START:
            start, end = segments[i]
            nodes[i] = sweep.add((start[1], 0), i)

            above = nodes[i].succ()
            below = nodes[i].pred()

            if above is not None:
                update_events(cursor, above.value, i)

            if below is not None:
                update_events(cursor, i, below.value)

        elif event == Event.END:
            above = nodes[i].succ()
            below = nodes[i].pred()

            sweep.remove_node(nodes[i])
            nodes[i] = None

            if above is not None and below is not None:
                update_events(cursor, above.value, below.value)

        elif event == Event.INTERSECTION:
            if len(result) > 0 and \
                    (result[-1] == (i, j) or result[-1] == (j, i)):

                continue

            if nodes[i].key[0] > nodes[j].key[0]:
                upper = j
                lower = i
            else:
                upper = i
                lower = j

            sweep.remove_node(nodes[i])
            sweep.remove_node(nodes[j])

            _, y = line_intersection(segments[i], segments[j])

            nodes[upper] = sweep.add((y, 1), upper)
            nodes[lower] = sweep.add((y, -1), lower)

            above = nodes[upper].succ()
            below = nodes[lower].pred()

            if above is not None:
                update_events(cursor, above.value, upper)

            if below is not None:
                update_events(cursor, lower, below.value)

            result.append((upper, lower))

    return result


# segments = [((0, 0), (20, 20)),
#             ((2, 5), (8, 5)),
#             ((5, 10), (17, 15)),
#             ((5, 15), (15, 5)), ]

segments = [((-75, -75), (75, 75)),
            ((-75, 75), (25, -25)),
            ((-75, 50), (75, 50)), ]

# segments = [((-100, 0), (100, 0)),  # 0 -
#             ((-75, 25), (50, -50)),  # 1 \
#             ((75, 25), (-50, -50)), ]  # 2 /

# segments = [((-75, 0), (50, 75)),
#             ((-50, 75), (75, 0)),
#             ((-50, 25), (25, -75))]

# segments = [((-75, -75), (75, 75)),
#             ((-25, 74), (50, -99)),
#             ((-75, -25), (75, -75))]

print(find_intersections(segments))


# %%

def draw_segments(ax, segments):
    axes = []
    for start, end in segments:
        axes.extend(
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    marker="o",
                    zorder=1,
                    markersize=4,
                    color="black"))

    return axes


def draw_intersections(ax, segments, intersections=None):
    if intersections is None:
        intersections = find_intersections(segments)

    axes = []

    for seg_a, seg_b in intersections:
        x, y = line_intersection(segments[seg_a], segments[seg_b])
        axes.append(
            ax.scatter(x, y,
                       color="red",
                       zorder=2))

    return axes


def draw_texts(ax, segments):
    axes = []

    for i, seg in enumerate(segments):
        start, end = seg
        # mx = (start[0] + end[0]) / 2.0
        # my = (start[1] + end[1]) / 2.0

        mx = start[0] - 4
        my = start[1] + 2
        axes.append(ax.text(mx, my, str(i)))

    return axes


fig, ax = plt.subplots()

ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

draw_segments(ax, segments)
draw_intersections(ax, segments)
draw_texts(ax, segments)

fig.show()

# %%


class Application(object):
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

        self.segments = []
        self.actors = []
        self.c_points = None

        self.fig.canvas.mpl_connect("button_press_event", self.add_segment)

        self.draw()
        self.fig.show()

    def draw(self):
        self.clear_actors()
        self.actors.extend(draw_segments(self.ax, self.segments))
        self.actors.extend(draw_intersections(self.ax, self.segments))
        self.actors.extend(draw_texts(self.ax, self.segments))

    def add_segment(self, event=None):
        if event.inaxes not in {actor.axes for actor in self.actors} \
                and event.inaxes != self.ax:
            return

        mx, my = event.xdata, event.ydata
        if not self.c_points:
            self.c_points = (mx, my)
            self.actors.append(
                self.ax.scatter([mx], [my])
            )
            return

        self.segments.append((self.c_points, (mx, my)))
        self.c_points = None

        self.draw()

    def clear(self, event=None):
        self.set_segments([])

    def clear_actors(self):
        for actor in self.actors:
            actor.remove()

        self.actors.clear()

    def set_segments(self, segments):
        self.clear_actors()
        self.segments = segments
        self.draw()


app = Application()

# %%
