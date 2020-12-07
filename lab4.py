
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

# %% [markdown]
# ### Funkcje sprawdzające przecięcie odcinków

# %%


def does_intersecs(l1, l2):
    a, b = l1
    c, d = l2

    o = [orient(a, b, c),
         orient(a, b, d),
         orient(c, d, a),
         orient(c, d, b), ]

    return o[0] != o[1] and o[2] != o[3]


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

# %% [markdown]
# ### Implementacja kopca oraz drzewa BST

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

# %% [markdown]
# ### Typ rodzaju wydarzenia

# %%


class Event(int, Enum):
    START = 0
    INTERSECTION = 1
    END = 2

# %% [markdown]
# ### Funkcja sprawdzająca czy istnieje jakiekolwiek przecięcie

# %%


def any_intersects(segments: list):
    queue = Heap()
    nodes = [None] * len(segments)
    sweep = Tree()

    for i, seg in enumerate(segments):
        start, end = seg

        if start[0] > end[0]:
            start, end = end, start
            segments[i] = start, end

        queue.insert((start[0], i, None, Event.START))
        queue.insert((end[0], i, None, Event.END))

    def check(i, j):
        return i != j and does_intersecs(segments[i], segments[j])

    while queue:
        broom, i, j, event = queue.pop()

        if event == Event.START:
            start, end = segments[i]
            nodes[i] = sweep.add((start[1], 0), i)

            above = nodes[i].succ()
            below = nodes[i].pred()

            if above is not None and check(above.value, i):
                return True

            if below is not None and check(i, below.value):
                return True

        elif event == Event.END:
            above = nodes[i].succ()
            below = nodes[i].pred()

            sweep.remove_node(nodes[i])
            nodes[i] = None

            if above is not None and below is not None and \
                    check(above.value, below.value):
                return True

    return False

# %% [markdown]
# ### Zbiory testowe

# %%


SETS = {
    "X": [((-91.22427690891698, 80.36030219083375),
           (75.00948932484928, -91.810384265567)),
          ((-86.77158674194109, -89.95509669599372),
           (85.77015722837433, 69.5996342873087)),
          ((2.282216597576536, -58.04415049933323),
           (24.174609918541307, 43.996665827197376)),
          ((29.74047262726114, -59.157323041077206),
           (50.890750920396584, 55.1283912446371)),
          ((-74.15563126884275, -10.548788718257157),
           (-37.42093739129173, -10.177731204342507))],

    "K": [((-74.15563126884275, -86.53125087373218),
           (-26.289211973852034, 63.13985569478302)),
          ((-46.69737523915816, -86.90230838764683),
           (10.816539417613654, 58.55223706689864)),
          ((-18.12594666772958, -89.49971098504943),
           (38.64585296121291, 53.35743187209343)),
          ((26.772012515943885, -86.90230838764683),
           (69.81468413004407, 50.01791424686152)),
          ((-94.56379453414888, -79.48115810935369),
           (-62.28179082357374, 64.48915728953315)),
          ((-90.85321939500231, 54.84166192775206),
           (89.48073236752089, 16.251680480627755)),
          ((-94.93485204806353, 21.44648567543294),
           (83.17275463097172, -32.727911356106944)),
          ((-94.93485204806353, -11.5776330629715),
           (66.47516650481217, -60.5572248997062)),
          ((-94.19273702023423, -38.66483157874144),
           (68.33045407438544, -89.49971098504943))],


    "A": [((0, 0), (20, 20)),
          ((2, 5), (8, 5)),
          ((5, 10), (17, 15)),
          ((5, 15), (15, 5)), ],

    "B": [((-75, -75), (75, 75)),
          ((-75, 75), (25, -25)),
          ((-75, 50), (75, 50)), ],

    "C": [((-100, 0), (100, 0)),  # 0 -
          ((-75, 25), (50, -50)),  # 1 \
          ((75, 25), (-50, -50)), ],  # 2 /

    "D": [((-75, 0), (50, 75)),
          ((-50, 75), (75, 0)),
          ((-50, 25), (25, -75))],

    "E": [((-75, -75), (75, 75)),
          ((-25, 74), (50, -99)),
          ((-75, -25), (75, -75))],

}

# %% [markdown]
# ### Algortym właściwy

# %%


def find_intersections(segments: list):
    queue = Heap()
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

    def update_events(broom, i, j):
        nonlocal nodes, sweep, segments

        if i != j and does_intersecs(segments[i], segments[j]):
            x, _ = line_intersection(segments[i], segments[j])

            if broom > x:
                return

            queue.insert((x, i, j, Event.INTERSECTION))

    while queue:
        broom, i, j, event = queue.pop()

        if event == Event.START:
            start, end = segments[i]
            nodes[i] = sweep.add((start[1], 0), i)

            above = nodes[i].succ()
            below = nodes[i].pred()

            if above is not None:
                update_events(broom, above.value, i)

            if below is not None:
                update_events(broom, i, below.value)

        elif event == Event.END:
            above = nodes[i].succ()
            below = nodes[i].pred()

            sweep.remove_node(nodes[i])
            nodes[i] = None

            if above is not None and below is not None:
                update_events(broom, above.value, below.value)

        elif event == Event.INTERSECTION:
            if len(result) > 0 and \
                    (result[-1] == (i, j) or result[-1] == (j, i)):

                continue

            upper, lower = \
                (j, i) if nodes[i].key[0] > nodes[j].key[0] else (i, j)

            sweep.remove_node(nodes[i])
            sweep.remove_node(nodes[j])

            _, y = line_intersection(segments[i], segments[j])

            nodes[upper] = sweep.add((y, 1), upper)
            nodes[lower] = sweep.add((y, -1), lower)

            above = nodes[upper].succ()
            below = nodes[lower].pred()

            if above is not None:
                update_events(broom, above.value, upper)

            if below is not None:
                update_events(broom, lower, below.value)

            result.append((upper, lower))

    return result


print(find_intersections(SETS["B"]))


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

segments = SETS["B"]
draw_segments(ax, segments)
draw_intersections(ax, segments)
draw_texts(ax, segments)

fig.show()

# %% [markdown]
# ### Interfejs wyświetlania odcinków i przecięć oraz dodawania odcinków

# %%


class Application(object):
    def __init__(self, segments=None, gen_n=None):
        self.fig = plt.figure(figsize=(9, 7))

        SPECS = {"width_ratios": [7, 2],
                 "hspace": 0.0,
                 "wspace": 0.0}

        gs1 = plt.GridSpec(1, 2, **SPECS)
        gs1.tight_layout(self.fig)
        gs2 = plt.GridSpec(10, 2, **SPECS)

        self.ax = self.fig.add_subplot(gs1[0])

        self.text_ax = self.ax.text(-100, 105, "Application")

        self.ax.set_aspect("equal")
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)

        if segments:
            self.segments = segments
        else:
            self.segments = []

        self.actors = []
        self.c_points = None

        self.fig.canvas.mpl_connect("button_press_event", self.add_segment)

        btn_ax = self.fig.add_subplot(gs2[1])
        self.reset_btn = wig.Button(btn_ax, "Reset")
        self.reset_btn.on_clicked(self.clear)

        btn_ax = self.fig.add_subplot(gs2[3])
        self.n_input = wig.TextBox(btn_ax, "N", initial=str(gen_n))
        self.n_input.on_submit(self.gen)

        btn_ax = self.fig.add_subplot(gs2[5])
        self.gen_btn = wig.Button(btn_ax, "Generuj")
        self.gen_btn.on_clicked(self.gen)

        if gen_n:
            self.gen()

        self.draw()
        self.fig.show()

    def draw(self):
        self.clear_actors()
        self.actors.extend(draw_segments(self.ax, self.segments))
        self.actors.extend(draw_texts(self.ax, self.segments))
        self.actors.extend(draw_intersections(self.ax, self.segments))

        n = len(find_intersections(self.segments))
        self.text_ax.set_text(
            f"Liczba odcinków: {len(self.segments)}\nLiczba przecięć {n}")

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

    def gen(self, event=None):
        n = int(self.n_input.text)

        segments = []

        while n > 0:
            x1 = np.random.uniform(-100, 100)
            y1 = np.random.uniform(-100, 100)
            x2 = np.random.uniform(-100, 100)
            y2 = np.random.uniform(-100, 100)

            if abs(x1 - x2) < EPSILON:
                continue

            segments.append(((x1, y1), (x2, y2)))

            n -= 1

        self.set_segments(segments)


app = Application()

# %% [markdown]
# ### Animacja algorytmu

# %%


class StopAnimation(Exception):
    def __init__(self, frame):
        self.frame = frame

    @staticmethod
    def check(frame, draw, segments, args):
        if frame <= 0:
            draw(segments, *args)
            raise StopAnimation(frame)


def _find(frame, segments, args):
    queue = Heap()
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

    def draw(segments, ax_broom, ax_cross, ax_start, ax_inter, ax_inter2, ax_end, *args):
        ax_start.set_data([], [])
        ax_inter.set_data([], [])
        ax_inter2.set_data([], [])
        ax_end.set_data([], [])

        intersections = [line_intersection(segments[a], segments[b])
                         for a, b in result]

        ax_cross.set_data(
            [p[0] for p in intersections],
            [p[1] for p in intersections]
        )

        if not queue:
            ax_broom.set_data([100, 100], [-100, 100])
            return

        broom, i, j, event = queue.peek()
        ax_broom.set_data([broom, broom], [-100, 100])

        if event == Event.START:
            start, end = segments[i]
            ax_start.set_data([start[0], end[0]], [start[1], end[1]])

        elif event == Event.END:
            start, end = segments[i]
            ax_end.set_data([start[0], end[0]], [start[1], end[1]])

        elif event == Event.INTERSECTION:
            ax_inter.set_data([segments[i][0][0],
                               segments[i][1][0], ],
                              [segments[i][0][1],
                               segments[i][1][1], ])
            ax_inter2.set_data([segments[j][0][0],
                                segments[j][1][0], ],
                               [segments[j][0][1],
                                segments[j][1][1], ])

    def update_events(broom, i, j):
        nonlocal nodes, sweep, segments

        if i != j and does_intersecs(segments[i], segments[j]):
            x, _ = line_intersection(segments[i], segments[j])

            if broom > x:
                return

            queue.insert((x, i, j, Event.INTERSECTION))

    while queue:
        frame -= 1
        StopAnimation.check(frame, draw, segments, args)

        broom, i, j, event = queue.pop()

        if event == Event.START:
            start, end = segments[i]
            nodes[i] = sweep.add((start[1], 0), i)

            above = nodes[i].succ()
            below = nodes[i].pred()

            if above is not None:
                update_events(broom, above.value, i)

            if below is not None:
                update_events(broom, i, below.value)

        elif event == Event.END:
            above = nodes[i].succ()
            below = nodes[i].pred()

            sweep.remove_node(nodes[i])
            nodes[i] = None

            if above is not None and below is not None:
                update_events(broom, above.value, below.value)

        elif event == Event.INTERSECTION:
            if len(result) > 0 and \
                    (result[-1] == (i, j) or result[-1] == (j, i)):

                continue

            upper, lower = \
                (j, i) if nodes[i].key[0] > nodes[j].key[0] else (i, j)

            sweep.remove_node(nodes[i])
            sweep.remove_node(nodes[j])

            _, y = line_intersection(segments[i], segments[j])

            nodes[upper] = sweep.add((y, 1), upper)
            nodes[lower] = sweep.add((y, -1), lower)

            above = nodes[upper].succ()
            below = nodes[lower].pred()

            if above is not None:
                update_events(broom, above.value, upper)

            if below is not None:
                update_events(broom, lower, below.value)

            result.append((upper, lower))

        frame -= 1
        StopAnimation.check(frame, draw, segments, args)

    return frame


def _frame(frame, segments, *args):
    try:
        frame = _find(frame, segments, args)
        raise StopAnimation(frame)
    except StopAnimation as stop:
        return stop.frame


def animate(segments):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    draw_segments(ax, segments)

    args = (
        ax.plot([], [], color="blue", zorder=5)[0],
        ax.plot([], [], color="red", marker="o",
                linestyle="none", zorder=4)[0],
        ax.plot([], [], color="green", linewidth=3, zorder=3)[0],
        ax.plot([], [], color="cyan", marker="o", zorder=3)[0],
        ax.plot([], [], color="cyan", marker="o", zorder=3)[0],
        ax.plot([], [], color="red", linewidth=3, zorder=3)[0],
    )

    max_frames = 10000
    frames = max_frames - \
        _frame(max_frames, segments, *args) + 2

    ani = anim.FuncAnimation(fig, _frame, frames=frames,
                             fargs=(segments, *args))
    plt.close(fig)
    return ani


ani = animate(SETS["D"])
display(ani)

# %% [markdown]
# ### Zapis animacji do pliku GIF
# %%
if SAVE_FILES:
    ani.save("Lab4Raport/ani.gif")

# %% [markdown]
# ### Zbiór testowy
# %%
app2 = Application(SETS["X"])
animate(app2.segments)

# %% [markdown]
# ### Zbiór testowy
# %%
app3 = Application(SETS["K"])
animate(app3.segments)

# %% [markdown]
# ### Zbiór losowy
# %%
app4 = Application(gen_n=15)
animate(app4.segments)
