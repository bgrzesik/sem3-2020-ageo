# %% [markdown]
# ### Importy
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
# ### Zbiory testowe
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

    "X1": [[75.85227272727275, 62.29128014842303],
           [50.24930426716142, 74.90723562152138],
           [-74.79707792207792, -0.41743970315397405],
           [49.50718923933212, -74.99999999999997],
           [76.2233302411874, -63.868274582560275],
           [49.50718923933212, -50.13914656771797],
           [75.85227272727275, -36.781076066790334],
           [51.3624768089054, -25.649350649350623],
           [75.4812152133581, -11.92022263450832],
           [49.87824675324677, -0.41743970315397405],
           [74.73910018552877, 12.940630797773679],
           [49.87824675324677, 24.814471243042703],
           [75.4812152133581, 37.05936920222638],
           [50.99141929499075, 49.6753246753247]],

    "X2": [[-49.594155844155836, 75.28988868274587],
           [-74.8260667903525, 63.78710575139149],
           [-49.594155844155836, 49.68692022263454],
           [-74.8260667903525, 36.32884972170689],
           [-50.33627087198515, 25.56818181818184],
           [-75.19712430426715, 11.096938775510239],
           [-49.594155844155836, -0.7769016697587858],
           [-75.56818181818181, -13.392857142857125],
           [-49.223098330241186, -25.266697588126135],
           [-74.45500927643783, -37.88265306122446],
           [-50.33627087198515, -49.756493506493484],
           [-75.19712430426715, -63.11456400742112],
           [-51.078385899814464, -74.98840445269015],
           [75.45222634508352, -0.4058441558441359]],

    "S": [[84.35760667903526, 93.83116883116887],
          [29.441094619666075, 91.23376623376626],
          [23.133116883116884, 91.23376623376626],
          [0.498608534322841, 90.12059369202228],
          [-8.035714285714278, 88.63636363636368],
          [-14.34369202226344, 87.5231910946197],
          [-29.55705009276437, 84.55473098330245],
          [-45.512523191094616, 80.10204081632656],
          [-53.675788497217056, 75.64935064935068],
          [-59.24165120593692, 72.30983302411877],
          [-69.26020408163265, 65.25974025974028],
          [-74.8260667903525, 60.43599257884975],
          [-81.50510204081633, 53.75695732838594],
          [-83.73144712430425, 47.820037105751425],
          [-84.47356215213358, 43.36734693877554],
          [-83.73144712430425, 35.9461966604824],
          [-80.0208719851577, 26.66975881261598],
          [-74.08395176252318, 20.36178107606682],
          [-67.77597402597402, 16.280148423005585],
          [-55.902133580705, 11.45640074211505],
          [-40.68877551020407, 6.261595547309867],
          [-35.12291280148422, 4.4063079777365886],
          [-16.941094619666032, -1.1595547309832739],
          [-5.067254174397021, -4.870129870129844],
          [3.838126159554747, -9.322820037105728],
          [14.598794063079794, -18.228200371057483],
          [19.051484230055678, -27.504638218923915],
          [20.164656771799656, -36.03896103896102],
          [17.9383116883117, -44.573283858998124],
          [13.114564007421166, -50.13914656771797],
          [7.548701298701303, -56.447124304267135],
          [-8.406771799628928, -64.23933209647494],
          [-16.941094619666032, -64.98144712430425],
          [-38.091372912801475, -71.28942486085342],
          [-64.06539888682745, -78.33951762523189],
          [-72.2286641929499, -81.67903525046381],
          [-83.3603896103896, -85.38961038961037],
          [-95.60528756957328, -94.29499072356214],
          [-74.8260667903525, -96.52133580705006],
          [-60.35482374768088, -93.18181818181816],
          [-54.04684601113172, -92.0686456400742],
          [-14.34369202226344, -89.10018552875694],
          [3.096011131725419, -82.05009276437846],
          [20.164656771799656, -73.88682745825601],
          [30.925324675324703, -64.98144712430425],
          [36.12012987012989, -53.84972170686454],
          [39.45964749536179, -44.573283858998124],
          [37.604359925788515, -26.7625231910946],
          [33.89378478664193, -17.48608534322817],
          [23.504174397031562, -4.12801484230053],
          [7.177643784786653, 4.4063079777365886],
          [-8.777829313543577, 9.230055658627123],
          [-22.506957328385894, 15.166975881261635],
          [-38.462430426716125, 26.29870129870133],
          [-46.62569573283858, 36.31725417439705],
          [-49.594155844155836, 46.70686456400745],
          [-39.5756029684601, 65.25974025974028],
          [-19.167439703153974, 76.76252319109466],
          [3.838126159554747, 84.1836734693878],
          [26.47263450834882, 87.89424860853435]],
}
# %% [markdown]
# ### Funkcja rozdzielania punktów na dwa łańcuchy
# Funkcja zwraca dwie listy  (przykładowo `([0, 4, ...], [8, 7, ...])`),
# które zawierają kolejne indeksy punktów z tablicy wejściowej. Indeksy punktów
# są w kolejności od najwyższego do najmniejszego.
# %%


def split_chains(points, axis=1):
    p_max, _ = max(enumerate(points), key=lambda ip: ip[1][axis])
    p_min, _ = min(enumerate(points), key=lambda ip: ip[1][axis])

    chain_a, chain_b = [], []

    i = p_max
    while i != p_min:
        chain_a.append(i)
        i += 1
        i %= len(points)

    chain_a.append(i)

    while i != p_max:
        chain_b.append(i)
        i += 1
        i %= len(points)

    chain_b.append(i)

    chain_b.reverse()
    return chain_a, chain_b

# %% [markdown]
# ### Testowe uruchomienie funkcji oraz funkcja wizualizacji
# %%


def plot_points(ax, points, indices, **kw_args):
    points = np.array(points)[indices]
    ax.plot(points[:, 0], points[:, 1], **kw_args)


points = SETS["B"]
fig, ax = plt.subplots()
left, right = split_chains(points)
fig.show()

plot_points(ax, points, left, color="b")
plot_points(ax, points, right, color="r")

# %% [markdown]
# ### Funkcja sprawdzająca monotoniczność
# %%


def check_monotone(points, axis=1):
    left, right = split_chains(points, axis=axis)

    for i in range(1, len(left)):
        if points[left[i]][axis] > points[left[i - 1]][axis]:
            return False

    for i in range(1, len(right)):
        if points[right[i]][axis] > points[right[i - 1]][axis]:
            return False

    return True


# %% [markdown]
# ### Testowe uruchomienie funkcji i wizualizacja
# %%
print(check_monotone(SETS["B"]))
print(check_monotone(SETS["nonB"]))
# %% [markdown]
# ### Funkcja sortująca punkty wg Y w czasie $ O(n) $
# Zdefniowany został też tym enumeryczny `Chain`, umożliwiający identyfikacje na którym
# łacuchu leży dany punkt. Wartości zostały przypisane jako maska bitowa `0b01`
# oznacza że punkt leży na lewym łacuchu, a `0b10` że punkt leży na prawym łacychu.
# Alternatywa tych flag `0b11 (0b01 | 0b10)`, oznacza że punkt leży na obu
# łacuchach - punkty najwyżej lub najniżej.
# Funkcja wraca listę krotek, gdzie pierwsza wartość to indeks w tablicy
# wejściowych punktów, a druga to maska bitowa typu `Chain`.
# %%


class Chain(int, Flag):
    LEFT, RIGHT, BOTH = 0b01, 0b10, 0b11


def chain_sort(points, left, right):
    li, ri = 1, 1
    indices = [(left[0], Chain.BOTH)]

    while li < len(left) - 1 or ri < len(right) - 1:
        if li >= len(left) - 1:
            indices.append((right[ri], Chain.RIGHT))
            ri += 1
        elif ri >= len(right) - 1:
            indices.append((left[li], Chain.LEFT))
            li += 1
        elif points[left[li]][1] >= points[right[ri]][1]:
            indices.append((left[li], Chain.LEFT))
            li += 1
        elif points[left[li]][1] < points[right[ri]][1]:
            indices.append((right[ri], Chain.RIGHT))
            ri += 1
        else:
            assert False

    indices.append((left[-1], Chain.BOTH))
    return indices

# %% [markdown]
# ### Implementacja triangulacji wielokątów monotonicznych.
# Funkcja zwaca listę krotek indeksów punktów w tabeli wejściowej.
# %%


def monotone_triangulate(points, axis=1):
    left, right = split_chains(points, axis=axis)
    indices = chain_sort(points, left, right)
    queue = indices[:2]
    triangles = []

    def _is(x, a):
        return (x & a) == a

    del left, right

    for i in range(2, len(points)):
        p = indices[i]

        if (queue[-1][1] | p[1]) == Chain.BOTH:
            while len(queue) >= 2:
                a = queue.pop()
                triangles.append((a[0], queue[-1][0], p[0]))

            queue = [indices[i - 1], indices[i]]
        else:
            while len(queue) >= 2:
                ori = orient(points[queue[-2][0]],
                             points[p[0]],
                             points[queue[-1][0]])

                if (_is(p[1], Chain.LEFT) and ori == Orient.CW) \
                        or (_is(p[1], Chain.RIGHT) and ori == Orient.CCW):

                    triangles.append((queue[-2][0], queue[-1][0], p[0]))
                    queue.pop()
                else:
                    break
            queue.append(p)

    return triangles


# %% [markdown]
# ### Testowe uruchomienie oraz wizualizacja
# %%
points = SETS["A"]
fig, ax = plt.subplots()
left, right = split_chains(points)

plot_points(ax, points, left, color="b")
plot_points(ax, points, right, color="r")

triangles = monotone_triangulate(points)
print(triangles)

for a, b, c in triangles:
    x = [points[a][0], points[b][0], points[c][0]]
    y = [points[a][1], points[b][1], points[c][1]]
    ax.fill(x, y, alpha=0.3)

# %% [markdown]
# ### Funkcja klasyfikacji punktów
# Zdefiniowany został pomocniczy typ enumeryczny `Classification`, który służy
# przedstawia klasyfikację punktu.
# Funkcja zwraca tablicę wartości enumerycznych `Classification`, gdzie wartość
# dotyczy punktu pod takim samym indeksem w tablicy wejściowej.
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

        result[i] = cl

    return result


# %% [markdown]
# ### Testowe uruchomienie
# %%


CLASSIFICATION_COLORS = {
    Classification.STARTING: "#5CC83B",
    Classification.FINISHING: "#EB3223",
    Classification.JOINING: "#323693",
    Classification.DIVIDING: "#C2DFE2",
    Classification.REGULAR: "#60350F",
}


def plot_classification(ax, points, cl):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    c = [CLASSIFICATION_COLORS[p] for p in cl]

    sc1 = ax.scatter(x, y, color="black", marker="o", linewidth=3, zorder=2)
    sc2 = ax.scatter(x, y, c=c, marker="o", linewidth=1, zorder=3)

    return sc1, sc2


points = SETS["W"]
cl = classify_points(SETS["W"])

fig, ax = plt.subplots()
plot_points(ax, points, list(range(len(points))) + [0], color="b", marker="")
plot_classification(ax, points, cl)
fig.show()
# %% [markdown]
# ### Implementacja programu wizualizacji oraz wprowadzania punktów i testowania wybranych funkcji
# Zdefniniowana została klasa przchowująca zbiorczo funkcjonalonść. Użeteczne
# podczas testowania mogą być funkcje obiektu:
# * `app.set_points(...)` - funkcja ustawiająca punkty na podane punkty
# * `app.check_monotone(None)` - funkcja sprawdzająca monotoczność wielokąta
# * `app.triangulate(None)` - funkcja dokonująca triangulacji
# * `app.classify(None)` - funkcja dokonująca klasyfikacji punktów
# * `app.set_circle(None)` - funkcja ustawiająca zbiór punktów na zbiór mający
#                            imitować okrąg
# * `app.set_harmonia(None)` - funkcja zbiór punktów na zbiór przypominający
#                              harmonijkę
# * `app.reset(None)` - usuwa wszystkie punkty
#
# Punkty można wprowadzać za pomocą myszki, klikając na wybrany punkt na
# płaszczyźnie na którym chcemy dodać punkt. Po prawej stronie znajduje się lista
# przycisków:
# * **Sprawdź monotoczność** - uruchomienie sprawdzania Y-mononotoczniości oraz
#                              wyświetlenie wyniku nad wizualizacją
# * **Trianguluj** - uruchomienie algorytmu triangulacji oraz pokazanie podziału
# * **Reset** - usunięcie wszystkich punktów i wyczyszczenie wizualizacji
# * **Okrąg** - zresetowanie wizualizacji i dodanie okręgu
# * **Harmonijka** - zresetowanie wizualizacji i dodanie "harmonijki"
# * **Zklasyfikuj** - uruchomienie klasyfikacji punktów i naniesienie klas na
#                     wizualizacje
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
        self.circle_btn = wig.Button(btn_ax, "Okrąg")
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

        arr = np.array(self.points)
        self.line_l.set_data(arr[left].T)
        self.line_r.set_data(arr[right].T)

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
            x = [self.points[a][0], self.points[b][0], self.points[c][0]]
            y = [self.points[a][1], self.points[b][1], self.points[c][1]]
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
        cl = classify_points(self.points)
        self.class_axes = plot_classification(self.ax, self.points, cl)

    def clear_classify(self):
        if self.class_axes:
            self.class_axes[0].remove()
            self.class_axes[1].remove()
            self.class_axes = None


app = Application()
app.set_points(SETS["S"])

# %% [markdown]
# ### Animacja algorytmu triangulacji
# Wykonanie animacji zostało podzielone na trzy fazy, w pierwszej dokonywany jest
# podział punktów na łańcuchy lewy i prawy. W drugiej punkty są sortowane na
# podstawie łańcuchów od najwyższego do najniżeszego. W trzeciej fazie
# przedstawiony jest algorytm triangulacji.
# %%


def _set_data(ax, points, indices):
    x = [points[i][0] for i in indices]
    y = [points[i][1] for i in indices]
    ax.set_data(x, y)


def _frame_phase1(frame, points, black, blue, red):
    axis = 1
    p_max, _ = max(enumerate(points), key=lambda ip: ip[1][axis])
    p_min, _ = min(enumerate(points), key=lambda ip: ip[1][axis])

    frame -= 1
    black.set_linestyle("solid")
    black.set_marker("")
    black.set_data(
        [-110, 110, 110, -110],
        [points[p_max][1], points[p_max][1], points[p_min][1], points[p_min][1]]
    )

    chain_a, chain_b = [], []

    i = p_max
    while i != p_min and frame > 0:
        chain_a.append(i)
        i += 1
        i %= len(points)
        frame -= 1

    if frame > 0:
        frame -= 1
        chain_a.append(i)

    while i != p_max and frame > 0:
        chain_b.append(i)
        i += 1
        i %= len(points)
        frame -= 1

    if frame > 0:
        frame -= 1
        chain_b.append(i)

    _set_data(blue, points, chain_a)
    _set_data(red, points, chain_b)

    chain_b.reverse()
    return frame, chain_a, chain_b


def _frame_phase2(frame, points, left, right, black):
    li, ri = 1, 1
    indices = [(left[0], Chain.BOTH)]

    while (li < len(left) - 1 or ri < len(right) - 1) and frame > 0:
        frame -= 1

        if li >= len(left) - 1:
            indices.append((right[ri], Chain.RIGHT))
            ri += 1
        elif ri >= len(right) - 1:
            indices.append((left[li], Chain.LEFT))
            li += 1
        elif points[left[li]][1] >= points[right[ri]][1]:
            indices.append((left[li], Chain.LEFT))
            li += 1
        elif points[left[li]][1] < points[right[ri]][1]:
            indices.append((right[ri], Chain.RIGHT))
            ri += 1
        else:
            assert False

    if frame > 0:
        indices.append((left[-1], Chain.LEFT))
        frame -= 2

    black.set_linestyle("solid")
    black.set_marker("")
    black.set_data(
        [-110, 110],
        [points[indices[-1][0]][1], points[indices[-1][0]][1]]
    )

    return frame, indices


def _frame_phase3(frame, points, indices, black, green, ax, polygons):
    axis = 1
    queue = indices[:2]
    triangles = []

    def _stop(i):
        nonlocal frame, black, green, ax, polygons, triangles

        black.set_linestyle("none")
        black.set_marker("o")

        _set_data(black, points, [q[0] for q in queue])
        _set_data(green, points, [i])

        for a, b, c in triangles[len(polygons):]:
            polygons.append(ax.fill(
                [points[a][0], points[b][0], points[c][0]],
                [points[a][1], points[b][1], points[c][1]]
            )[0])

        return frame

    def _is(x, a):
        return (x & a) == a

    if frame < 0:
        return _stop(2)
    frame -= 1

    for i in range(2, len(points)):
        if frame < 0:
            return _stop(i)
        frame -= 1

        p = indices[i]

        if (queue[-1][1] | p[1]) == Chain.BOTH:
            while len(queue) >= 2:
                if frame < 0:
                    return _stop(i)
                frame -= 1

                a = queue.pop()
                triangles.append((a[0], queue[-1][0], p[0]))

            queue = [indices[i - 1], indices[i]]
        else:
            while len(queue) >= 2:
                if frame < 0:
                    return _stop(i)
                frame -= 1

                ori = orient(points[queue[-2][0]],
                             points[p[0]],
                             points[queue[-1][0]])

                if (_is(p[1], Chain.LEFT) and ori == Orient.CW) \
                        or (_is(p[1], Chain.RIGHT) and ori == Orient.CCW):

                    triangles.append((queue[-2][0], queue[-1][0], p[0]))
                    queue.pop()
                else:
                    break
            queue.append(p)

    return _stop(i)


def _frame(frame, points, black, blue, red, green, ax, polygons):
    while frame == 0 and polygons:
        polygons.pop().remove()

    green.set_data([], [])

    frame, left, right = _frame_phase1(frame, points, black, blue, red)
    if frame > 0:
        frame, indices = _frame_phase2(frame, points, left, right, black)

    if frame > 0:
        frame = _frame_phase3(frame, points, indices,
                              black, green, ax, polygons)

    if frame > 0:
        black.set_data([], [])
        green.set_data([], [])
        frame -= 2

    return frame


def animate(points):
    fig, ax = plt.subplots()
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    ax.scatter(
        [p[0] for p in points],
        [p[1] for p in points],
    )
    black, = ax.plot([], [], color="black", marker="o", zorder=3)
    green, = ax.plot([], [], color="#33ff33", marker="o", zorder=3)
    blue, = ax.plot([], [], color="blue", marker="o", zorder=2)
    red, = ax.plot([], [], color="red", marker="o", zorder=2)
    polygons = []

    max_frames = 10000
    frames = max_frames - \
        _frame(max_frames, points, black, blue, red, green, ax, polygons) + 2

    ani = anim.FuncAnimation(fig, _frame, frames=frames,
                             fargs=(points, black, blue, red, green, ax, polygons))

    plt.close(fig)
    return ani


ani = animate(SETS["D"])
ani
# %% [markdown]
# ### Zapisanie animacji do pliku gif
# %%
if SAVE_FILES:
    ani.save("Lab3Raport/triangulate.gif")

# %% [markdown]
# ### Tą komórką można uruchomić animacje z widoku dodawania punktów
# %%
if len(app.points) >= 3:
    display(animate(app.points))

# %%
