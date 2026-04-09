import numpy as np
import matplotlib.pyplot as plt
import Hopfield
import GridDrawer

def show_bitmap(vec, ax):
    vec = np.asarray(vec)

    img = vec.reshape(7, 5)

    img = 1 - (img if img.max() <= 1 else img / img.max())

    # Force black/white rendering
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 8, 1))
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest', origin='upper', extent=[0, 5, 0, 7])
    ax.grid(which="both", color="black", linewidth=0.5)

n0 = np.array([
    0, 0, 1, 1, 0,
    0, 1, 0, 0, 1,
    0, 1, 0, 0, 1,
    0, 1, 0, 0, 1,
    0, 1, 0, 0, 1,
    0, 1, 0, 0, 1,
    0, 0, 1, 1, 0
])
n1 = np.array([
    0, 0, 1, 0, 0,
    0, 1, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 1, 1, 1, 0
])
n2 = np.array([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    0, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    0, 1, 1, 1, 1
])
n3 = np.array([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    0, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    0, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])
'''
n4 = np.array([
    0, 0, 0, 1, 0,
    0, 0, 1, 1, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 1, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0
])
n5 = np.array([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 0,
    0, 0, 0, 0, 1,
    0, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])
n6 = np.array([
    0, 0, 1, 1, 0,
    0, 1, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])
n7 = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    0, 0, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 1, 0, 0, 0
])
n8 = np.array([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])
n9 = np.array([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    0, 0, 0, 1, 0,
    0, 1, 1, 0, 0
])
fig, ax = plt.subplots(3, 4)
show_bitmap(n0, ax[0,0])
show_bitmap(n1, ax[0,1])
show_bitmap(n2, ax[0,2])
show_bitmap(n3, ax[0,3])
show_bitmap(n4, ax[1,0])
show_bitmap(n5, ax[1,1])
show_bitmap(n6, ax[1,2])
show_bitmap(n7, ax[1,3])
show_bitmap(n8, ax[2,0])
show_bitmap(n9, ax[2,1])
'''

# ENTRENAR
hopfield = Hopfield.Hopfield()
hopfield.entrenar(np.array([n0,n1,n2,n3]))

drawer = GridDrawer.GridDrawer(hopfield=hopfield)
plt.show()