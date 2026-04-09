import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import Hopfield

ROWS, COLS = 7, 5

class GridDrawer:
    def __init__(self, hopfield: Hopfield.Hopfield, rows=ROWS, cols=COLS):
        self.hopfield = hopfield

        self.rows, self.cols = rows, cols
        self.data = np.zeros((rows, cols), dtype=int)

        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.im = self.ax.imshow(self.data, cmap="gray_r", vmin=0, vmax=1,
                                 extent=[0, cols, 0, rows], origin="upper")

        # grid lines
        self.ax.set_xticks(np.arange(0, cols+1, 1))
        self.ax.set_yticks(np.arange(0, rows+1, 1))
        self.ax.grid(which="both", color="black", linewidth=0.5)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_title("Click to draw")

        # buttons
        ax_clear = plt.axes([0.1, 0.05, 0.15, 0.075])
        ax_run = plt.axes([0.3, 0.05, 0.15, 0.075])
        self.btn_clear = Button(ax_clear, "Clear")
        self.btn_run = Button(ax_run, "Retrieve")

        self.btn_clear.on_clicked(self.clear)
        self.btn_run.on_clicked(self.retrieve)

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        col = int(event.xdata)
        row = self.rows - int(event.ydata) - 1
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row, col] = 1 - self.data[row, col]  # toggle
            self.im.set_data(self.data)
            self.fig.canvas.draw()

    def clear(self, event=None):
        self.data[:] = 0
        self.im.set_data(self.data)
        self.fig.canvas.draw()

    def retrieve(self, event=None):
        flat = self.data.flatten()
        y = self.hopfield.recuperar(flat)

        fig, ax = plt.subplots(1, len(y))
        if len(y) == 1:
            ax = [ax]
        for i, out in enumerate(y):
            img = out.reshape(self.rows, self.cols)
            ax[i].imshow(img, cmap="gray_r", vmin=0, vmax=1,
                         extent=[0, self.cols, 0, self.rows], origin="upper")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"Step {i}")
        plt.show()