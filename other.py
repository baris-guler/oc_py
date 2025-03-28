import matplotlib.pyplot as plt
import numpy as np
import time

class ClickAndDragSelector:
    def __init__(self, ax, fig, Ecorr, OC):
        self.save = False
        self.ax = ax
        self.fig = fig
        self.start_point = None
        self.rect = None
        self.Ecorr = Ecorr
        self.OC = OC
        self.selected = np.full_like(Ecorr, False, dtype=bool)
        self.drawing = False  # Flag to indicate if the rectangle is currently being drawn
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press) 
        self.ax.figure.canvas.mpl_connect('close_event', self.on_close)
        plt.text(0.05, 0.95, 'Press "enter" to save selection, Press "esc" to cancel', transform=plt.gcf().transFigure, va='top') 


    def on_press(self, event):
        if event.button == 1 and self.ax.get_navigate_mode() != "ZOOM" and self.ax.get_navigate_mode() != "PAN":  # Left mouse button
            self.press_time = time.time()
            if not self.drawing:  # Check if not already drawing
                self.start_point = (event.xdata, event.ydata)
                self.rect = plt.Rectangle((0, 0), 0, 0, alpha=0.5)
                self.ax.add_patch(self.rect)
                if hasattr(self, 'plot3') and event.xdata is not None and event.ydata is not None:
                    for plot in self.plot3:
                        if plot is not None:
                            plot.remove()
                self.drawing = True

    def on_motion(self, event):
        if self.start_point is not None and self.drawing and event.xdata is not None and event.ydata is not None:
            width = event.xdata - self.start_point[0]
            height = event.ydata - self.start_point[1]
            self.rect.set_width(width)
            self.rect.set_height(height)
            self.rect.set_xy(self.start_point)
            self.ax.figure.canvas.draw()
        if event.xdata is not None and event.ydata is not None and not self.drawing:
            if hasattr(self, 'plot3'):
                for plot in self.plot3:
                    if plot is not None:
                        plot.remove()
            diff_x = (self.Ecorr - event.xdata) / np.abs(np.max(self.Ecorr) - np.min(self.Ecorr))
            diff_y = (self.OC - event.ydata) / np.abs(np.max(self.OC) - np.min(self.OC))

            distances = np.sqrt(diff_x**2 + diff_y**2)
            self.closest_index = np.argmin(distances)

            if self.selected[self.closest_index]:
                self.plot3 = self.ax.plot(self.Ecorr[self.closest_index], self.OC[self.closest_index], 'x', color='red', markersize=15)
            else:
                self.plot3 = self.ax.plot(self.Ecorr[self.closest_index], self.OC[self.closest_index], '.', color='blue', markersize=15)
            self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1 and self.drawing:  # Left mouse button and drawing
            release_time = time.time()
            duration = release_time - self.press_time
            if duration < 0.12:  
                self.clicked(event)
            self.drawing = False  # Set drawing flag to False upon release
            if self.rect:
                self.rect.remove()  # Remove the existing rectangle
            if event.xdata is not None and event.ydata is not None:
                self.spx = self.start_point[0]
                self.spy = self.start_point[1]
                if self.spx > event.xdata:
                    self.spx, self.epx = event.xdata, self.spx
                else:
                    self.epx = event.xdata
                if self.spy > event.ydata:
                    self.spy, self.epy = event.ydata, self.spy
                else:
                    self.epy = event.ydata
                
                changed = (self.Ecorr >= self.spx) & (self.Ecorr <= self.epx) & (self.OC >= self.spy) & (self.OC <= self.epy)
                self.selected[changed] = ~self.selected[changed]
                # Find the scatter plot object in the axes and update marker style   
            else:
                print("No valid selection. Don't go outside of the graph when drawing")
            if hasattr(self, 'plot2'):
                for plot in self.plot2:
                    plot.remove()
            self.plot2 = self.ax.plot(self.Ecorr[self.selected], self.OC[self.selected], 'x', color='red')
            self.plot3 = self.ax.plot([],[])
            self.ax.figure.canvas.draw_idle()  # Force canvas redraw

    def on_key_press(self, event):
        if event.key == "enter":
            self.save = True
            plt.close()
        elif event.key == "escape":
            self.save = False
            self.selected = None
            plt.close()

    def on_close(self, event):
        if self.save:
            pass
        else:
            self.selected = None

    def clicked(self, event):
        if event.button == 1 and self.ax.get_navigate_mode() != "ZOOM":
            if hasattr(self, 'plot2'):
                for plot in self.plot2:
                    try:
                        plot.remove()
                    except:
                        pass
            if event.xdata is not None and event.ydata is not None:
                self.selected[self.closest_index] = ~self.selected[self.closest_index]
                self.plot2 = self.ax.plot(self.Ecorr[self.selected], self.OC[self.selected], 'x', color='red')
                self.plot3 = self.ax.plot([],[])
            self.ax.figure.canvas.draw_idle()  # Force canvas redraw
            self.on_motion(event)

class Vertical_selector:
    def __init__(self, ax, fig, Ecorr, OC):
        self.save = False
        self.ax = ax
        self.fig = fig
        self.vlines = []
        self.vlines_x = []
        self.Ecorr = Ecorr
        self.OC = OC
        self.selected = np.full_like(Ecorr, False, dtype=bool)
        self.drawing = False  # Flag to indicate if the rectangle is currently being drawn
        
        # Registering event handlers to the axis object
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press) 
        self.ax.figure.canvas.mpl_connect('close_event', self.on_close)
        
        plt.text(0.05, 0.95, 'Press v to draw vertical line, r to revert vertical line \n'
                             'Press "enter" to save selection, Press "esc" to cancel and quit', 
                 transform=plt.gcf().transFigure, va='top') 
    
    def on_key_press(self, event):
        if event.key == "v" and event.xdata is not None and len(self.vlines) < 2:
            vline = self.ax.axvline(x=event.xdata, color='red')
            self.vlines.append(vline)
            self.ax.figure.canvas.draw_idle()
        elif event.key == "v" and len(self.vlines) == 2:
            print("Maximum number of vertical lines reached")
        elif event.key == "r" and len(self.vlines)>0:
            self.vlines[-1].remove()
            self.vlines = self.vlines[:-1]
            self.ax.figure.canvas.draw_idle()
        elif event.key == "enter" and len(self.vlines) == 2:
            self.save = True
            plt.close()
        elif event.key == "enter" and len(self.vlines) < 2:
            print("Please draw two vertical lines")
        elif event.key == "escape":
            self.save= False
            plt.close()

    def on_close(self, event):
        if self.save:
            self.vlines_x = [vline.get_xdata()[0] for vline in self.vlines]
            self.vlines_x.sort()
        else:
            self.vlines_x = []