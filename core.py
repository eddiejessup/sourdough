import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

class PointPicker(object):
    ''' Pick points randomly
    '''
    def __init__(self, rs):
        self.rs = rs

    def pick(self):
        return self.rs[np.random.randint(len(self.rs))]

class StreamPicker(PointPicker):
    ''' Pick points from a text file
    '''
    def __init__(self, rs, f, cs):
        if len(rs) != len(cs):
            raise Exception('Require as many points as characters')
        self.rs = rs
        self.f = f
        self.cs = cs

    def pick(self):
        while True:
            c = self.f.read(1)
            if not len(c): raise EOFError
            if c in self.cs: break
        return self.rs[self.cs.index(c)]

class DonkeyWalk(object):
    ''' Iterate the Donkey's Walk algorithm
        (Note: nobody else calls it that, I don't know if it has a name)
    '''
    def __init__(self, x_0, d, picker):
        self.x = x_0
        self.d = d
        self.picker = picker

    def iterate(self):
        self.x += d * (self.picker.pick() - self.x)

class DonkeyWalkLogger(object):
    ''' Keep track of the state of the algorithm iterator's state
    '''
    def __init__(self, dw, dat_max=np.inf):
        ''' dw: algorithm iterator object
            dat_max: maximum number of states to track before complaining
        '''
        self.dw = dw
        self.dat_max = dat_max
        self.xs = []

    def log(self):
        # Save current x
        self.xs.append(self.dw.x.copy())
        # If data has been filled, complain
        if len(self.xs) > self.dat_max:
            raise EOFError

    def plot(self, fname=None):
        fig = pp.figure(frameon=False)
        ax = fig.gca()
        ax.set_aspect('equal')

        # Boring plotty things
        rs = self.dw.picker.rs
        x_max, x_min, y_max, y_min = rs[:, 0].max(), rs[:, 0].min(), rs[:, 1].max(), rs[:, 1].min()
        x_range = x_max - x_min
        y_range = y_max - y_min
        b = 0.1
        ax.set_xticks([])
        ax.set_yticks([])

        xs = np.array(self.xs)

        # ax.scatter(*xs.T, s=0.1)
        H, xedges, yedges = np.histogram2d(xs[:,0], xs[:,1], bins=2*[400])
        extent = [x_min, x_max, y_min, y_max]
        ax.imshow(H.T, extent=extent, interpolation='bicubic', origin='lower', norm=mpl.colors.LogNorm())

        # Plot fixed points
        ax.scatter(*rs.T, s=40, c='black', lw=0)
        # If picker is using a stream, plot the associated characters
        if isinstance(self.dw.picker, StreamPicker):
            for i in range(len(rs)):
                rtxt = 1.1 * rs[i]
                ax.text(rtxt[0], rtxt[1], self.dw.picker.cs[i], fontsize=20, color='red', ha='center', va='center')
        print(fname, np.std(H))

        ax.set_xlim([x_min-b*x_range, x_max+b*x_range])
        ax.set_ylim([y_min-b*y_range, y_max+b*y_range])

        # ax.set_title('%s' % len(self.xs))
        if fname is not None: fig.savefig('%s.png' % fname, bbox_inches='tight')
        else: fig.canvas.show()

# Dimensionality (only works in 2D but nicer than writing 2 everywhere)
dim = 2
# Displacement fraction, 0 <= d <= 1
d = 0.5
# Number of fixed points
n = 4

def main():
    # Set initial position
    x_0 = np.array([0.0, 0.0], dtype=np.float)

    # Generate points of a regular polygon with n sides
    rs = np.zeros([n, dim], dtype=np.float)
    for i in range(n):
        rs[i, 0] = np.cos(2.0 * np.pi * (float(i) / n))
        rs[i, 1] = np.sin(2.0 * np.pi * (float(i) / n))

    # Make a point picker object
    fname = 'rand_%i_%g' % (n, d)
    picker = PointPicker(rs)
    fname = 'me_low'
    # picker = StreamPicker(rs, open('Data/%s.txt' % fname, 'r'), ('A', 'T', 'G', 'C'))
    # picker = StreamPicker(rs, open('Data/%s.txt' % fname, 'r'), ('a', 'e', 'i', 'o', 'u'))
    picker = StreamPicker(rs, open('Data/%s.txt' % fname, 'r'), ('a', 'e', 'i', 'o'))

    # Make the algorithm iterator object
    dw = DonkeyWalk(x_0, d, picker)
    # Make an object to log the state
    dwl = DonkeyWalkLogger(dw, dat_max=1573150)
    # Do the iteration
    while True:
        try:
            dw.iterate()
            dwl.log()
        except EOFError:
            break

    dwl.plot('Images/%s' % fname)

if __name__ == '__main__':
    main()