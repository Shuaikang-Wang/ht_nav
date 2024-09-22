import numpy as np
import shapely
from shapely.plotting import plot_polygon, plot_points


def intersect2(a, b, c, d):
    # standard form line eq Line_AB
    a1 = b[1] - a[1]
    b1 = a[0] - b[0]
    c1 = a1 * a[0] + b1 * a[1]

    # standard form line eq Line_CD
    a2 = d[1] - c[1]
    b2 = c[0] - d[0]
    c2 = a2 * c[0] + b2 * c[1]

    determinant = a1 * b2 - a2 * b1

    if (determinant == 0):
        return np.inf, np.inf
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return np.array([x, y])


class Building(object):
    def __init__(self, anchors: np.ndarray, safe_margin: float) -> None:
        self.anchors = anchors
        self.polygon = shapely.Polygon(self.anchors)

        self.center = np.asarray(self.polygon.centroid.coords).squeeze()

        self.safe_margin = safe_margin

        self.debug = False
        self.generate_expansion_polygon()

    def generate_expansion_polygon(self):

        # determine if a list of polygon points are in clockwise order?
        sum = 0
        for i in range(self.anchors.shape[0]):
            j = (i + 1) % self.anchors.shape[0]
            x1, y1 = self.anchors[i]
            x2, y2 = self.anchors[j]
            sum += (x2 - x1) * (y2 + y1)
        if sum > 0:  # in clockwise order
            side = 'left'
        else:  # counter-clockwise
            side = 'right'

        lines: list[shapely.LineString] = []
        for i in range(self.anchors.shape[0]):
            j = (i + 1) % self.anchors.shape[0]
            lines.append(shapely.LineString([self.anchors[i], self.anchors[j]]).parallel_offset(self.safe_margin, side))

        exp_anchors = np.zeros((0, 2))
        for i in range(self.anchors.shape[0]):
            j = (i + 1) % self.anchors.shape[0]
            a, b = np.array(lines[i].coords)
            c, d = np.array(lines[j].coords)
            pt = intersect2(a, b, c, d)
            exp_anchors = np.vstack((exp_anchors, pt))

        self.expansion_polygon = shapely.Polygon(exp_anchors)
        self.expansion_anchors = exp_anchors

        if self.debug:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_polygon(self.polygon)
            plot_polygon(self.expansion_polygon, edgecolor="r")
            plot_points(self.polygon.centroid)
            plot_points(self.expansion_polygon.centroid, color="red")
            # plot_polygon(b3, edgecolor="green")
            # plot_polygon(b4, edgecolor="k")
            # for i in range(self.anchors.shape[0]):
            #     plot_line(lines[i], color='k')

            plt.show()
            print("debug")


if __name__ == "__main__":
    vertices = [(2.719999400006, 1.166712154433349),
  (2.71999940000587, 0.6800034725663933),
  (1.880001999966867, 0.6800005999940002),
  (1.8800005999940999, 1.384759496087541),
  (2.0000005999940997, 1.384759496087541),
  (2.0000005999940997, 0.8000005999940001),
  (2.5999994000058697, 0.8000005999940001),
  (2.5999994000058697, 1.265552266578111),
  (2.599999400006, 1.6034091001902981),
  (2.719999400006, 1.6034091001902981)]


    b = Building(np.array(vertices), 5)
    print("new vertices", b.expansion_anchors)
