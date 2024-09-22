import numpy as np
import math

from ENV.geometry import Squircle


class Line(object):
    def __init__(self, points):
        """
        :param points: [start_point, end_point]
        """
        self.points = points
        self.is_extended = False


class PolygonPartition(object):
    def __init__(self, polygon_vertices):
        """
        param polygon_vertices: vertex in anticlockwise
        """
        self.vertices = polygon_vertices
        self.sides = None
        self.separated_triangles = None
        self.round_num = 5

        self.round_vertices()
        self.construct_sides()

    def round_vertices(self):
        new_polygon = []
        for vertex in self.vertices:
            new_vertex = [np.round(vertex[0], self.round_num),
                          np.round(vertex[1], self.round_num)]
            new_polygon.append(new_vertex)

        self.vertices = new_polygon

    def construct_sides(self):
        self.sides = []
        for index in range(len(self.vertices) - 1):
            end_points = [self.vertices[index], self.vertices[index + 1]]
            side = Line(end_points)
            self.sides.append(side)

    @staticmethod
    def line_perpendicular(line1, line2, threshold=0.2):
        dx1 = line1.points[1][0] - line1.points[0][0]
        dy1 = line1.points[1][1] - line1.points[0][1]
        mag1 = (dx1 ** 2 + dy1 ** 2) ** 0.5

        dx2 = line2.points[1][0] - line2.points[0][0]
        dy2 = line2.points[1][1] - line2.points[0][1]
        mag2 = (dx2 ** 2 + dy2 ** 2) ** 0.5

        dot_product = dx1 * dx2 + dy1 * dy2
        cos_theta = dot_product / (mag1 * mag2)

        return abs(cos_theta) < threshold

    @staticmethod
    def line_parallel(line1, line2, threshold=0.2):
        dx1 = line1.points[1][0] - line1.points[0][0]
        dy1 = line1.points[1][1] - line1.points[0][1]
        slope1 = dy1 / dx1 if dx1 != 0 else 10.0

        dx2 = line2.points[1][0] - line2.points[0][0]
        dy2 = line2.points[1][1] - line2.points[0][1]
        slope2 = dy2 / dx2 if dx2 != 0 else 10.0

        if abs(slope1) > 10.0:
            slope1 = 10.0
        if abs(slope2) > 10.0:
            slope2 = 10.0

        # print(slope1, slope2)
        # Check if the absolute difference in slopes is within the threshold
        return abs(slope1 - slope2) <= threshold

    @staticmethod
    def line_intersection(line1, line2):
        x1, y1 = line1.points[0]
        x2, y2 = line1.points[1]
        x3, y3 = line2.points[0]
        x4, y4 = line2.points[1]

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if det == 0:
            return None

        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

        intersection_point = [intersection_x, intersection_y]
        return intersection_point

    @staticmethod
    def is_end_point(line, point):
        for point_i in line.points:
            dx = point_i[0] - point[0]
            dy = point_i[1] - point[1]
            if (dx ** 2 + dy ** 2) ** 0.5 < 1e-5:
                return True
        return False

    @staticmethod
    def point_on_line(line, point):
        start_point = line.points[0]
        end_point = line.points[1]
        segment_length_square = (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2

        dot_product = ((point[0] - start_point[0]) * (end_point[0] - start_point[0]) +
                       (point[1] - start_point[1]) * (end_point[1] - start_point[1]))

        if dot_product < 0 or dot_product > segment_length_square:
            return False

        dot_product = dot_product / segment_length_square

        distance_to_segment_square = (dot_product * (end_point[1] - start_point[1]) + start_point[1] - point[1]) ** 2 + \
                                     (dot_product * (end_point[0] - start_point[0]) + start_point[0] - point[0]) ** 2

        return distance_to_segment_square < 1e-2

    def point_in_polygon(self, point):
        """
        :return: Ture: include the point on the sides of polygon
        """
        n = len(self.vertices)
        inside = False

        for i in range(n):
            p1x, p1y = self.vertices[i]
            p2x, p2y = self.vertices[(i + 1) % n]

            if (point[0] == p1x and point[1] == p1y) or (point[0] == p2x and point[1] == p2y):
                return True
            if (min(p1x, p2x) <= point[0] <= max(p1x, p2x) and
                    min(p1y, p2y) <= point[1] <= max(p1y, p2y)):
                return True

        p1x, p1y = self.vertices[0]
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n]
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def line_segmentation(line, threshold=0.02):
        start_point = line.points[0]
        end_point = line.points[1]
        segment_length = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        num_segments = int(segment_length // threshold) + 1

        if num_segments == 1:
            return [start_point, end_point]

        segment_points = []
        for i in range(1, num_segments):
            ratio = i / num_segments
            x = start_point[0] + ratio * (end_point[0] - start_point[0])
            y = start_point[1] + ratio * (end_point[1] - start_point[1])
            segment_points.append((x, y))

        return [start_point] + segment_points + [end_point]

    def is_rectangle_inside_polygon(self, rectangle_vertices):
        rect_sides = []
        for index in range(len(rectangle_vertices) - 1):
            end_points = [rectangle_vertices[index], rectangle_vertices[index + 1]]
            side = Line(end_points)
            rect_sides.append(side)
        end_points = [rectangle_vertices[-1], rectangle_vertices[0]]
        side = Line(end_points)
        rect_sides.append(side)

        for side in rect_sides:
            side_segmentation = self.line_segmentation(side)
            for point in side_segmentation:
                if not self.point_in_polygon(point):
                    return False
        return True

    @staticmethod
    def select_farthest_points(points):
        all_pairs = []
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                distance_i_j = math.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)
                all_pairs.append((distance_i_j, (points[i], points[j])))
        all_pairs.sort(reverse=True)
        return all_pairs[0][1]

    @staticmethod
    def check_line_rectangular_side(line_1, line_2):
        start_point_1 = line_1.points[0]
        end_point_1 = line_1.points[1]
        start_point_2 = line_2.points[0]
        end_point_2 = line_2.points[1]

        if (abs(start_point_1[0] - start_point_2[0]) < 1e-5 and abs(end_point_1[0] - end_point_2[0]) < 1e-5) or \
                (abs(start_point_1[0] - end_point_2[0]) < 1e-5 and abs(end_point_1[0] - start_point_2[0]) < 1e-5) or \
                (abs(start_point_1[1] - start_point_2[1]) < 1e-5 and abs(end_point_1[1] - end_point_2[1]) < 1e-5) or \
                (abs(start_point_1[1] - end_point_2[1]) < 1e-5 and abs(end_point_1[1] - start_point_2[1]) < 1e-5):
            return True
        else:
            return False

    @staticmethod
    def convert_lines_to_squircle(line_1, line_2):
        start_point_1 = line_1.points[0]
        end_point_1 = line_1.points[1]
        start_point_2 = line_2.points[0]
        end_point_2 = line_2.points[1]
        if abs(start_point_1[0] - end_point_1[0]) < 1e-2:
            width = min(
                [math.sqrt((start_point_1[0] - start_point_2[0]) ** 2 + (start_point_1[1] - start_point_2[1]) ** 2),
                 math.sqrt((start_point_1[0] - end_point_2[0]) ** 2 + (start_point_1[1] - end_point_2[1]) ** 2),
                 math.sqrt((end_point_1[0] - start_point_2[0]) ** 2 + (end_point_1[1] - start_point_2[1]) ** 2),
                 math.sqrt((end_point_1[0] - end_point_2[0]) ** 2 + (end_point_1[1] - end_point_2[1]) ** 2)]
            )
            # print("width", width)
            height = math.sqrt((start_point_1[0] - end_point_1[0]) ** 2 + (start_point_1[1] - end_point_1[1]) ** 2)
            center = [(start_point_1[0] + start_point_2[0]) / 2, (start_point_1[1] + end_point_1[1]) / 2]
        else:
            width = math.sqrt((start_point_1[0] - end_point_1[0]) ** 2 + (start_point_1[1] - end_point_1[1]) ** 2)
            height = min(
                [math.sqrt((start_point_1[0] - start_point_2[0]) ** 2 + (start_point_1[1] - start_point_2[1]) ** 2),
                 math.sqrt((start_point_1[0] - end_point_2[0]) ** 2 + (start_point_1[1] - end_point_2[1]) ** 2),
                 math.sqrt((end_point_1[0] - start_point_2[0]) ** 2 + (end_point_1[1] - start_point_2[1]) ** 2),
                 math.sqrt((end_point_1[0] - end_point_2[0]) ** 2 + (end_point_1[1] - end_point_2[1]) ** 2)]
            )
            # print("width", width)
            center = [(start_point_1[0] + end_point_1[0]) / 2, (start_point_1[1] + start_point_2[1]) / 2]
        squircle = Squircle(type_='obstacle', center=center, width=width, height=height)
        return squircle

    @staticmethod
    def convert_lines_to_rect_vertices(line_1, line_2):
        rect_vertices = [line_1.points[0], line_1.points[1]]
        for point in line_2.points:
            if abs(point[0] - rect_vertices[-1][0]) < 1e-5 or abs(point[1] - rect_vertices[-1][1]) < 1e-5:
                rect_vertices.append(point)
        return rect_vertices

    def polygon_partition(self):
        all_squircles = []

        # extending lines
        for i, line_i in enumerate(self.sides):
            all_intersections = [line_i.points[0], line_i.points[1]]
            for j, line_j in enumerate(self.sides):
                if j == i:
                    continue
                else:
                    if not self.line_perpendicular(line_i, line_j):
                        continue
                    else:
                        intersection_point = self.line_intersection(line_i, line_j)
                        round_intersection = [np.round(intersection_point[0], self.round_num),
                                              np.round(intersection_point[1], self.round_num)]
                        intersection_point = round_intersection
                        if not self.point_on_line(line_j, intersection_point):
                            # print("not point on line", intersection_point, line_j.points)
                            continue
                        if self.is_end_point(line_i, intersection_point):
                            # print("is end point", intersection_point, line_i.points)
                            continue
                        # print("intersection_point", intersection_point)
                        all_intersections.append(intersection_point)

            if len(all_intersections) == 2:
                continue
            # print("new_line_points", new_line_points)
            new_line_points = self.select_farthest_points(all_intersections)
            old_line_points = line_i.points
            if abs(np.linalg.norm(np.array(new_line_points[0]) - np.array(new_line_points[1])) -
                   np.linalg.norm(np.array(old_line_points[0]) - np.array(new_line_points[1]))) > 1e-5:
                line_i.is_extended = True
            line_i.points = new_line_points

        for i, line_i in enumerate(self.sides):
            for j, line_j in enumerate(self.sides[i + 1:]):
                j = j + i + 1
                if i == j:
                    continue
                # print("line_j", line_j.points)
                if self.line_parallel(line_i, line_j):
                    # print("line_i", line_i.points)
                    # print("line_j", line_j.points)
                    if self.check_line_rectangular_side(line_i, line_j):
                        rect_vertices = self.convert_lines_to_rect_vertices(line_i, line_j)
                        if self.is_rectangle_inside_polygon(rect_vertices):
                            # print("line_j", line_j.points)
                            squircle = self.convert_lines_to_squircle(line_i, line_j)
                            all_squircles.append(squircle)
        for line in [self.sides[0], self.sides[-1]]:
            if not line.is_extended:
                print("====extended======")
                norm_vector = [0.0, 1.0]
                radius = 0.2
                line_j = Line([[line.points[0][0] + radius * norm_vector[0],
                                line.points[0][1] + radius * norm_vector[1]],
                               [line.points[1][0] + radius * norm_vector[0],
                                line.points[1][1] + radius * norm_vector[1]]])
                # print("line_j", line_j.points)
                squircle = self.convert_lines_to_squircle(line, line_j)
                all_squircles.append(squircle)

        return all_squircles
