import numpy as np
from typing import Union
import functools

# some constants
INSIDE = 0  # 0000
LEFT = 1  # 0001
RIGHT = 2  # 0010
BOTTOM = 4  # 0100
UP = 8  # 1000


class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def norm(self, p: int = 2) -> float:
        return (self.x**p + self.y**p) ** (1.0 / p)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y

    def __mul__(self, other: float) -> "Point":
        return Point(other * self.x, other * self.y)

    def __rmul__(self, other: float) -> "Point":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Point":
        return self.__mul__(1.0 / other)

    def isInside(self, other: Union["Line", "Rectangle", "Circle", "Ring"]) -> bool:
        if isinstance(other, Line):
            AM = Line(other.p1, self)
            MB = Line(self, other.p2)
            return np.close(np.abs(AM.dot(BM)), AM.length * MB.length)

        elif isinstance(other, Rectangle):
            # Based on https://stackoverflow.com/a/2763387
            AB = Line(other.c1, other.c2)
            AM = Line(other.c1, self)
            BC = Line(other.c2, other.c3)
            BM = Line(other.c2, self)

            return 0 <= AB.dot(AM) <= AB.dot(AB) and 0 <= BC.dot(BM) <= BC.dot(BC)

        elif isinstance(other, Circle):
            return self.distanceTo(other.m) <= other.r

        elif isinstance(other, Ring):
            return other.r_inner <= self.distanceTo(other.m) <= other.r_outer

        raise NotImplementedError

    def hasPassed(
        self,
        other: Union["Point", "Line", "Rectangle", "Circle", "Ring"],
        direction: "Point",
    ) -> bool:
        if isinstance(other, Point):
            p = other
        elif isinstance(other, Line):
            p = (other.p1 + other.p2) / 2.0
        elif isinstance(other, Rectangle):
            p = (other.c1 + other.c2 + other.c3 + other.c4) / 4.0
        elif isinstance(other, Circle):
            p = other.m
        elif isinstance(other, Ring):
            p = other.m
        else:
            raise NotImplementedError
        return direction.dot(p - self) <= 0

    def distanceTo(
        self, other: Union["Point", "Line", "Rectangle", "Circle", "Ring"]
    ) -> float:
        if isinstance(other, Point):
            return (self - other).norm(p=2)

        elif isinstance(other, Line):
            # Based on https://math.stackexchange.com/a/330329
            s2_minus_s1 = other.p2 - other.p1
            that = (self - other.p1).dot(s2_minus_s1) / s2_minus_s1.dot(s2_minus_s1)
            tstar = np.minimum(1, np.maximum(0, that))
            return (other.p1 + tstar * s2_minus_s1 - self).norm(p=2)

        elif isinstance(other, Rectangle):
            if self.isInside(other):
                return 0
            E = other.edges
            return np.min([self.distanceTo(e) for e in E])

        elif isinstance(other, Circle):
            return np.maximum(0, self.distanceTo(other.m) - other.r)

        elif isinstance(other, Ring):
            d = self.distanceTo(other.m)
            return np.max([r_inner - d, d - r_outer, 0])

        else:
            try:
                return other.distanceTo(
                    self
                )  # do we really need to try this? Does it ever succeed?
            except NameError:
                raise NotImplementedError
            print("Something went wrong!")
            raise


"""
Given three colinear points p, q, r, the function checks if 
point q lies on line segment 'pr' 
"""


def onSegment(p: Point, q: Point, r: Point) -> bool:
    return (
        q.x <= np.maximum(p.x, r.x)
        and q.x >= np.minimum(p.x, r.x)
        and q.y <= np.maximum(p.y, r.y)
        and q.y >= np.minimum(p.y, r.y)
    )


"""
To find orientation of ordered triplet (p, q, r). 
The function returns following values 
0 --> p, q and r are colinear 
1 --> Clockwise 
2 --> Counterclockwise 
"""


def orientation(p: Point, q: Point, r: Point) -> int:
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/ for details of below formula.
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return "Line(" + str(self.p1) + ", " + str(self.p2) + ")"

    # Line clips SELF using OTHER as a rectangular screen, then returns the clipped line
    def line_clip(self, other: Union["Rectangle"]):
        if not isinstance(other, Rectangle):
            raise NotImplementedError

        # clip the line if some part of it is outside the rectangle OTHER
        # using the Cohen Sutherland algorithm referened from:
        # https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
        new_p1 = self.p1
        new_p2 = self.p2

        y_max = other.corners[0].y
        y_min = other.corners[2].y
        x_max = other.corners[0].x
        x_min = other.corners[2].x

        p1_code = _calculate_line_clip_code(new_p1, other.corners)
        p2_code = _calculate_line_clip_code(new_p2, other.corners)
        while True:

            # if the entire line is inside the rectangle, just return itself
            if (p1_code | p2_code) == INSIDE:
                return Line(new_p1, new_p2)

            # if the entire line is outside the rectangle, just return itself
            if (p1_code & p2_code) != INSIDE:
                return None
                # return Line(new_p1, new_p2)

            # figure out which point is outside
            outside_point_code = p1_code if p1_code != INSIDE else p2_code

            # Now find the intersection point;
            # use formulas:
            #   slope = (y1 - y0) / (x1 - x0)
            #   x = x0 + (1 / slope) * (ym - y0), where ym is ymin or ymax
            #   y = y0 + slope * (xm - x0), where xm is xmin or xmax
            # No need to worry about divide-by-zero because, in each case, the
            # outcode bit being tested guarantees the denominator is non-zero

            x, y = (0, 0)

            if outside_point_code & UP:
                x = new_p1.x + (new_p2.x - new_p1.x) * (y_max - new_p1.y) / (
                    new_p2.y - new_p1.y
                )
                y = y_max
            elif outside_point_code & BOTTOM:
                x = new_p1.x + (new_p2.x - new_p1.x) * (y_min - new_p1.y) / (
                    new_p2.y - new_p1.y
                )
                y = y_min
            elif outside_point_code & RIGHT:
                y = new_p1.y + (new_p2.y - new_p1.y) * (x_max - new_p1.x) / (
                    new_p2.x - new_p1.x
                )
                x = x_max
            elif outside_point_code & LEFT:
                y = new_p1.y + (new_p2.y - new_p1.y) * (x_min - new_p1.x) / (
                    new_p2.x - new_p1.x
                )
                x = x_min

            # recalculate line clip codes
            if outside_point_code == p1_code:
                new_p1 = Point(x, y)
                p1_code = _calculate_line_clip_code(new_p1, other.corners)
            else:
                new_p2 = Point(x, y)
                p2_code = _calculate_line_clip_code(new_p2, other.corners)

        return Line(new_p1, new_p2)

    def intersectsWith(self, other: Union["Line", "Rectangle", "Circle", "Ring"]):
        if isinstance(other, Line):
            p1 = self.p1
            q1 = self.p2
            p2 = other.p1
            q2 = other.p2

            # Based on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
            # Find the four orientations needed for general and special cases
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special Cases
            # p1, q1 and p2 are colinear and p2 lies on segment p1q1
            if o1 == 0 and onSegment(p1, p2, q1):
                return True

            # p1, q1 and q2 are colinear and q2 lies on segment p1q1
            if o2 == 0 and onSegment(p1, q2, q1):
                return True

            # p2, q2 and p1 are colinear and p1 lies on segment p2q2
            if o3 == 0 and onSegment(p2, p1, q2):
                return True

            # p2, q2 and q1 are colinear and q1 lies on segment p2q2
            if o4 == 0 and onSegment(p2, q1, q2):
                return True

            return False  # Doesn't fall in any of the above cases

        elif isinstance(other, Rectangle):
            if self.p1.isInside(other) or self.p2.isInside(other):
                return True
            E = other.edges
            for edge in E:
                if self.intersectsWith(edge):
                    return True
            return False

        elif isinstance(other, Circle):
            return other.m.distanceTo(self) <= other.r

        elif isinstance(other, Ring):
            return (
                other.m.distanceTo(self.p1) >= other.r_inner
                or other.m.distanceTo(self.p2) >= other.r_inner
            ) and other.m.distanceTo(self) < other.r_outer

        raise NotImplementedError

    @property
    def length(self):
        return self.p1.distanceTo(self.p2)

    def dot(self, other: "Line") -> float:  # assumes Line is a vector from p1 to p2
        v1 = self.p2 - self.p1
        v2 = other.p2 - other.p1
        return v1.dot(v2)

    def hasPassed(
        self,
        other: Union["Point", "Line", "Rectangle", "Circle", "Ring"],
        direction: Point,
    ) -> bool:
        p = (self.p1 + self.p2) / 2.0
        return p.hasPassed(other, direction)

    def distanceTo(
        self, other: Union["Point", "Line", "Rectangle", "Circle", "Ring"]
    ) -> float:
        if isinstance(other, Point):
            return other.distanceTo(self)

        elif isinstance(other, Line):
            if self.intersectsWith(other):
                return 0.0
            return np.min(
                [
                    self.p1.distanceTo(other.p1),
                    self.p1.distanceTo(other.p2),
                    self.p2.distanceTo(other.p1),
                    self.p2.distanceTo(other.p2),
                ]
            )

        elif isinstance(other, Rectangle):
            if self.intersectsWith(other):
                return 0.0
            other_edges = other.edges
            return np.min([self.distanceTo(e) for e in other_edges])

        elif isinstance(other, Circle):
            return np.maximum(0, other.m.distanceTo(self) - other.r)

        elif isinstance(other, Ring):
            if self.intersectsWith(other):
                return 0.0
            p1m = self.p1.distanceTo(other.m)
            if p1m < other.r_inner:  # the line is inside the ring
                p2m = self.p2.distanceTo(other.m)
                return other.r_inner - np.maximum(p1m, p2m)
            else:  # the line is completely outside
                return np.maximum(0, other.m.distanceTo(self) - other.r_outer)

        raise NotImplementedError


class Rectangle:
    def __init__(
        self, c1: Point, c2: Point, c3: Point
    ):  # 3 points are enough to represent a rectangle
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c3 + c1 - c2

    def __str__(self):
        return (
            "Rectangle("
            + str(self.c1)
            + ", "
            + str(self.c2)
            + ", "
            + str(self.c3)
            + ", "
            + str(self.c4)
            + ")"
        )

    @property
    def area(self):
        area = 0
        j = len(self.corners) - 1
        for i in range(len(self.corners)):
            area += (self.corners[j].x + self.corners[i].x) * (
                self.corners[j].y - self.corners[i].y
            )
            j = i

        return abs(area / 2)

    @property
    def edges(self):
        e1 = Line(self.c1, self.c2)
        e2 = Line(self.c2, self.c3)
        e3 = Line(self.c3, self.c4)
        e4 = Line(self.c4, self.c1)
        return [e1, e2, e3, e4]

    @property
    def corners(self):
        return [self.c1, self.c2, self.c3, self.c4]

    # check what percent of SELF intersects with OTHER
    def intersectPercent(self, other: Union["Rectangle"]):
        # if it doesn't collide, then there is no intersection
        if not self.intersectsWith(other):
            return 0

        # line clip every edge
        poly_points = []
        for i, edge in enumerate(self.edges):
            if edge.intersectsWith(other):
                clipped_line = edge.line_clip(other)
                if clipped_line is not None:
                    poly_points.append(clipped_line.p1)
                    poly_points.append(clipped_line.p2)
            print()

        # Check if any of the parking spot corners are in the car.
        # Need to add if for example, only 1 line of the car actually intersects with the parking spot.
        # In that case, the above clip loop would only have 2 points to calculate area with, which is not enough
        for corner in other.corners:
            if corner.isInside(self):
                poly_points.append(corner)

        # sort the points in clockwise order
        # referenced @ciamej's answer from:
        # https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
        center_point = Point(
            np.mean([p.x for p in poly_points]), np.mean([p.y for p in poly_points])
        )
        sorter = PointSorter(center_point)
        poly_points = sorter.sortPoints(poly_points)

        # calculate the area of the polygon. Reference:
        # https://www.mathopenref.com/coordpolygonarea2.html
        area = 0
        j = len(poly_points) - 1
        for i in range(len(poly_points)):
            area += (poly_points[j].x + poly_points[i].x) * (
                poly_points[j].y - poly_points[i].y
            )
            j = i  # j is previous point to i

        area = abs(area / 2)

        percent = area / self.area

        return percent

    def intersectsWith(
        self, other: Union["Line", "Rectangle", "Circle", "Ring"]
    ) -> bool:
        if isinstance(other, Line):
            return other.intersectsWith(self)

        elif (
            isinstance(other, Rectangle)
            or isinstance(other, Circle)
            or isinstance(other, Ring)
        ):
            E = self.edges
            for e in E:
                if e.intersectsWith(other):
                    return True
            return False

        raise NotImplementedError

    def hasPassed(
        self,
        other: Union["Point", "Line", "Rectangle", "Circle", "Ring"],
        direction: Point,
    ) -> bool:
        p = (self.c1 + self.c2 + self.c3 + self.c4) / 4.0
        return p.hasPassed(other, direction)

    def distanceTo(
        self, other: Union["Point", "Line", "Rectangle", "Circle", "Ring"]
    ) -> float:
        if isinstance(other, Point) or isinstance(other, Line):
            return other.distanceTo(self)

        elif (
            isinstance(other, Rectangle)
            or isinstance(other, Circle)
            or isinstance(other, Ring)
        ):
            if self.intersectsWith(other):
                return 0.0
            E = self.edges
            return np.min([e.distanceTo(other) for e in E])

        raise NotImplementedError  # TODO: implement the other cases


class Circle:
    def __init__(self, m: Point, r: float):
        self.m = m
        self.r = r

    def __str__(self):
        return "Circle(" + str(self.m) + ", radius = " + str(self.r) + ")"

    def intersectsWith(self, other: Union["Line", "Rectangle", "Circle", "Ring"]):
        if isinstance(other, Line) or isinstance(other, Rectangle):
            return other.intersectsWith(self)

        elif isinstance(other, Circle):
            return self.m.distanceTo(other.m) <= self.r + other.r

        elif isinstance(other, Ring):
            return (
                other.r_inner - self.r
                <= self.m.distanceTo(other.m)
                <= self.r + other.r_outer
            )

        raise NotImplementedError

    def hasPassed(
        self,
        other: Union["Point", "Line", "Rectangle", "Circle", "Ring"],
        direction: Point,
    ) -> bool:
        return self.m.hasPassed(other, direction)

    def distanceTo(
        self, other: Union["Point", "Line", "Rectangle", "Circle", "Ring"]
    ) -> float:
        if (
            isinstance(other, Point)
            or isinstance(other, Line)
            or isinstance(other, Rectangle)
        ):
            return other.distanceTo(self)

        elif isinstance(other, Circle):
            return np.maximum(0, self.m.distanceTo(other.m) - self.r - other.r)

        elif isinstance(other, Ring):
            if self.intersectsWith(other):
                return 0.0
            d = self.m.distanceTo(other.m)
            return np.maximum(other.r_inner - d, d - other.r_outer) - self.r

        raise NotImplementedError


class Ring:
    def __init__(self, m: Point, r_inner: float, r_outer: float):
        self.m = m
        assert r_inner < r_outer
        self.r_inner = r_inner
        self.r_outer = r_outer

    def __str__(self):
        return (
            "Ring("
            + str(self.m)
            + ", inner radius = "
            + str(self.r_inner)
            + ", outer radius = "
            + str(self.r_outer)
            + ")"
        )

    def intersectsWith(self, other: Union["Line", "Rectangle", "Circle", "Ring"]):
        if (
            isinstance(other, Line)
            or isinstance(other, Rectangle)
            or isinstance(other, Circle)
        ):
            return other.intersectsWith(self)

        elif isinstance(other, Ring):
            d = self.m.distanceTo(other.m)
            if d > self.r_outer + other.r_outer:
                return False  # rings are far away
            if d + self.r_outer < other.r_inner:
                return False  # self is completely inside other
            if d + other.r_outer < self.r_inner:
                return False  # other is completely inside self
            return True

        raise NotImplementedError

    def hasPassed(
        self,
        other: Union["Point", "Line", "Rectangle", "Circle", "Ring"],
        direction: Point,
    ) -> bool:
        return self.m.hasPassed(other, direction)

    def distanceTo(
        self, other: Union["Point", "Line", "Rectangle", "Circle", "Ring"]
    ) -> float:
        if (
            isinstance(other, Point)
            or isinstance(other, Line)
            or isinstance(other, Rectangle)
            or isinstance(other, Circle)
        ):
            return other.distanceTo(self)

        if isinstance(other, Ring):
            if d > self.r_outer + other.r_outer:
                return d - self.r_outer - other.r_outer  # rings are far away
            if d + self.r_outer < other.r_inner:
                return (
                    other.r_inner - d - self.r_outer
                )  # self is completely inside other
            if d + other.r_outer < self.r_inner:
                return (
                    self.r_inner - d - other.r_outer
                )  # other is completely inside self
            return 0

        raise NotImplementedError  # TODO: implement the other cases


# helper function to get line clipping in entity Line
def _calculate_line_clip_code(point: Point, corners: [Point]):

    code = INSIDE

    if point.x < corners[2].x:
        code |= LEFT
    elif point.x > corners[0].x:
        code |= RIGHT
    if point.y > corners[0].y:
        code |= UP
    elif point.y < corners[2].y:
        code |= BOTTOM

    return code


# helper class to sort a series of Points given a center point
class PointSorter:
    def __init__(self, center: Point):
        self.center = center

    def sortPoints(self, points: [Point]):
        return sorted(points, key=functools.cmp_to_key(self._cmp_point))

    # sort the points in clockwise order
    # referenced @ciamej's answer from:
    # https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
    def _cmp_point(self, a, b):
        if a.x - self.center.x >= 0 and b.x - self.center.x < 0:
            return True
        if a.x - self.center.x < 0 and b.x - self.center.x >= 0:
            return False
        if a.x - self.center.x == 0 and b.x - self.center.x == 0:
            if a.y - self.center.y >= 0 or b.y - self.center.y >= 0:
                return a.y > b.y
            return b.y > a.y

        # compute the cross product of vectors (self.center -> a) x (self.center -> b)
        det = (a.x - self.center.x) * (b.y - self.center.y) - (b.x - self.center.x) * (
            a.y - self.center.y
        )
        if det < 0:
            return True
        if det > 0:
            return False

        # points a and b are on the same line from the self.center
        # check which point is closer to the self.center
        d1 = (a.x - self.center.x) * (a.x - self.center.x) + (a.y - self.center.y) * (
            a.y - self.center.y
        )
        d2 = (b.x - self.center.x) * (b.x - self.center.x) + (b.y - self.center.y) * (
            b.y - self.center.y
        )
        return d1 > d2
