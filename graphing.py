import numpy as np
import math
import matplotlib.pyplot as plt


'''
Credit to drawing polygon method:
https://stackoverflow.com/questions/43971259/how-to-draw-polygons-with-python
Credit to sorting points method:
https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
'''


# If Ax =/>=/<= b where A has 2 columns, graph the system of inequalities
# if A[0] == 0, then line is horizontal; if A[1] == 0, then line is vertical
#_A = np.array([[1, 3], [0, 4], [1, 1]])
Array = np.array([[2, 3]])
#_b = np.array([[5], [6], [3]])
bee = np.array([[5]])

# temporary, set later to main.py's format and input
sign = ">="

def graph_feasible_region(_A, _b, inequality):
    A = np.array([[0, 0], [0, 0]])
    b = np.array([0, 0])

    shape = _A.shape

    points = []

    # for tracking if an inconsistent system of equations has been given
    equations = {}
    
    if shape[0] == 0:
        # how did this even happen
        print("The given array was empty. Or the programmer messed up.")
        exit()


    # graph_single_line is called when only one line is provided, or two lines are provided but
    # one is horizontal/vertical
    def graph_single_line(newA, newb, ineq, custom_edge = False, value = 0, horizontal = False):
        """
        Given A, b, and an inequality, graphs the region above/below the given line according to the
        given inequality. If custom_edge is true, one of the edges can be specified at the given
        value and whether it's horizontal or vertical.

        This method directly graphs the line. Return after calling this method.

        Invariant: A is a tuple and b is an int. newA and newb are used in order to avoid naming conflicts.

        Method: find x- and y- intercepts, then set outer bounds based on intercepts
        if custom_edge is true, then the value given is set as one of the outer bounds
            - if horizontal is true, given value is a y-value; otherwise it's an x-value
        """

        if ineq == "=":
            print("Error: given inequality in graph_single_line is '='. Blame the programmer.")
            exit()

        points = []  # clear points in case some were added previously

        # check if the given line will ever intercept x or y axes
        has_x_inter = newA[0] != 0
        has_y_inter = newA[1] != 0
        x_intercept = 0
        y_intercept = 0

        # calculate intercepts
        if has_x_inter:
            x_intercept = newb / newA[0]  # since Ax = b <=> a_0x + a_1y = b; x = b/a_0 - a_1 * 0
        if has_y_inter:
            y_intercept = newb / newA[1]

        # if custom edge is given, modify intercepts accordingly
        # TODO: fix calculation of intercepts
        if custom_edge:
            if has_x_inter and horizontal:
                x_intercept = value / newA[0]
            if has_y_inter and not horizontal:
                y_intercept = value / newA[1]


        # horizontal line given, draw a rectangle
        if not has_x_inter:
            # set boundaries
            y_top = y_intercept
            y_bottom = -y_intercept
            x_left = -y_intercept
            x_right = y_intercept

            if ineq == "<=":
                if y_intercept < 0:
                    y_bottom = y_intercept * 2
                    x_left = y_bottom
                elif y_intercept == 0:
                    y_bottom = -10
                    x_left = -10

                if custom_edge and not horizontal:
                    x_right = value
                    x_left = -value

                    if value < 0:
                        x_left = value * 2
                    elif value == 0:
                        x_left = -10

            else:  # inequality is >=
                y_top = y_intercept * 2
                y_bottom = y_intercept
                x_left = y_intercept
                x_right = y_intercept * 2

                if y_intercept < 0:
                    y_top = -y_intercept
                    x_right = y_top
                if y_intercept == 0:
                    y_bottom = 10
                    x_left = 10

                if custom_edge and not horizontal:
                    x_right = value * 2
                    x_left = value

                    if value < 0:
                        x_right = -value
                    elif value == 0:
                        x_right = 10
                

            # add points sequentially
            points.append([x_right, y_top])
            points.append([x_right, y_bottom])
            points.append([x_left, y_bottom])
            points.append([x_left, y_top])


        # vertical line given, draw a rectangle
        elif not has_y_inter:
            # set boundaries
            y_top = x_intercept
            y_bottom = -x_intercept
            x_left = -x_intercept
            x_right = x_intercept

            if ineq == "<=":
                if x_intercept < 0:
                    y_bottom = x_intercept * 2
                    x_left = y_bottom
                elif x_intercept == 0:
                    y_bottom = -10
                    x_left = -10

                if custom_edge and horizontal:
                    y_top = value
                    y_bottom = -value

                    if value < 0:
                        y_bottom = value * 2
                    elif value == 0:
                        y_bottom = -10

            else:  # inequality is >=
                y_top = x_intercept * 2
                y_bottom = x_intercept
                x_left = x_intercept
                x_right = x_intercept * 2

                if x_intercept < 0:
                    y_top = -x_intercept
                    x_right = y_top
                if x_intercept == 0:
                    y_bottom = 10
                    x_left = 10

                if custom_edge and horizontal:
                    y_top = value * 2
                    y_bottom = value

                    if value < 0:
                        y_top = -value
                    elif value == 0:
                        y_top = 10
                

            # add points sequentially
            points.append([x_right, y_top])
            points.append([x_right, y_bottom])
            points.append([x_left, y_bottom])
            points.append([x_left, y_top])


        # neither horizontal nor vertical, draw a triangle
        elif ineq == "<=":
            point1 = [x_intercept, 0]
            point2 = [0, y_intercept]
            point3 = [0, 0]  # changed later, if necessary

            if custom_edge:
                if horizontal:
                    point1 = [x_intercept, value]
                else:
                    point2 = [value, y_intercept]
                
                point3 = [x_intercept, y_intercept]
            else:
                if x_intercept < 0:
                    point3[0] = x_intercept
                if y_intercept < 0:
                    point3[1] = y_intercept

            points.append(point1)
            points.append(point2)
            points.append(point3)
            

        else:  # inequality is >=
            point1 = [x_intercept, 0]
            point2 = [0, y_intercept]
            point3 = [x_intercept, y_intercept]  # changed later, if necessary

            if custom_edge:
                if horizontal:
                    point1 = [x_intercept, value]
                    point3 = [0, value]
                else:
                    point2 = [value, y_intercept]
                    point3 = [value, 0]
            else:
                if x_intercept < 0:
                    point3[0] = 0
                if y_intercept < 0:
                    point3[1] = 0

            points.append(point1)
            points.append(point2)
            points.append(point3)


        points.append(points[0])  # add the first point again to create a closed loop

        xs, ys = zip(*points)

        plt.figure()
        plt.plot(xs, ys)
        plt.grid()
        plt.fill(xs, ys)
        plt.show()



    if shape[0] == 1:
        # only one line was given, call graph_single_line
        graph_single_line(_A[0], _b, inequality)
        return


    # get intersect points of inequalities/lines
    # points are sorted later, only if necessary
    for i in range(shape[0]):
        for j in range(shape[0]):
            if i < j:
                A[0, :] = _A[i, :]
                A[1, :] = _A[j, :]

                b[0] = _b[i, :]
                b[1] = _b[j, :]

                line1 = (_A[i, 0], _A[i, 1])
                line2 = (_A[j, 0], _A[j, 1])

                # check if equation already exists and gives different answer
                if equations.__contains__(line1) and not math.isclose(equations[line1], b[0]):
                    print("Error: inconsistent system of equations")  # make more specific later
                    exit()
                elif equations.__contains__(line2) and not math.isclose(equations[line2], b[1]):
                    print("Error: inconsistent system of equations")  # make more specific later
                    exit()

                equations[line1] = b[0]
                equations[line2] = b[1]

                point = np.linalg.solve(A, b)
                points.append(point)

                print("Starting point is", point)


    # if fewer than 3 points exist, need to add boundaries on edges (can't draw infinitely)
    if len(points) <= 2:

        # only occurs when all but one line is parallel
        # results from inconsistent system, should be caught above
        if len(points) == 2:
            print("Error, two points found but system was not inconsistent??? This code is garbo.")
            exit()

        # occurs when only 2 lines are present
        # inconsistent system can cause this, but should be caught above
        elif len(points) == 1:
            """ 
            METHOD:
            # polygon will be 3- to 5-sided, depending on slope of lines provided
            # depending if sign is >= or <=, pick higher/lower of one line's intercept with the preset x-max
            (x-max is absolute value of first point x-value * 5, same for y-max)
            then add x-max, y-max to points (or min for <=), then other line's intercept (check if intercept is beyond -x-max)
            exit out of if statement at this point, regular code will handle rest
            x-max, y-max is to set a bound on the projection
            """

            if not shape[0] == 2:
                print("Error:", shape[0], "lines are present - logic went wrong... who wrote this code?!")
                exit()

            # cannot draw region if inequalities given are equations
            if inequality == "=":
                print("Error: ")
                exit()

            point = points[0]

            # find x, y min and max
            x_range = abs(point[0] * 10)
            x_range = max(x_range, 10)  # set minimum range in case point is on x or y axis
            x_max = point[0] + x_range
            x_min = point[0] - x_range

            y_range = abs(point[1] * 5)
            y_range = max(y_range, 5)
            y_max = point[1] + y_range
            y_min = point[1] - y_range

            print("x_max is", x_max)
            print("x_min is", x_min)
            print("y_max is", y_max)
            print("y_min is", y_min)

            # get equations for the two lines
            line1 = (_A[0, 0], _A[0, 1])
            line2 = (_A[1, 0], _A[1, 1])

            # list for each line's intercepts with edge of screen (x/y max and min)
            l1points = []
            l2points = []

            
            # calculate intercepts of both lines with all edges
            # then add intercepts and corners in order, clockwise
            # both lines cannot be both horizontal or both vertical, otherwise point would not be found
            # NOTE: if any line is horizontal or vertical, call graph_single line with the horizontal/vertical line as an edge instead

            # calculate line1
            # line1 is horizontal
            if line1[0] == 0:
                graph_single_line(line2, _b[1], inequality, True, _b[0], True)
                return

            # line1 is vertical
            elif line1[1] == 0:
                graph_single_line(line2, _b[1], inequality, True, _b[0], False)
                return

            else:
                # find intercept with right edge
                A[0, :] = np.array([line1[0], line1[1]])
                A[1, :] = np.array([1, 0])  # create vertical line at x_max
                b[0] = _b[0, :]
                b[1] = np.array([x_max])
                r_point = np.linalg.solve(A, b)

                # find intercept with bottom edge
                A[0, :] = np.array([line1[0], line1[1]])
                A[1, :] = np.array([0, 1])  # create horizontal line at y_min
                b[0] = _b[0, :]
                b[1] = np.array([y_min])
                b_point = np.linalg.solve(A, b)

                # find intercept with left edge
                A[0, :] = np.array([line1[0], line1[1]])
                A[1, :] = np.array([1, 0])  # create vertical line at x_min
                b[0] = _b[0, :]
                b[1] = np.array([x_min])
                l_point = np.linalg.solve(A, b)

                # find intercept with top edge
                A[0, :] = np.array([line1[0], line1[1]])
                A[1, :] = np.array([0, 1])  # create horizontal line at y_max
                b[0] = _b[0, :]
                b[1] = np.array([y_max])
                t_point = np.linalg.solve(A, b)

                l1points.append(r_point)
                l1points.append(l_point)
                l1points.append(b_point)
                l1points.append(t_point)

                print("l1points are", l1points)

            # line 2
            # line2 is horizontal
            if line2[0] == 0:
                graph_single_line(line1, _b[0], inequality, True, _b[1], True)
                return

            # line2 is vertical
            elif line2[1] == 0:
                graph_single_line(line1, _b[0], inequality, True, _b[1], False)
                return

            else:
                # find intercept with right edge
                A[0, :] = np.array([line2[0], line2[1]])
                A[1, :] = np.array([1, 0])  # create vertical line at x_max
                b[0] = _b[1, :]
                b[1] = np.array([x_max])
                r_point = np.linalg.solve(A, b)

                # find intercept with bottom edge
                A[0, :] = np.array([line2[0], line2[1]])
                A[1, :] = np.array([0, 1])  # create horizontal line at y_min
                b[0] = _b[1, :]
                b[1] = np.array([y_min])
                b_point = np.linalg.solve(A, b)

                # find intercept with left edge
                A[0, :] = np.array([line2[0], line2[1]])
                A[1, :] = np.array([1, 0])  # create vertical line at x_min
                b[0] = _b[1, :]
                b[1] = np.array([x_min])
                l_point = np.linalg.solve(A, b)

                # find intercept with top edge
                A[0, :] = np.array([line2[0], line2[1]])
                A[1, :] = np.array([0, 1])  # create horizontal line at y_max
                b[0] = _b[1, :]
                b[1] = np.array([y_max])
                t_point = np.linalg.solve(A, b)

                l2points.append(r_point)
                l2points.append(l_point)
                l2points.append(b_point)
                l2points.append(t_point)

                print("l2points are", l2points)


            if inequality == "<=":
                '''
                Method: find the points that intersect x_max on each line and choose the lowest one
                If this point is below y_min, choose the point on that line that intersects with y_min instead and append to points

                All cases with vertical and horizontal lines have already been filtered out above and set to graph_single_line
                '''

                # find right point(s)
                l1_right_point = ((x_min, y_max))  # set to other extreme
                l2_right_point = ((x_min, y_max))
                l1_rightmost = False  # tracks if line1 or line2 is the rightmost

                # find the point on each line that intersects with the right edge
                for point1 in l1points:
                    if math.isclose(point1[0], x_max):
                        l1_right_point = point1
                        break  # there should only be at most one point on the right edge, otherwise I messed up bad

                for point2 in l2points:
                    if math.isclose(point2[0], x_max):
                        l2_right_point = point2
                        break

                # NOTE: vertical/horizontal line cases have been moved to len(points) == 0 case
                # find lower one of l1_right_point and l2_right_point, select as right point
                if l1_right_point[1] < l2_right_point[1]:
                    l1_rightmost = True
                    right_point = l1_right_point
                else:
                    right_point = l2_right_point

                if right_point[1] < y_min:  
                    # if point of intersect with right edge is below bottom edge, then pick point on line with bottom edge instead
                    if l1_rightmost:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_min):
                                right_point = point1
                    else:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_min):
                                right_point = point2
                elif right_point[1] > y_max:
                    # if point of intersect with right edge is above top edge, pick point on line w/ top edge instead
                    # add top right corner after
                    if l1_rightmost:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_max):
                                right_point = point1
                    else:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_max):
                                right_point = point2

                points.append(right_point)
                print("Added point", right_point, "on right")

                if math.isclose(right_point[1], y_max):
                    points.append((x_max, y_max))  # add top right corner to polygon if necessary
                    print("Added top right corner at", (x_max, y_max))

                if not math.isclose(right_point[1], y_min):
                    points.append((x_max, y_min))  # add bottom right corner to polygon if necessary
                    print("Added bottom right corner at", (x_max, y_min))


                # find left point(s)
                left_point = ((x_max, y_max))  # set to other extreme
                if l1_rightmost:
                    # find the point on l2 that is on x_min
                    for point2 in l2points:
                        if math.isclose(point2[0], x_min):
                            left_point = point2
                            break
                    
                    # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                    if left_point[1] > y_max:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_max):
                                left_point = point2
                    elif left_point[1] < y_min:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_min):
                                left_point = point2
                else:
                    # same logic as above, but for l1 instead
                    for point1 in l1points:
                        if math.isclose(point1[0], x_min):
                            left_point = point1
                            break

                    # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                    if left_point[1] > y_max:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_max):
                                left_point = point1
                    elif left_point[1] < y_min:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_min):
                                left_point = point1

                if not math.isclose(left_point[1], y_min):
                    points.append((x_min, y_min))  # add bottom left corner to polygon if necessary
                    print("Added bottom left corner at", (x_min, y_min))

                if math.isclose(left_point[1], y_max):
                    points.append((x_min, y_max))  # add top left corner to polygon if necessary
                    print("Added top left corner at", (x_min, y_max))

                points.append(left_point)
                print("Added point", left_point, "on left")


            elif inequality == ">=":
                '''
                Same method as above, only swap "lowest" and "highest"
                For more detailed comments, see above
                '''
                # find right point(s)
                l1_right_point = ((x_min, y_min))  # set to other extreme
                l2_right_point = ((x_min, y_min))
                l1_rightmost = False

                # find the point on each line that intersects with the right edge
                for point1 in l1points:
                    if math.isclose(point1[0], x_max):
                        l1_right_point = point1
                        break
                for point2 in l2points:
                    if math.isclose(point2[0], x_max):
                        l2_right_point = point2
                        break

                # find higher one of l1_right_point and l2_right_point, select as right point
                if l1_right_point[1] > l2_right_point[1]:
                    l1_rightmost = True
                    right_point = l1_right_point
                else:
                    right_point = l2_right_point

                if right_point[1] > y_max:
                    # if point of intersect with right edge is above top edge, pick point on line w/ top edge instead
                    if l1_rightmost:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_max):
                                right_point = point1
                    else:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_max):
                                right_point = point2
                elif right_point[1] < y_min:  
                    # if point of intersect with right edge is below bottom edge, then pick point on line with bottom edge instead
                    # add bottom right corner after
                    if l1_rightmost:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_min):
                                right_point = point1
                    else:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_min):
                                right_point = point2

                points.append(right_point)
                print("Added point", right_point, "on right")

                if math.isclose(right_point[1], y_min):
                    points.append((x_max, y_min))  # add bottom right corner to polygon if necessary
                    print("Added top right corner at", (x_max, y_min))

                if not math.isclose(right_point[1], y_max):
                    points.append((x_max, y_max))  # add top right corner to polygon if necessary
                    print("Added top right corner at", (x_max, y_max))

                
                # find left point(s)
                left_point = ((x_max, y_min))  # set to other extreme
                if l1_rightmost:
                    # find the point on l2 that is on x_min
                    # if this point is above y_max or below y_min, then pick the point on either that is highest in x
                    for point2 in l2points:
                        if math.isclose(point2[0], x_min):
                            left_point = point2
                            break

                    # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                    if left_point[1] > y_max:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_max):
                                left_point = point2
                    elif left_point[1] < y_min:
                        for point2 in l2points:
                            if math.isclose(point2[1], y_min):
                                left_point = point2
                else:
                    # same logic as above, but for l1 instead
                    for point1 in l1points:
                        if math.isclose(point1[0], x_min):
                            left_point = point1
                            break
                    
                    # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                    if left_point[1] > y_max:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_max):
                                left_point = point1
                    elif left_point[1] < y_min:
                        for point1 in l1points:
                            if math.isclose(point1[1], y_min):
                                left_point = point1

                if not math.isclose(left_point[1], y_max):
                    points.append((x_min, y_max))  # add top left corner to polygon if necessary
                    print("Added top left corner at", (x_min, y_max))

                if math.isclose(left_point[1], y_min):
                    points.append((x_min, y_min))  # add bottom left corner to polygon if necessary
                    print("Added bottom left corner at", (x_min, y_min))

                points.append(left_point)
                print("Added point", left_point, "on left")
                

            # code should not get to this point, since the filter was added before any calculations above
            else:
                print("Error: inequality is = symbol, even though it should have been filtered out already.")
                exit()

        
        # no intersect points - occurs when only 1 line is present
        # this clause should never be executed, since the 1 line case should not be solvable with np.linalg.solve
        # inconsistent system can cause this, but should be caught above
        else:
            print("Error: somehow no points were found. The person who wrote this reallllly messed up.")
            exit()


    # 3 or more points exist - don't need to add boundaries or other points, but need to sort points so they
    # are in a clockwise order for drawing properly
    # need to sort these points due to how pyplot takes input to draw polygons
    else:
        """
        Method: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
        """
        origin = points[0]  # set origin to the first point in the list
        refvec = [0, 1]  # reference vector for calculations

        def find_cw_angle_and_distance(point):
            # get vector between point and the origin
            v = [point[0] - origin[0], point[1] - origin[1]]
            # get length of vector
            v_len = math.hypot(v[0], v[1])

            # if the length is zero, there is no angle nor distance - return
            if v_len == 0:
                return -math.pi, 0

            # normalize the vector in order to find the directional angle
            norm = [v[0] / v_len, v[1]/v_len]
            dot_product = norm[0] * refvec[0] + norm[1] * refvec[1]
            diff_product = refvec[1] * norm[0] - refvec[0] * norm[1]

            angle = math.atan2(diff_product, dot_product)

            # convert negative angles to positive angles
            if angle < 0:
                return 2 * math.pi + angle, v_len
            return angle, v_len


        # use new function with sorted function to sort points list
        points = sorted(points, key = find_cw_angle_and_distance)


    points.append(points[0])  # add the first point again to create a closed loop

    xs, ys = zip(*points)

    plt.figure()
    plt.plot(xs, ys)
    plt.grid()
    plt.fill(xs, ys)
    plt.show()



graph_feasible_region(Array, bee, ">=")