import numpy as np
import math
import matplotlib.pyplot as plt


'''
TODO LIST:
- Fix for n > 3 inequalities - need to sort either points or inequalities so that they are in order (cw or ccw, doesn't matter)
- Finish for n = 1 inequality - finish graph_single_line function
- Finish n = 2 inequalities where one (or both) are horizontal or vertical - this will use the graph_single_line function

Completed:
- most n = 2 inequalities - as long as none are vertical or horizontal, graphing completely works
- n = 3 inequalities - graphing works for all cases tested so far
'''


# If Ax =/>=/<= b where A has 2 columns, graph the system of inequalities
# if A[0] == 0, then line is horizontal; if A[1] == 0, then line is vertical
_A = np.array([[1, 3], [1, 1], [0, 4]])
_b = np.array([[5], [3], [6]])

# temporary, set later to main.py's format and input
symbol = ">="

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
    elif shape[0] == 1:
        # only one line was given, call graph_single_line
        graph_single_line(_A, _b, inequality)
        return

    # get intersect points of inequalities/lines
    # TODO: sort points in clockwise motion
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


    # https://stackoverflow.com/questions/43971259/how-to-draw-polygons-with-python

    if len(points) <= 2:
        if len(points) == 2:
            # only occurs when all but one line is parallel
            # results from inconsistent system, should be caught above
            print("Error, two points found but system was not inconsistent??? This code is garbo.")
            exit()

        elif len(points) == 1:
            # occurs when only 2 lines are present
            if not shape[0] == 2:
                print("Error:", shape[0], "lines are present - logic went wrong... who wrote this code?!")
                exit()

            """ 
            METHOD:
            # polygon will be 3- to 5-sided, depending on slope of lines provided
            # depending if sign is >= or <=, pick higher/lower of one line's intercept with the preset x-max
            (x-max is absolute value of first point x-value * 5, same for y-max)
            then add x-max, y-max to points (or min for <=), then other line's intercept (check if intercept is beyond -x-max)
            exit out of if statement at this point, regular code will handle rest
            x-max, y-max is to set a bound on the projection
            """
            # inconsistent system can cause this, but should be caught above
            
            #TODO: insert if statement to throw error if symbol is = instead of >=/<=

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
            # then add intercepts and corners in order (either clockwise or counterclockwise)
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
                #print("intercept point found with x =", x_max, "is", r_point)

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


            # TODO: send all cases of vertical or horizontal lines to len(points) == 0 case, with line acting as new x/y max/min
            # replace later with actual format of inequalities
            if inequality == "<=":
                # find right point(s)
                l1_right_point = ((x_min, y_max))  # set to other extreme
                l2_right_point = ((x_min, y_max))
                l1_rightmost = False  # tracks if line1 or line2 is the rightmost

                '''
                Method: find the points that intersect x_max on each line and choose the lowest one
                If this point is below y_min, choose the point on that line that intersects with y_min instead and append to points

                If either line is vertical (does not intersect x_max and right_point is still [x_min, y_max]), then pick the point that
                intersects y_min on that line and append to points
                '''
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


                # ------------------------------------
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


            # replace symbol later
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

                
                # ------------------------------------
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
                

        else:
            # no intersect points - occurs when only 1 line is present
            # this clause should never be executed, since the 1 line case should not be solvable with np.linalg.solve
            # inconsistent system can cause this, but should be caught above

            print("Error: somehow no points were found. The person who wrote this reallllly messed up.")
            exit()


    points.append(points[0])  # add the first point again to create a closed loop

    xs, ys = zip(*points)

    plt.figure()
    plt.plot(xs, ys)
    plt.grid()
    plt.fill(xs, ys)
    plt.show()


def graph_single_line(A, b, inequality, custom_edge = False, value = 0, horizontal = False):
    # find x- and y- intercepts, then set outer bounds based on intercepts
    # if custom_edge is true, then the value given is set as one of the outer bounds
    #   - if horizontal is true, given value is a y-value; otherwise, x-value
    # A is a tuple, b is an int

    x_intercept = b / A[0]  # since Ax = b <=> a_0x + a_1y = b; x = b/a_0 - a_1 * 0
    y_intercept = b / A[1]

    # if custom edge is not given, base graph around y-intercept
    if inequality == "<=":
        # TODO: add custom edge check first, then do regular calculations after


        x = (b - A[1] * y_intercept) / A[0]
        x_max = x
        x_min = -x

        if x < 0:
            x_min = x
            x_max = -x

        

    # line is horizontal
    if A[0] == 0:
        pass
    # line is vertical
    elif A[1] == 0:
        pass

    pass


graph_feasible_region(_A, _b, "<=")