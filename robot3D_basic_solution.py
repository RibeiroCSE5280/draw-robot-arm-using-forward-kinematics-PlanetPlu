#!/usr/bin/env python
# coding: utf-8

import numpy as np
from vedo import *

def RotationMatrix(theta, axis_name):
    """ calculate single rotation of $theta$ matrix around x,y or z
        code from: https://programming-surgeon.com/en/euler-angle-python-en/
    input
        theta = rotation angle(degrees)
        axis_name = 'x', 'y' or 'z'
    output
        3x3 rotation matrix
    """

    c = np.cos(theta * np.pi / 180)
    s = np.sin(theta * np.pi / 180)

    if axis_name == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]])
    if axis_name == 'y':
        rotation_matrix = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]])
    elif axis_name == 'z':
        rotation_matrix = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])
    else:
        rotation_matrix = np.eye(4)

    return rotation_matrix


def createCoordinateFrameMesh():
    """Returns the mesh representing a coordinate frame
    Args:
      No input args
    Returns:
      F: vedo.mesh object (arrows for axis)

    """
    _shaft_radius = 0.05
    _head_radius = 0.10
    _alpha = 1

    # x-axis as an arrow
    x_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(1, 0, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='red',
                        alpha=_alpha)

    # y-axis as an arrow
    y_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 1, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='green',
                        alpha=_alpha)

    # z-axis as an arrow
    z_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 0, 1),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='blue',
                        alpha=_alpha)

    originDot = Sphere(pos=[0, 0, 0],
                       c="black",
                       r=0.10)

    # Combine the axes together to form a frame as a single mesh object
    F = x_axisArrow + y_axisArrow + z_axisArrow + originDot

    return F


def getLocalFrameMatrix(R_ij, t_ij):
    """Returns the matrix representing the local frame
    Args:
      R_ij: rotation of Frame j w.r.t. Frame i
      t_ij: translation of Frame j w.r.t. Frame i
    Returns:
      T_ij: Matrix of Frame j w.r.t. Frame i.

    """
    # Rigid-body transformation [ R t ]
    T_ij = np.block([[R_ij, t_ij],
                     [np.zeros((1, 3)), 1]])

    return T_ij


def createArmPartMesh(L):
    # Create a sphere to show an example of a joint
    radius = 0.4
    sphere1 = Sphere(r=radius).pos(0, 0, 0).color("gray").alpha(.8)

    # Create coordinate frame mesh and transform
    Frame1Arrows = createCoordinateFrameMesh()

    # Making a cylinder, and add to local coordinate frame
    link1_mesh = Cylinder(r=0.4,
                          height=L,
                          pos=(L / 2 + radius, 0, 0),
                          c="white",
                          alpha=.8,
                          axis=(1, 0, 0)
                          )

    # Combine all parts into a single object
    # End effector has no cylinder attached
    Part = Frame1Arrows + link1_mesh + sphere1

    return Part


def forward_kinematics(Phi,	L1,	L2,	L3,	L4):
    """Calculate the local-to-global frame matrices,
    and the location of the end-effector.

    Args:
        Phi (4x1 nd.array):         Array containing the four joint angles
        L1, L2, L3, L4 (float):     lengths of the parts of the robot arm.

                                e.g., Phi = np.array([0, -10, 20, 0])

    Returns:

        T_01, T_02, T_03, T_04:     4x4	nd.arrays of local-to-global matrices
                                    for each frame.

        e:                          3x1 nd.array of 3-D coordinates, the
                                    location of the end-effector in space.
    """

    # Radius of sphere
    radius = 0.4
    arm_loc = np.array([[3], [2], [0.0]])  # Frame's origin (w.r.t. previous frame)

    # For T_01
    phi1 = Phi[0]   # Rotation angle of part 1 in degrees

    # Matrix of Frame 1 (written w.r.t. Frame 0, which is the previous frame)
    R_01 = RotationMatrix(phi1, axis_name='z')  # Rotation matrix
    t_01 = arm_loc                              # Frame's origin (w.r.t. previous frame)

    T_01 = getLocalFrameMatrix(R_01, t_01)      # Matrix of Frame 1 w.r.t. Frame 0

    # For T_02
    phi2 = Phi[1]   # Rotation angle of part 2 in degrees

    # Matrix of Frame 2 (written w.r.t. Frame 1, which is the previous frame)
    R_12 = RotationMatrix(phi2, axis_name='z')      # Rotation matrix
    t_12 = np.array([[L1+2*radius], [0.0], [0.0]])  # Frame's origin (w.r.t. previous frame)

    T_12 = getLocalFrameMatrix(R_12, t_12)      # Matrix of Frame 2 w.r.t. Frame 1

    # Matrix of Frame 2 w.r.t. Frame 0 (i.e., the world frame)
    T_02 = T_01 @ T_12


    # For T_03
    phi3 = Phi[2]   # Rotation angle of part 3 in degrees

    # Matrix of Frame 3 (written w.r.t. Frame 2, which is the previous frame)
    R_23 = RotationMatrix(phi3, axis_name='z')      # Rotation matrix
    t_23 = np.array([[L2+2*radius], [0.0], [0.0]])  # Frame's origin (w.r.t. previous frame)

    # Matrix of Frame 3 w.r.t. Frame 2
    T_23 = getLocalFrameMatrix(R_23, t_23)

    # Matrix of Frame 3 w.r.t. Frame 0 (i.e., the world frame)
    T_03 = T_01 @ T_12 @ T_23



    # For T_04
    phi4 = Phi[3]  # Rotation angle of part 3 in degrees

    # Matrix of Frame 4 (written w.r.t. Frame 3, which is the previous frame)
    R_34 = RotationMatrix(phi4, axis_name='z')      # Rotation matrix
    t_34 = np.array([[L3+radius], [0.0], [0.0]])  # Frame's origin (w.r.t. previous frame)

    # Matrix of Frame 4 w.r.t. Frame 3
    T_34 = getLocalFrameMatrix(R_34, t_34)

    # Matrix of Frame 4 w.r.t. Frame 0 (i.e., the world frame)
    T_04 = T_01 @ T_12 @ T_23 @ T_34

    # For e
    e = T_04[0:3,-1]    # Last column of last frame matrix

    return T_01, T_02, T_03, T_04, e


def get_frames(Phi, L1, L2, L3, L4):

    T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4)

    # Construct arm part
    Frame1 = createArmPartMesh(L1)
    # Transform the part to position it at its correct location and orientation
    Frame1.apply_transform(T_01)

    # Construct arm part
    Frame2 = createArmPartMesh(L2)
    # Transform the part to position it at its correct location and orientation
    Frame2.apply_transform(T_02)

    # Construct arm part
    Frame3 = createArmPartMesh(L3)
    # Transform the part to position it at its correct location and orientation
    Frame3.apply_transform(T_03)

    # Create the coordinate frame mesh and transform. This point is the end-effector. So, I am
    # just creating the coordinate frame.
    Frame4 = createCoordinateFrameMesh()
    # Transform the part to position it at its correct location and orientation
    Frame4.apply_transform(T_04)

    return Frame1, Frame2, Frame3, Frame4

def main():
    # Set the limits of the graph x, y, and z ranges
    axes = Axes(xrange=(0, 20), yrange=(-2, 10), zrange=(0, 6))

    # Lengths of the parts
    L1, L2, L3, L4 = [5, 8, 3, 0]
    #Phi = np.array([30, -50, -30, 0])

    #Frame1, Frame2, Frame3, Frame4 = get_frames(Phi, L1, L2, L3, L4)

    # Show everything
    # show([Frame1, Frame2, Frame3, Frame4], axes, viewup="z").close()

    # Showing Animation
    plt = Plotter(interactive=False)


    for i in range(0,30,1):
        Phi = np.array([i, 0, -i, 0])
        Frame1, Frame2, Frame3, Frame4 = get_frames(Phi, L1, L2, L3, L4)
        plt.clear()
        plt.show([Frame1, Frame2, Frame3, Frame4],axes, viewup="z")
        plt.render()

    for i in range(0,60,1):
        Phi = np.array([30, i, -30+2*i, 0])
        Frame1, Frame2, Frame3, Frame4 = get_frames(Phi, L1, L2, L3, L4)
        plt.clear()
        plt.show([Frame1, Frame2, Frame3, Frame4],axes, viewup="z")
        plt.render()

    for i in range(0,30,1):
        Phi = np.array([30-3*i, 60, 90, 0])
        Frame1, Frame2, Frame3, Frame4 = get_frames(Phi, L1, L2, L3, L4)
        plt.clear()
        plt.show([Frame1, Frame2, Frame3, Frame4],axes, viewup="z")
        plt.render()

    plt.interactive().close()

if __name__ == '__main__':
    main()



