### Doc for research thoughts and TODOs


## 8/15/26
* If I am using 2D keypoint ID, in real and ray-traced synthetic image, to establish correspondence... This 'inital correspondence problem' will hapepen each time we come in and out of eclipse.
    * black out background and illuminate the mock sat??


A way of obtaining a 'good initial estimate' of the relative pose is needed.
    - batch estmation ...


I am generating measurements with my CV algs.
Write a rel pos nav filter that uses CW EOMs.

* Model: 
    * CW EOMs translational states
    * attitude controller to match body rates! rotational states
        - angular velocity signal from back-differenced keypoints
        tracked across multiple frames






I want to restructure the document to be:

## 8/28/26
# Modeling
## 1 relative vector geometry, Camera and rigid body

## 2 relative vector geometry, Camera and rigid body, insert image plane

# Using the model to estimate parameters of interest
state = specific parameters of interest

## Defining the cost function
# define generic least squared error (LSE) cost function

# applying the LSE cost function to point cloud 
define measurement (refer to geometry) 
define measurement model (refer to geometry)
error = measurement - model(state)
# applying the LSE cost function to images
define measurement (refer to geometry)
define measurement model (refer to geometry)
error = measurement - model(state)

## The correspondence problem
# methods of how to establish correspondence, pcd
# methods of how to establish correspondence, monocular

## How to solve the non-linear least squares (NLLS) problem
Once we have selected a measurement, a measurement model, and established the correspondence set,
we must CHOOSE how to solve the NLLS problem
# method 1: Gauss-Newton
