# Document to plan for structure of panel interview presentation



## High level overview

* Title
* Agenda
* Personal Introduction (1 slide)

<!-- For the slides below I want a transition slide -->
* Prior Work Experience [\section] 
    - Sandia [blank transition slide]
        - content slide [\begin{frame}{}]
    - Firefly [blank transition slide]
        - content slide [\begin{frame}{}]
    - LASR [blank transition slide]
        - content slide [\begin{frame}{}]
        - Computer Vision's role in the lab

<!-- For this section we want to explicitly define the problem I am finding a solution to before moving into content. -->

<!-- When covering the mathematical details of the approaches to solve the stated problem,
Highlight the model DESIGN CHOICES and ASSUMPTIONS made. 
The notation in the slides for these DESIGN CHOICES and ASSUMPTIONS should be consistent. -->

<!-- I do not want words on the slides, but rather math and diagrams that I will talk through. -->

* Technical Projects
    - AERO 626 MEKF / EKF / EGMM Lunar Landar Project [\section]
        <!-- math is all located in projects/AERO626ProjectReport/main.tex -->
        - Problem Statement
        - choice of reference frame
        - full states to estimate, their dynamics
        - explaination of the landmark measurement / generation
        - how to estimate these states (EKF translational / MEKF attitude)
        <!-- I am going to give an outline but give suggestions as to what I should include in the slides -->
        - MEKF (report section IIIA1. Multiplicative Error Quaternion Formulation)
            - Error quaternion formulation
            - gyroscope model
            - state space dynamics
            - measurement model jacobians
        - Brief EKF translational state formulation
            - measurement model jacobians
        - Simulation environment and results...

    - Research 
        - Integration of ZED2i camera, target tracking relative position algorithm 
        - Relative Pose Estimation Problem
        - Relative Pose Estimation Assumptions and Geometry
        - Point-Cloud Registration Approach
        - Monocular Point-n-Perspective Approach

* Application of knowledge at Firefly [\section]

* Connection to this role [\section]