import os, sys
import numpy as np

import copy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import pickle

# from EkfPoseEstimator import EkfErrorState, EkfReferenceState, EkfPoseEstimator


# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion




## create landmark map 
import numpy as np
def generateLandmarks(
    radiusBodyKm,
    mapPos1sigma,
    nPoints, 
    randomSeed=None
):
    if randomSeed is not None:
        np.random.seed(randomSeed)

    landmarks_N_km = []

    rng = np.random.default_rng()


    # random vectors
    v = rng.normal(loc=0, scale=1, size=(nPoints,3))
    v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)

    # true landmarks
    trueLandmarks = v_norm * radiusBodyKm

    # noisy map 
    mapLandmarks = rng.normal(loc=trueLandmarks, scale=mapPos1sigma, size=(nPoints,3))

    return trueLandmarks, mapLandmarks


### return landmarks in view with Gaussian noise
def getLandmarkMeasurements(
    r_BM_M_truth,
    v_BM_M_truth,
    q_BM_truth,
    distanceThresholdKm,
    trueLandmarks,
    lvlh_oneSigmaArray,
    maxNumMeasurements,
    _TRUE_MEAS_FLAG = False,
    randomSeed=None,
    debugPlot = False,
    firstPass =True
):
    """
    Generate landmark measurements within a given distance AND within a viewing cone.
    Cone Axis is assumed to be nadir pointing.

    Args:
        r_BM_M_truth (np.ndarray): True body position in M frame (3,)
        q_BM_truth (Quaternion): True quaternion body-to-M frame
        distanceThresholdKm (float): Max distance to consider [km]
        trueLandmarks (np.ndarray): Nx3 array of landmark positions [km]
        lvlh_oneSigmaArray (array): Std dev of measurement noise in LVLH frame [km]
        randomSeed (int, optional): RNG seed

    Returns:
        outputZkMat (np.ndarray): [n_visible x 4] matrix [x_B, y_B, z_B, landmarkID]
    """

    if randomSeed is not None:
        np.random.seed(randomSeed)
    rng = np.random.default_rng(randomSeed)

    ## filter candidiate landmarks by norm of relative position threshold
    # Compute relative position vectors in M frame
    r_LB_M = trueLandmarks - r_BM_M_truth
    distances = np.linalg.norm(r_LB_M, axis=1)
    # Filter by distance
    withinDistance = distances < distanceThresholdKm
    r_LB_M = r_LB_M[withinDistance]
    landmarkIndices = np.where(withinDistance)[0]
    if r_LB_M.shape[0] == 0:
        return np.empty((0, 4))
    # Rotate relative vectors into the body frame
    r_LB_B = np.array([q_BM_truth.rotate(vec) for vec in r_LB_M])
    r_LB_B_visible = r_LB_B
    visibleIndices = landmarkIndices

    ## Create LVLH frame and apply measurement noise in this frame
    # make LVLH frame
    rhat = r_BM_M_truth / np.linalg.norm(r_BM_M_truth)
    h = np.cross(r_BM_M_truth, v_BM_M_truth)
    zhat = h / np.linalg.norm(h)                   # orbital angular momentum dir
    rhat = rhat - zhat * np.dot(rhat, zhat)
    rhat /= np.linalg.norm(rhat)
    yhat = np.cross(zhat, rhat)
    yhat /= np.linalg.norm(yhat)
    
    # MCMF to LVLH
    TLM = np.concatenate((rhat.reshape(-1,1),yhat.reshape(-1,1),zhat.reshape(-1,1)),axis=1).T
    # MCMF to Body
    TBM = q_BM_truth.to_dcm()
    # Body to LVLH
    TLB = (TLM@TBM.T)
    T_body_to_lvlh_truth = TLB

    # rotate into lvlh
    r_LB_LVLH_visible = np.zeros_like(r_LB_B_visible)
    nVisibleMeas = r_LB_B_visible.shape[0]
    for i in range(nVisibleMeas):
        r_LB_LVLH_visible[i,:] = (TLB @ r_LB_B_visible[i,:].T).reshape(1,3)
    
    # add noise to each axis
    # oneSigmaLVLH = lvlh_oneSigmaArray
    # oneSigmaBody = TLB.T @ oneSigmaLVLH
    # PvvBodyFrame = np.diag(oneSigmaBody**2)

    oneSigmaLVLH = lvlh_oneSigmaArray        # shape (3,)
    PvvLVLH = np.diag(oneSigmaLVLH**2)       # 3×3 covariance in LVLH

    PvvBodyFrame = TLB.T @ PvvLVLH @ TLB

    # rhat
    if _TRUE_MEAS_FLAG:
        oneSigmaLVLH = np.zeros_like(oneSigmaLVLH)
    
    noisyMeasurementsLVLH_r = rng.normal(
        loc=r_LB_LVLH_visible[:,0].reshape(-1,1), scale=oneSigmaLVLH[0], size=(nVisibleMeas,1)
    )
    # vhat
    noisyMeasurementsLVLH_v = rng.normal(
        loc=r_LB_LVLH_visible[:,1].reshape(-1,1), scale=oneSigmaLVLH[1], size=(nVisibleMeas,1)
    )

    # hhat
    noisyMeasurementsLVLH_h = rng.normal(
        loc=r_LB_LVLH_visible[:,2].reshape(-1,1), scale=oneSigmaLVLH[2], size=(nVisibleMeas,1)
    )

    # construct measurement matrix
    noisyMeasurementsLVLH = np.column_stack((
        noisyMeasurementsLVLH_r,
        noisyMeasurementsLVLH_v,
        noisyMeasurementsLVLH_h
    ))
    # rotate back into body frame
    noisyMeasurementsBody = np.zeros_like(noisyMeasurementsLVLH)
    for i in range(nVisibleMeas):
        noisyMeasurementsBody[i,:] = (TLB.T @ noisyMeasurementsLVLH[i,:].T).reshape(1,3)


    ## Limit the number of landmarks available
    #  Append landmark indices
    if noisyMeasurementsBody.shape[0]>maxNumMeasurements:
        noisyMeasurementsBody = noisyMeasurementsBody[:maxNumMeasurements,:]
        visibleIndices = visibleIndices[:maxNumMeasurements]
        
    outputZkMat = np.hstack((noisyMeasurementsBody, visibleIndices.reshape(-1, 1)))


    ## plot the true landmarks as scattered plot 
    # overlay current postion
    if debugPlot is True:
        
        if firstPass is True:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            firstPass = False        
        # --- Plot ---

        # planet 
        # --- Planet sphere ---
        planetColor = '#555555'
        planetRadius = 1737.4 # km
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        xs = planetRadius * np.outer(np.cos(u), np.sin(v))
        ys = planetRadius * np.outer(np.sin(u), np.sin(v))
        zs = planetRadius * np.outer(np.ones_like(u), np.cos(v))
        # ax.plot_surface(xs, ys, zs, color=planetColor, alpha=0.3)

        # 'visible' landmarks
        r_LM_M_visible = trueLandmarks[visibleIndices,:]
        ax.scatter(trueLandmarks[:,0], trueLandmarks[:,1], trueLandmarks[:,2], 
                c='k', s=1, label='True Landmarks')
        
        ax.scatter(r_LM_M_visible[:,0], r_LM_M_visible[:,1], r_LM_M_visible[:,2], 
                c='y', s=8, label='Visable Landmarks')
        ax.scatter(r_BM_M_truth[0],r_BM_M_truth[1], r_BM_M_truth[2], 
                c='r', s=30, label='Truth Position')
        
        


        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_zlabel('z [km]')
        ax.set_title('Current Visible Landmarks on Lunar Surface')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.show()


    return outputZkMat, PvvBodyFrame, T_body_to_lvlh_truth


### MCMF propagation ###
w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])
def dqdt_wrapper(t, q):
    return Quaternion.dqdt(t, w_MN_M, q)

def propagateMCMF(tkm,tk,q_MN_tkm):
    # --- attitude --- #
    sol = solve_ivp(
        fun=dqdt_wrapper,
        t_span=[tkm, tk],
        y0=q_MN_tkm,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
    )

    q_MN_tk = sol.y[:, -1]

    # Normalize quaternion
    if q_MN_tk[-1] < 0:
        q_MN_tk = -q_MN_tk
    q_MN_tk /= np.linalg.norm(q_MN_tk)

    return q_MN_tk





if __name__ == "__main__":


    rng = np.random.seed(42)
    bodyRadius_km = 1737.4

    nLandmarks = 10000
    mapSigma = .01 #km
    
    trueLandmarks,mapLandmarks = generateLandmarks(
        randomSeed=rng,
        nPoints=nLandmarks,
        radiusBodyKm=bodyRadius_km,
        mapPos1sigma=mapSigma
    )

    # statistics
    errorMap = trueLandmarks-mapLandmarks
    std = np.std(errorMap,axis=0)

    # --- Plot ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # landmarks
    ax.scatter(trueLandmarks[:,0], trueLandmarks[:,1], trueLandmarks[:,2], 
            c='k', s=8, label='True Landmarks')
    # ax.scatter(mapLandmarks[:,0], mapLandmarks[:,1], mapLandmarks[:,2], 
    #         c='r', s=8, label='Map Landmarks')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    # ax.set_title('Generated Landmarks on Lunar Surface')
    
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()

    # # save to file
    # landmarkPicklePath = os.path.join('data', "landmarks.pkl")
    # with open(landmarkPicklePath, "wb") as f:
    #     pickle.dump({"trueLandmarks": trueLandmarks, "mapLandmarks": mapLandmarks}, f)

    # print(f"Saved trueLandmarks and mapLandmarks to {landmarkPicklePath}")