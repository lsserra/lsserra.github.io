import numpy as np
import os, sys, copy
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import pickle


# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# EKF
from EkfPoseEstimator import EkfPosVelState, MekfState, EkfPoseEstimator, FullFilterState

# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

# simululation assistance
from faciliateSimulation import generateLandmarks, propagateMCMF, getLandmarkMeasurements
from prjs.aero626.analysis.PlottingAnalysisTools import (
    plot_landmark_innovations,
    plotPosVelStateErrorAnd3sigma,
    plotMekfAttitudeErrorAnd3Sigma,
    plot_landmark_innovations_lvlh
    )


from prjs.aero626.constants import (
    SIM_CONFIG_STRING,
    EKF_CONFIG_STRING,
    FILTER_OUTDIR_STRING,
    PATH2SIMDATADIR,
    SIM_DIR_PREFIX
)


def RunNavFromSimData(runDataDir, DECOUPLED_FLAG, TRUE_MEAS_FLAG, saveDataBool = True, EKF_ONLY_FLAG = False):
    # initalize random seed 
    random_seed = 42
    # random_seed = None
    rng = np.random.default_rng(random_seed)


    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav_TenthPeriod.pkl"
    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav_OnePeriod.pkl"
    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav.pkl"


    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav_OnePeriod_LowTipOff_NoController.pkl"
    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav_HighTipOff_With_Controller.pkl"
    
    
    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonGrav_TrueGyro_NoRate_IdentityAtt.pkl"
    
    
    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav.pkl"
    SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav_HighTip_Ctrlr_Mass.pkl"
    # SIM_PKL_FILE_STR = "data/MoonCentralBody_MoonEarthGrav_OnePeriod_LowTipOff_NoController_Mass.pkl"


    with open(SIM_PKL_FILE_STR, "rb") as f:
        sim_data = pickle.load(f)

    # write to the sim config file which sim pkl file was used
    sim_config_path = os.path.join(runDataDir,"simulation_config.txt")
    with open(sim_config_path, "w") as f:
        f.write(f"Basilisk Pkl File: {SIM_PKL_FILE_STR}\n\n")
        f.write(f"TRUE MEASUREMENTS FLAG: {TRUE_MEAS_FLAG}\n\n")
        f.write(f"DECOUPLED TRANS ATT FLAG: {DECOUPLED_FLAG}\n\n")


    timeData = sim_data["time"] * 1e-9 # to seconds
    sc_pos   = sim_data["sc_pos"] / 1000 # to km
    sc_vel   = sim_data["sc_vel"] / 1000
    moon_pos = sim_data["moon_pos"] / 1000
    moon_vel = sim_data["moon_vel"] / 1000
    gyro_meas = sim_data["gryoAngVel"] # rad/s
    gyro_time = sim_data["timeGyro"] * 1e-9 # to seconds
    q_BN_truth = sim_data["q_BN_truth"]
    print("Basilisk Simulation data successfully unboxed.")

    ## limit sim time for testing ##
    # t = 0.1*P = 650.9640763304291
    t_diff = timeData - 650.9640763304291
    t_diff = timeData - 200.
    idxCap = np.where(t_diff > 0.0)[0][0]
    # idxCap = 25
    # idxCap = 125
    # idxCap = 700
    # idxCap = 1500
    # idxCap = 4000
    # idxCap = None
    if idxCap is not None:
        timeData = timeData[:idxCap]
        sc_pos = sc_pos[:idxCap,:]
        sc_vel = sc_vel[:idxCap,:]
        moon_pos = moon_pos[:idxCap,:]
        moon_vel = moon_vel[:idxCap,:]
        gyro_meas = gyro_meas[:idxCap,:]
        gyro_time = gyro_time[:idxCap]
        q_BN_truth = q_BN_truth[:idxCap]
        print(f"Simulation data truncated to {idxCap} steps for testing.")

        ## report the time cutoff too!

    ## always make t0 =0
    t0 = timeData[0]
    timeData = timeData - t0
    # reset t0
    t0 = timeData[0]

    ## set measurement update at _ Hz ##
    measFreq = 1.0 # Hz
    simdt = timeData[1] - timeData[0]
    measDt = 1.0 / measFreq
    simIterPublishMeasBound = int(np.round(measDt / simdt))
    measCounter = 0

    ## measurement outage duration
    measOutDuration_idxArray = np.array(())







    with open("data/landmarks.pkl", "rb") as f:
        landmark_data = pickle.load(f)
    trueLandmarks = landmark_data["trueLandmarks"]
    mapLandmarks = landmark_data["mapLandmarks"]
    print("Landmark data successfully unboxed.")


    # DCM definition of MCMF frame
    earthObliquityToEcliptic = 23.44 # deg
    lunarEarthMoonIncl = 5.14 # deg
    lunarObliquityToEarthMoonPlane = 6.68 # deg

    angleJ200toMCMF = -earthObliquityToEcliptic-lunarEarthMoonIncl+lunarObliquityToEarthMoonPlane

    lunarObliquityToEcliptic = 1.54 # deg


    zhat = np.array([0.,0.,1.])
    xhat = np.array([1.,0.,0.])
    mhat3 = DCM.T2(np.deg2rad(angleJ200toMCMF)) @ zhat
    xhat1_orthog = xhat - np.dot(xhat,mhat3) * xhat
    rhat1_orthog = xhat1_orthog / np.linalg.norm(xhat1_orthog)
    mhat2 = np.cross(mhat3,rhat1_orthog)
    mhat1 = np.cross(mhat2,mhat3)
    T_MN = np.column_stack([mhat1,mhat2,mhat3])

    cosang = np.clip(np.dot(mhat3, zhat), -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cosang))


    q_MN_0 = Quaternion.from_DCM(T_MN)
    q_MN_0.ensureScalarPos()
    q_MN_0.normalize()



    # let's propagate truth MCMF frame attitude
    # w_RN_R = np.array([0.0, 0.0, 2*np.pi/29.530/24/3600])
    w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])



    # lets set up a for loop over sim time

    r_BM_M_TruthStoreList = []
    Mdrdt_BM_M_M_TruthStoreList = [] # time derivative of position B wrt M, as seen from M, coordinatized in M
    q_MN_store = [] # MCMF defintion
    q_MN_tkm =q_MN_0.as_array()

    # list for MCMF to body truth attitude
    q_BM_TruthStoreList = []
    SimTimeStore = []



    # list for storing true gyro bias
    gyroBiasTruthList = []

    # list for LVLH measurement noise
    lvlh_oneSigmaArrayInput_TruthStoreList = []
    # list for true TLB (Body to LVLH)
    TBodyToLVLH_TruthStoreList = []






    ## Initialize EKF ##
    # state sizes
    nx = 6
    nz = 3
    # ekf object
    ekf = EkfPoseEstimator()
    ekf._DECOUPLED_FLAG = DECOUPLED_FLAG
    # Process Noise
    psd_r = .0001 # 
    psd_Mdrdt = .001 # 
    Qww_posVel = np.diag([psd_r,psd_r,psd_r,psd_Mdrdt,psd_Mdrdt,psd_Mdrdt])


    ### intial conditions
    



    # covariance 
    sigma_r = 1. # km
    sigma_Mdrdt = .1 # km/s

    ## Initial Pos Vel state obj ##
    posVelState0 = EkfPosVelState(nx=nx)
    Pxx0 = np.diag([sigma_r,sigma_r,sigma_r,sigma_Mdrdt,sigma_Mdrdt,sigma_Mdrdt])
    Pxx0 = Pxx0 @ Pxx0.T
    posVelState0.Pxx = Pxx0
    posVelState0.t = t0

    # MCMF position and velocity
    r_BM_M0 = q_MN_0.rotate(sc_pos[0,:])
    rdot_BM_M = q_MN_0.rotate(sc_vel[0,:])
    Mdrdt_BM_M0 = rdot_BM_M - (np.cross(w_MN_M, r_BM_M0))

    # fill Gaussian corrupted inital states
    posVelState0.r_BM_M_mean = rng.normal(
            loc=r_BM_M0, scale=sigma_r, size=r_BM_M0.shape
        )
    posVelState0.Mdrdt_BM_M_mean = rng.normal(
            loc=Mdrdt_BM_M0, scale=sigma_Mdrdt, size=Mdrdt_BM_M0.shape
        )


    ## Initial MEKF state obj ##
    mekfState0 = MekfState(nx=nx)

    ## covariance
    sigmaAtt = np.array((np.deg2rad(.1),np.deg2rad(.1),np.deg2rad(.1))) # deg -> rad
    sigmaGyroBias = np.deg2rad(.2)/3600 # deg/hr -> rad/s
    Pxx0 = block_diag(np.diag(sigmaAtt),sigmaGyroBias*np.eye(3))
    Pxx0 = Pxx0@Pxx0.T

    ## init sc body attitude
    q_NM_0 = q_MN_0.inverse()
    q_BN_0 = q_BN_truth[0]
    q_BM_true_0 = q_BN_0*q_NM_0
    # Gaussian currupted attitude
    bodyErrorEulerVector= rng.normal(
            loc=np.zeros((3,1)), scale=np.deg2rad(.1), size=np.zeros((3,1)).shape
        )
    phi = np.linalg.norm(bodyErrorEulerVector)
    ehat = bodyErrorEulerVector/phi
    qbodyErrorEulerVector = Quaternion.from_axis_angle(axis=ehat.flatten(),angle=phi)
    mekfState0.q_BMref = q_BM_true_0 * qbodyErrorEulerVector
    


    mekfState0.Pxx = Pxx0
    # time
    mekfState0.t = t0

    # process noise
    sigma_gyroMeasNoise = np.sqrt(10)*10e-9 # rad/sec^(1/2)
    sigma_gyroBiasNoise = np.sqrt(10)*10e-12 # rad/sec^(3/2)
    Qmekf = block_diag(sigma_gyroMeasNoise * np.eye(3),sigma_gyroBiasNoise * np.eye(3)) # TODO check!

    # pass IC's, process noise PSD to filter
    ekf.initialize(
        initPosVelState=posVelState0,
        initMekfState=mekfState0,
        QPosVel=Qww_posVel,
        Qmekf=Qmekf) 



    ## landmark measurement initialization ##
    radiusMoonkm = 1737.4 # km
    normAltKm = np.linalg.norm(r_BM_M0) - radiusMoonkm
    
    # load map 
    ekf.loadLandmarkMap(trueLandmarks) 


    # parameters for 'optical sensor suite'
    # relativeDistanceThresholdKm = normAltKm + 100 # km
    # max number of landmark measurements
    MAX_NUM_LANDMARK_MEAS = 3

    # add some bias to gryo
    trueBias = np.deg2rad(np.array((0.1,0.1,0.1)))/3600 # deg/hr to rad/s

    if not EKF_ONLY_FLAG:
        ## EGMF Initialization ##
        from prjs.aero626.egmf.MoonEGMF import MoonEGMF, MoonGaussianMixtureModel
        egmf = MoonEGMF()
        ## spread means accross 3 sigma with identical variance
        Lx = 9
        sigmaSpread = 3.
        mx0 = copy.deepcopy(ekf.mx_full.mx)
        Pxx0 = copy.deepcopy(ekf.mx_full.Pxx)
        gmm = MoonGaussianMixtureModel(Lx_input=Lx)
        ws,ms, _ = gmm.gaussian_to_gmm(mx=mx0,
                            Pxx=Pxx0,
                            Lx=Lx,
                            spread_sigma=sigmaSpread)

        
        ## loop to create full state objs
        for i in range(len(ws)):
            statei = FullFilterState(nx=12)
            statei.mx = ms[i]
            statei.Pxx = Pxx0
            statei.w = ws[i]
            statei.t = t0
            ## trasfer initial attitude and gyro bias error to reference states
            # attitude
            statei.q_BMref = copy.deepcopy(ekf.mx_full.q_BMref)
            attErrEulerVec = statei.mx[6:9].flatten()
            phi = np.linalg.norm(attErrEulerVec)
            if phi > 1e-16:
                ehat = attErrEulerVec/phi
                qbodyErrorEulerVector = Quaternion.from_axis_angle(axis=ehat.flatten(),angle=phi)
                statei.q_BMref = statei.q_BMref * qbodyErrorEulerVector
            # gyro bias
            statei.gyroBiasRef = copy.deepcopy(ekf.mx_full.gyroBiasRef)
            errorBias = statei.mx[9:].flatten()
            statei.gyroBiasRef = (statei.gyroBiasRef.flatten() + errorBias).reshape(-1,1)
            # set mekf error states to zero
            statei.mx[6:] = np.zeros_like((statei.mx[6:]))

            egmf.gaussianPdfList_.append(copy.deepcopy(statei))

        
    

        ## analaze initial GMM 
        gmm._gaussianPdfList = copy.deepcopy(egmf.gaussianPdfList_)
        ## change weights
        refGaussian0 = FullFilterState(nx=12)
        refGaussian0.mx = mx0
        refGaussian0.Pxx = Pxx0
        gmm.pdf_based_weights(referenceGaussian=refGaussian0)
        egmf.gaussianPdfList_ = copy.deepcopy(gmm._gaussianPdfList)

        

        
        gmm._dataDir = runDataDir
        gmm._EXPORT_FIGURES_FLAG = True
        stateIdxAnalyze = [0,3,6,9]
        for i,stateIdx in enumerate(stateIdxAnalyze):
            gmm.visualizeGmm(stateIdxOfInterest=stateIdx, referenceGaussian=refGaussian0, numSigma=4, numPoints=800)

        ## initialize egmf
        egmf._ekf = EkfPoseEstimator()
        egmf._ekf._GMF_FLAG = True
        egmf._ekf._DECOUPLED_FLAG = DECOUPLED_FLAG
        egmf._ekf.loadLandmarkMap(trueLandmarks) 

        # additive process noise
        Pwwkm1 = np.zeros_like((Pxx0))
        Pwwkm1Trans = block_diag(1e-6*np.eye(3),1e-3*np.eye(3))
        Pwwkm1 = block_diag(Pwwkm1Trans,Qmekf)
        egmf.Pwwkm1 = Pwwkm1

        # t=0, compute some GM stats
        mean,cov = egmf.computeBestEstMeanAndCovAtEpoch()
        gyroBiasEst,q_BM_est= egmf.computeBestEstQuaternionAndGryoBias()
        GMstate = FullFilterState(nx=len(egmf._ekf.mx_full.mx))
        GMstate.mx = mean
        GMstate.Pxx = cov
        GMstate.t = t0
        GMstate.gyroBiasRef = gyroBiasEst
        GMstate.q_BMref = q_BM_est
        egmf.storeGmBestGuess_.append(copy.deepcopy(GMstate))
        



    N = len(timeData)
    update_every = max(1, N // 50)   # ~50 updates total

    # main sim loop
    for i, tk in enumerate(timeData):
        if i == 0:
            
            # MCMF and sc body attitude 
            q_MN_store.append(q_MN_0)
            q_NM_0 = q_MN_0.inverse()
            q_BN_0 = q_BN_truth[i]
            q_BM_true_array = (q_BN_0*q_NM_0).as_array()
            q_BM_TruthStoreList.append(q_BM_true_array)

            # initial r_BM_M and Mdrdt_BM_M
            r_BM_M0 = q_MN_0.rotate(sc_pos[0,:])
            rdot_BM_M = q_MN_0.rotate(sc_vel[0,:])
            Mdrdt_BM_M0 = rdot_BM_M - (np.cross(w_MN_M, r_BM_M0))
            
            r_BM_M_TruthStoreList.append(r_BM_M0)
            Mdrdt_BM_M_M_TruthStoreList.append(Mdrdt_BM_M0)
            gyroBiasTruthList.append(trueBias)

            SimTimeStore.append(tk)

            continue
        

        ## periodically update terminal
        if i % update_every == 0:
            print(f"Progress: {i}/{N}  ({100*i/N:.1f}%)")

        tkm = timeData[i-1]
        SimTimeStore.append(tk)

        ### MCMF TRUTH GENERATION ###
        # --- MCMF Coordinate Frame --- #
        q_MN_tk = propagateMCMF(
            tkm=tkm,
            tk=tk,
            q_MN_tkm=q_MN_tkm)
        
        q_MN_tk_obj = Quaternion.from_array(q_MN_tk)
        q_MN_store.append(q_MN_tk_obj)
        q_MN_tkm = q_MN_tk

        # --- position --- #
        r_BM_N = sc_pos[i,:]
        r_BM_M = q_MN_tk_obj.rotate(r_BM_N)
        r_BM_M_TruthStoreList.append(r_BM_M)

        # --- velocity --- #
        rdot_BM_M = q_MN_tk_obj.rotate(sc_vel[i,:])
        Mdrdt_BM_M = rdot_BM_M - (np.cross(w_MN_M, r_BM_M))
        Mdrdt_BM_M_M_TruthStoreList.append(Mdrdt_BM_M)

        # --- Attitude --- #
        q_NM_i = q_MN_tk_obj.inverse()
        q_BN_i = q_BN_truth[i]
        q_BM_true = q_BN_i * q_NM_i 
        q_BM_true.ensureScalarPos()
        q_BM_TruthStoreList.append(q_BM_true.as_array())


        



        ### EKF Progpagation ###
        # grab gyro meas
        w_BN_B = gyro_meas[i,:]
        w_BN_B += trueBias
        gyroBiasTruthList.append(trueBias)
        ekf.propagate(toTime=tk, w_BN_B_meas=w_BN_B)
        if not EKF_ONLY_FLAG:
            egmf.PropagateMixtureEkf(toTime=tk, w_BN_B_meas=w_BN_B)


        ### EKF Measurement Update ###
        measCounter += 1
        didUpdate = False
        if measCounter >= simIterPublishMeasBound:
        # if False:
            measCounter = 0
            # measurement noise = f(altitude)
            radiusMoonkm = 1737.4 # km
            normAltKm = np.linalg.norm(r_BM_M) - radiusMoonkm
            relativeDistanceThresholdKm = normAltKm + 100 # km

            lvlh_oneSigmaArrayInput =np.array((normAltKm/100,normAltKm/200,normAltKm/200))

            visibleLandmarks, PvvBodyFrame, T_BodyToLvlh_truth = getLandmarkMeasurements(
                r_BM_M_truth=r_BM_M,
                v_BM_M_truth=Mdrdt_BM_M,
                q_BM_truth=q_BM_true,
                distanceThresholdKm=relativeDistanceThresholdKm,  #  km
                trueLandmarks=trueLandmarks,
                lvlh_oneSigmaArray=lvlh_oneSigmaArrayInput,
                _TRUE_MEAS_FLAG = TRUE_MEAS_FLAG,
                maxNumMeasurements=MAX_NUM_LANDMARK_MEAS,
                randomSeed=random_seed,
                debugPlot = False
            )

            if visibleLandmarks.shape[0] > 0 and (i not in measOutDuration_idxArray):
            # if False:
                TBodyToLVLH_TruthStoreList.append(T_BodyToLvlh_truth)
                lvlh_oneSigmaArrayInput_TruthStoreList.append(lvlh_oneSigmaArrayInput)
                ekf.updateWithLandmarks(
                    z_meas_matrix = visibleLandmarks, 
                    PvvBodyFrame=PvvBodyFrame,
                    measTime=tk)
                if not EKF_ONLY_FLAG:
                    egmf.LandmarkMeasUpdateEkf(
                        z_meas_matrix = visibleLandmarks, 
                        PvvBodyFrame=PvvBodyFrame,
                        measTime=tk
                        )
                didUpdate = True
                stopTimeLimit = 1.0
                runningPlots = False
                if runningPlots and (tk>stopTimeLimit):
                    tSim = np.array(copy.deepcopy(SimTimeStore))
                    ## plot data to analyze
                    plotPosVelStateErrorAnd3sigma(r_TruthList=copy.deepcopy(r_BM_M_TruthStoreList),
                                                v_TruthList=copy.deepcopy(Mdrdt_BM_M_M_TruthStoreList),
                                                t_TruthNpArray=tSim,
                                                posVelEstListLog=copy.deepcopy(ekf.posVelState_log)
                                                )
                
                    plotMekfAttitudeErrorAnd3Sigma(mekfStateList=copy.deepcopy(ekf.mekfState_log),
                                                q_BM_TruthList=copy.deepcopy(q_BM_TruthStoreList),
                                                t_TruthNpArray=tSim,
                                                gryoBiasTruthList = gyroBiasTruthList)
        
                    plot_landmark_innovations(
                        copy.deepcopy(ekf.innovation_log),
                        xLabel="Time [s]",
                        title="EKF Landmark Innovations",
                    )
                    plot_landmark_innovations_lvlh(
                        innovations_log=copy.deepcopy(ekf.innovation_log),
                        T_BodyToLvlh_List=copy.deepcopy(TBodyToLVLH_TruthStoreList),
                        lvlh_oneSigmaArrayInput=copy.deepcopy(lvlh_oneSigmaArrayInput_TruthStoreList)
                    )
                    
                    plt.show()
                    
        if not didUpdate:
            # update solution timing 
            ekf.mx_mekf_post_tk = copy.deepcopy(ekf.mx_mekf_prior_tk)
            ekf.mx_posVel_post_tk = copy.deepcopy(ekf.mx_posVel_prior_tk)
            ekf.mx_full.t = tk

            # log data
            ekf.posVelState_log.append(copy.deepcopy(ekf.mx_posVel_post_tk))
            ekf.mekfState_log.append(copy.deepcopy(ekf.mx_mekf_post_tk))


        # manually get ready for next time
        ekf.mx_posVel_prior_tk_ = ekf.mx_posVel_post_tk
        ekf.mx_mekf_prior_tk_ = ekf.mx_mekf_post_tk






## optionally save data 
    SimTimeStore = np.array(SimTimeStore)




    ## package EGMF pose solution
    if saveDataBool:

        from prjs.aero626.analysis.AnalysisTools import PoseAnalyzer        

        ## EKF
        # create 'full state covariance'
        PxxEkfFull = []
        for i,mTrans in enumerate(ekf.posVelState_log):
            mMekf = ekf.mekfState_log[i]
            PxxTrans = copy.deepcopy(mTrans.Pxx)
            PxxMefk = copy.deepcopy(mMekf.Pxx)
            PxxEkfFull.append(block_diag(PxxTrans,PxxMefk))

        _poseAnalyzerEkf = PoseAnalyzer()
        _poseAnalyzerEkf._pklName = "EKF.pkl"
        pklPath = os.path.join(run_dir, _poseAnalyzerEkf._pklName)
        dataDict = _poseAnalyzerEkf.CreatePklFileDataDict(
            input_truth_r = np.array(r_BM_M_TruthStoreList),
            input_truth_v = np.array(Mdrdt_BM_M_M_TruthStoreList),
            input_truth_q = np.array(q_BM_TruthStoreList),
            input_truth_t = np.array(SimTimeStore),

            input_est_r = np.array([s.r_BM_M_mean.flatten() for s in ekf.posVelState_log]),
            input_est_v = np.array([s.Mdrdt_BM_M_mean.flatten() for s in ekf.posVelState_log]),
            input_est_q = np.array([s.q_BMref.as_array() for s in ekf.mekfState_log]),
            input_est_Pxx = np.array(PxxEkfFull),
            input_est_t = np.array([s.t for s in ekf.posVelState_log]),

            input_ref_frame = "MCMF",
            input_tgt_frame = "Body",
            input_resolved_frame = "MCMF"
        )
        ## lets append the gryo bias data dict to the pose one
        gryoBiasDict= {
            'truth_gyroBias': np.array(gyroBiasTruthList),
            'est_gyroBias': np.array([s.gyroBiasRef.flatten() for s in ekf.mekfState_log])
        }
        ## lets append the innovations
        innDict = {
            'inn_array': np.array([entry.innovation.flatten() for entry in ekf.innovation_log]),
            'inn_t': np.array([entry.t for entry in ekf.innovation_log]),
            'inn_cov': np.array([entry.innovationCov for entry in ekf.innovation_log]),
            'inn_lm_id': np.array([entry.landmarkId.flatten() for entry in ekf.innovation_log]),
            'meas_noise_one_sigma_lvlh': np.array((lvlh_oneSigmaArrayInput_TruthStoreList))
        }
        dataDict = dataDict | gryoBiasDict | innDict
        with open(pklPath, "wb") as f:
            pickle.dump(dataDict, f)




        ## EGMF
        if not EKF_ONLY_FLAG:
            _poseAnalyzer = PoseAnalyzer()
            _poseAnalyzer._pklName = "EGMF.pkl"
            pklPath = os.path.join(run_dir, _poseAnalyzer._pklName)
            dataDict = _poseAnalyzer.CreatePklFileDataDict(
                input_truth_r = np.array(r_BM_M_TruthStoreList),
                input_truth_v = np.array(Mdrdt_BM_M_M_TruthStoreList),
                input_truth_q = np.array(q_BM_TruthStoreList),
                input_truth_t = np.array(SimTimeStore),

                input_est_r = np.array([s.mx[0:3].flatten() for s in egmf.storeGmBestGuess_]),
                input_est_v = np.array([s.mx[3:6].flatten() for s in egmf.storeGmBestGuess_]),
                input_est_q = np.array([s.q_BMref.as_array() for s in egmf.storeGmBestGuess_]),
                input_est_Pxx = np.array([s.Pxx for s in egmf.storeGmBestGuess_]),
                input_est_t = np.array([s.t for s in egmf.storeGmBestGuess_]),

                input_ref_frame = "MCMF",
                input_tgt_frame = "Body",
                input_resolved_frame = "MCMF"
            )
            ## lets append the gryo bias data dict to the pose one
            gryoBiasDict= {
                'truth_gyroBias': np.array(gyroBiasTruthList),
                'est_gyroBias': np.array([s.gyroBiasRef.flatten() for s in egmf.storeGmBestGuess_])
            }
            dataDict = dataDict | gryoBiasDict
            with open(pklPath, "wb") as f:
                pickle.dump(dataDict, f)
    




## function to dynamically write new simulation data dirs
def get_next_sim_dir():
    os.makedirs(PATH2SIMDATADIR, exist_ok=True)

    # find all directory names like sim-xxxx
    existing = [
        d for d in os.listdir(PATH2SIMDATADIR)
        if d.startswith(SIM_DIR_PREFIX)
    ]
    
    if not existing:
        next_index = 1
    else:
        # extract numeric suffixes
        numbers = [int(d.split("-")[1]) for d in existing]
        next_index = max(numbers) + 1

    # format as sim-0001
    dirname = f"{SIM_DIR_PREFIX}{next_index:04d}"
    full_path = os.path.join(PATH2SIMDATADIR, dirname)
    
    return full_path


# main call 
if __name__ == "__main__":

    ## config
    saveDataBool = True
    _TRUE_MEAS_FLAG = False
    _DECOUPLED_TRANS_MEKF = False
    ekfOnlyFlag = False
    


    # 1. Create new run directory
    run_dir = get_next_sim_dir()
    
    if saveDataBool:
        os.makedirs(run_dir, exist_ok=True)
        # create figures dir
        figDir= os.path.join(run_dir,'figures')
        os.makedirs(figDir, exist_ok=True)

        # 3. Create empty config files (or write your config content)
        sim_cfg_path = os.path.join(run_dir, SIM_CONFIG_STRING)

        open(sim_cfg_path, "w").close()
        print(f"Created new simulation run directory:\n{run_dir}")


    # run main simulation function
    RunNavFromSimData(
        runDataDir= run_dir,
        TRUE_MEAS_FLAG=_TRUE_MEAS_FLAG,
        DECOUPLED_FLAG=_DECOUPLED_TRANS_MEKF,
        saveDataBool=saveDataBool,
        EKF_ONLY_FLAG=ekfOnlyFlag
    )
    



