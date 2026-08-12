import sys, os
import matplotlib.pyplot as plt
import pickle

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
PATH2SIMDATADIR = "/Users/lukeserrano/repos/personal/basilisk/prjs/aero626/data"


runNum = 79
SAVE_FIGURES = True
REPORT_PLOTS = True

pth2data = os.path.join(PATH2SIMDATADIR, f"sim-{runNum:04d}")


from prjs.aero626.analysis.AnalysisTools import PoseAnalyzer


## EGMF
egmfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EGMF.pkl")

if os.path.exists(pklPath):

    # manually pull gyro data
    with open(pklPath, "rb") as f:
                    data = pickle.load(f)

    truth_gyroBias = data.get('truth_gyroBias', None)
    egmf_est_gyroBias = data.get('est_gyroBias', None)

    egmfPA._solutionName = "EGMF"
    egmfPA._NO_TITLE_FLAG = REPORT_PLOTS
    egmfPA._EXPORT_FIGURES_FLAG = SAVE_FIGURES
    egmfPA._dataDir = pth2data


    egmfPA.LoadFromPklFile(pkl_path=pklPath)

    # --- convert km → m for position ---
    egmfPA._est_r *= 1000
    egmfPA._truth_r *= 1000
    egmfPA._units_r = "m"

    # --- convert km/s → m/s for velocity ---
    egmfPA._est_v *= 1000
    egmfPA._truth_v *= 1000
    egmfPA._units_v = "m/s"

    # --- scale the covariance only once ---
    # position and velocity covariances both scale by (1000^2)
    egmfPA._est_Pxx[:, :6, :6] *= (1000**2)



    egmfPA.plotTransError3sigma()
    egmfPA.plotAttError3Sigma()

    # gyro bias plot 
    # rad to deg
    RAD2DEG2_PER_S2_TO_DEG2_PER_HR2 = (180/np.pi)**2 * (3600**2)
    est_Pxx =egmfPA._est_Pxx[:,9:,9:] * RAD2DEG2_PER_S2_TO_DEG2_PER_HR2
    truth_gyroBias = np.rad2deg(truth_gyroBias)*3600
    egmf_est_gyroBias = np.rad2deg(egmf_est_gyroBias)*3600

    egmfPA.plot3DimVecError3Sigma(
                                truth_v=truth_gyroBias,
                                truth_t=egmfPA._truth_t,
                                est_v=egmf_est_gyroBias,
                                est_Pxx=est_Pxx,
                                est_t=egmfPA._est_t,
                                strFigureTitle='EGMF Gryo Bias Error',
                                strXlabel='Time [s]',
                                strYunits='[Deg/Hr]'
            
    )






## EKF
ekfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EKF.pkl")

if os.path.exists(pklPath):
    # manually pull gyro data
    with open(pklPath, "rb") as f:
                data = pickle.load(f)

    truth_gyroBias = data.get('truth_gyroBias', None)
    ekf_est_gyroBias = data.get('est_gyroBias', None)

    ekfPA._solutionName = "EKF"
    ekfPA._NO_TITLE_FLAG = REPORT_PLOTS
    ekfPA._EXPORT_FIGURES_FLAG = SAVE_FIGURES
    ekfPA._dataDir = pth2data
    ekfPA.LoadFromPklFile(pkl_path=pklPath)

    # --- convert km → m for position ---
    ekfPA._est_r *= 1000
    ekfPA._truth_r *= 1000
    ekfPA._units_r = "m"

    # --- convert km/s → m/s for velocity ---
    ekfPA._est_v *= 1000
    ekfPA._truth_v *= 1000
    ekfPA._units_v = "m/s"

    # --- scale the covariance only once ---
    # position and velocity covariances both scale by (1000^2)
    ekfPA._est_Pxx[:, :6, :6] *= (1000**2)

    ekfPA.plotTransError3sigma()
    ekfPA.plotAttError3Sigma()


    # gyro bias plot 
    # rad to deg
    RAD2DEG2_PER_S2_TO_DEG2_PER_HR2 = (180/np.pi)**2 * (3600**2)
    est_Pxx =ekfPA._est_Pxx[:,9:,9:] * RAD2DEG2_PER_S2_TO_DEG2_PER_HR2

    truth_gyroBias = np.rad2deg(truth_gyroBias)*3600
    ekf_est_gyroBias = np.rad2deg(ekf_est_gyroBias)*3600

    ekfPA.plot3DimVecError3Sigma(
                                truth_v=truth_gyroBias,
                                truth_t=ekfPA._truth_t/60,
                                est_v=ekf_est_gyroBias,
                                est_Pxx=est_Pxx,
                                est_t=ekfPA._est_t/60,
                                strFigureTitle='EKF Gryo Bias Error',
                                strXlabel='Time [min]',
                                strYunits='Deg/Hr'
            
    )


    ## analyze ekf innovations
    # pull data

    inn_array = data.get('inn_array', None)
    inn_array *=1000
    inn_t = data.get('inn_t', None)
    inn_cov = data.get('inn_cov', None)
    inn_cov*=(1000**2)
    inn_lm_id = data.get('inn_lm_id', None)
    meas_noise_one_sigma_lvlh = data.get('meas_noise_one_sigma_lvlh',None) * 1000

    ekfPA.plot_landmark_innovations_lvlh(
        innArray = inn_array,
        innTime_array = inn_t,
        innCov = inn_cov,
        landmark_ids = inn_lm_id,
        est_r = ekfPA._est_r,
        est_v = ekfPA._est_v,
        est_q_array = ekfPA._est_q,
        lvlh_oneSigmaArrayInput = meas_noise_one_sigma_lvlh,
        xLabel="Time [s]",
        title="Landmark Innovations LVLH Frame",
        unitString = "m",
        show_measurement_noise=True,
        show_confidence=True,
        figsize=(8,5),
        _PLOT_BODY = False
        )












def writeGyroBiasStats(true_gryoBiasArray,est_gyroBiasArray,filePath,unitsString):
        # -----------------------
        #  Gyro Bias Errors
        # -----------------------
        e_r = true_gryoBiasArray - est_gyroBiasArray        # (N,3)
        rmse_r = np.sqrt(np.mean(np.sum(e_r**2, axis=1)))
        mae_r = np.mean(np.linalg.norm(e_r, axis=1))
        with open(filePath, "a") as f:
            f.write(f"Gyro Bias Error ({unitsString}):\n")
            f.write(f"   RMSE = {rmse_r:.6f}\n")
            f.write(f"   MAE  = {mae_r:.6f} \n\n")


## compute stats
# compute EGMF stats
pklPath = os.path.join(pth2data,"EGMF.pkl")
if os.path.exists(pklPath):
    egmfPA.extract_posterior(posteriori_time_array=inn_t)
    _ = egmfPA.compute_errors()
    egmfPA.write_stats_to_file()

    # gryo stats
    statsFile = os.path.join(egmfPA._dataDir,f"{egmfPA._solutionName}_stats.txt")
    est_ = egmf_est_gyroBias[egmfPA._logical_post_est,:]
    truth_ = truth_gyroBias[egmfPA._logical_post_truth,:]
    writeGyroBiasStats(true_gryoBiasArray=truth_,
                    est_gyroBiasArray = est_,
                    filePath = statsFile,
                    unitsString = 'deg/hr')



pklPath = os.path.join(pth2data,"EKF.pkl")
if os.path.exists(pklPath):
    # compute EKF stats
    ekfPA.extract_posterior(posteriori_time_array=inn_t)
    _ = ekfPA.compute_errors()
    ekfPA.write_stats_to_file()


    # gryo stats
    statsFile = os.path.join(ekfPA._dataDir,f"{ekfPA._solutionName}_stats.txt")
    est_ = ekf_est_gyroBias[ekfPA._logical_post_est,:]
    truth_ = truth_gyroBias[ekfPA._logical_post_truth,:]
    writeGyroBiasStats(true_gryoBiasArray=truth_,
                    est_gyroBiasArray = est_,
                    filePath = statsFile,
                    unitsString = 'deg/hr')




plt.show()


