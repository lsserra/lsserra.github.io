import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# attitude helpers
from helpers.attitude import DCM
from helpers.attitude.Quaternion import Quaternion

class PoseAnalyzer():
    ## define pkl accessing keys
    PKL_TRUTH_POS_KEY          = "truth_r"
    PKL_TRUTH_VEL_KEY          = "truth_v"
    PKL_TRUTH_ATT_KEY          = "truth_q"
    PKL_TRUTH_TIME_KEY          = "truth_t"

    PKL_EST_POS_KEY          = "est_r"
    PKL_EST_VEL_KEY          = "est_v"
    PKL_EST_ATT_KEY          = "est_q"
    PKL_EST_PXX_KEY            = "est_Pxx"
    PKL_EST_TIME_KEY          = "est_t"
    
    PKL_REF_FRAME_KEY          = "ref_frame"
    PKL_TGT_FRAME_KEY          = "tgt_frame"
    PKL_RESOLVED_FRAME_KEY     = "resolved_frame"

    def __init__(self):
        self._truth_r = None
        self._truth_v = None
        self._truth_q = None
        self._truth_t = None
        self._logical_post_truth = None

        self._est_r = None
        self._est_v = None
        self._est_q = None
        self._est_Pxx = None
        self._est_t = None

        self._est_r_post = None
        self._est_v_post = None
        self._est_q_post = None
        self._est_Pxx_post = None
        self._est_t_post = None
        self._logical_post_est = None

        self._units_r = "km"
        self._units_v = "km/s"
        self._units_t = "s"
        self._units_att = "deg"

        self._ref_frame = ""
        self._tgt_frame = ""
        self._resolved_frame = ""

        self._dataDir = ""

        self._solutionName = ""

        ## optionally remove titles for clean figures
        self._NO_TITLE_FLAG = False

        self._EXPORT_FIGURES_FLAG = False

        

    
    def LoadFromPklFile(self,pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # ---- Truth ----
        self._truth_r = data.get(PoseAnalyzer.PKL_TRUTH_POS_KEY, None)
        self._truth_v = data.get(PoseAnalyzer.PKL_TRUTH_VEL_KEY, None)
        self._truth_q = data.get(PoseAnalyzer.PKL_TRUTH_ATT_KEY, None)
        self._truth_t = data.get(PoseAnalyzer.PKL_TRUTH_TIME_KEY, None)

        # ---- Estimation ----
        self._est_r = data.get(PoseAnalyzer.PKL_EST_POS_KEY, None)
        self._est_v = data.get(PoseAnalyzer.PKL_EST_VEL_KEY, None)
        self._est_q = data.get(PoseAnalyzer.PKL_EST_ATT_KEY, None)
        self._est_Pxx = data.get(PoseAnalyzer.PKL_EST_PXX_KEY, None)
        self._est_t = data.get(PoseAnalyzer.PKL_EST_TIME_KEY, None)

        # ---- Frames ----
        self._ref_frame      = data.get(PoseAnalyzer.PKL_REF_FRAME_KEY, "")
        self._tgt_frame      = data.get(PoseAnalyzer.PKL_TGT_FRAME_KEY, "")
        self._resolved_frame = data.get(PoseAnalyzer.PKL_RESOLVED_FRAME_KEY, "")
    
    def plotTransError3sigma(self):

        ############################################
        # Position and Velocity filter state
        # extraction, interpolation of truth, and error calculation
        ############################################
        t_filt = self._est_t
        P_diag_posVel = np.array([np.diag(P) for P in self._est_Pxx])
        sigma3_posVel = 3 * np.sqrt(P_diag_posVel)

        # Extract reference state
        r_filt = self._est_r
        v_filt = self._est_v

        # Extract true state
        r_truth = self._truth_r
        v_truth = self._truth_v

        r_interp_truth = interp1d(self._truth_t, r_truth, axis=0)
        r_true_interp = r_interp_truth(t_filt)

        v_interp_truth = interp1d(self._truth_t, v_truth, axis=0)
        v_true_interp = v_interp_truth(t_filt)


        # Compute estimation error
        positionError = r_true_interp - r_filt
        velocityError = v_true_interp - v_filt
        nSolutions = len(r_filt)



        ############################################
        # Plot position and velocity estimation errors
        ############################################
        fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        pos_labels = ['X', 'Y', 'Z']
        vel_labels = ['X', 'Y', 'Z']
        legend_label = 'Est. Error'

        # titles
        figureTitleOpt = f"{self._solutionName} {self._resolved_frame} Frame Translational Estimation Error"
        if self._NO_TITLE_FLAG:
            positionTitle = ''
            velocityTitle = ''
            figureTitle = ''
        else:
            positionTitle = (f"Position of {self._tgt_frame} w.r.t. {self._ref_frame} resolved in {self._resolved_frame} Estimation Error")
            velocityTitle = (f"Velocity of {self._tgt_frame} w.r.t. {self._ref_frame} resolved in {self._resolved_frame} Estimation Error")
            figureTitle = figureTitleOpt 
        
        # Position error plots
        for i in range(3):
            
            axs[i, 0].plot(t_filt, positionError[:, i], 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i, 0].plot(t_filt, sigma3_posVel[:, i], 'r--', linewidth=1)
            axs[i, 0].plot(t_filt, -sigma3_posVel[:, i], 'r--', linewidth=1, label='±3σ confidence')
            axs[i, 0].set_ylabel(f'{pos_labels[i]} [{self._units_r}]')
            axs[i, 0].grid(True)
            
        axs[0, 0].legend(loc='upper right')
        axs[0,0].set_title(positionTitle)

        # Velocity error plots
        for i in range(3):
            axs[i, 1].plot(t_filt, velocityError[:,i], 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i, 1].plot(t_filt, sigma3_posVel[:, 3 + i], 'r--', linewidth=1)
            axs[i, 1].plot(t_filt, -sigma3_posVel[:, 3 + i], 'r--', linewidth=1,label='±3σ confidence')
            axs[i, 1].set_ylabel(f'{vel_labels[i]} [{self._units_v}]')
            axs[i, 1].grid(True)
            
        axs[0, 1].legend(loc='upper right')
        axs[0,1].set_title(velocityTitle)

        axs[-1, 0].set_xlabel(f'Time [{self._units_t}]')
        axs[-1, 1].set_xlabel(f'Time [{self._units_t}]')
        fig.suptitle(figureTitle, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{figureTitleOpt.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")
        return
    
    def plotAttError3Sigma(self):

        ############################################
        # MEKF filter state
        # extraction, interpolation of truth, and error calculation
        ############################################
        t_filt = self._est_t
        P_diag_att = np.array([np.diag(P[6:,6:]) for P in self._est_Pxx])
        # convert to deg
        P_diag_att = np.rad2deg(np.rad2deg(P_diag_att))
        sigma3_mekf = 3 * np.sqrt(P_diag_att)


        # interpolate truth solution
        q_BM_truth_array = self._truth_q
        q_BM_interp1dObj_truth = interp1d(self._truth_t, q_BM_truth_array, axis=0)
        q_BM_true_interp = q_BM_interp1dObj_truth(t_filt)
        
        # compute attitude error as principle rotation vector
        # body attitude error list
        PRV_BprimeB_list = []
        for i in range(self._est_q.shape[0]):
            # compute attitude error and store
            q_BM_true = Quaternion.from_array(q_BM_true_interp[i,:]).normalize()
            q_BM_filt = Quaternion.from_array(self._est_q[i,:])
            prv_BprimeB = Quaternion.computeEulerVecAttErrorFromQuats(
                q_ref=q_BM_true,
                q_est=q_BM_filt
            )
            PRV_BprimeB_list.append(prv_BprimeB)

        PRV_BprimeB_array = np.array(PRV_BprimeB_list)


        nSolutions= len(self._est_q)

        ############################################
        # Plot attitude estimation errors
        ############################################
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        body_labels = ['X', 'Y', 'Z']
        legend_label = 'Est. Error'

        # titles
        figureTitleOpt = f"{self._solutionName} Frame {self._tgt_frame} w.r.t. {self._ref_frame} Estimation Error"
        if self._NO_TITLE_FLAG:
            attTitle = ''
            figureTitle = ''
        else:
            attTitle = f"{self._tgt_frame} frame attitude error as principle rotation vector"
            figureTitle = figureTitleOpt 
            
            
        
        # Position error plots
        for i in range(3):
            axs[i].plot(t_filt, np.rad2deg(PRV_BprimeB_array[:, i]), 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i].plot(t_filt, (sigma3_mekf[:, i]), 'r--', linewidth=1)
            axs[i].plot(t_filt, (-sigma3_mekf[:, i]), 'r--', linewidth=1, label='±3σ confidence')
            axs[i].set_ylabel(f'{body_labels[i]} [{self._units_att}]')
            axs[i].grid(True)
        axs[0].legend(loc='upper right')
        axs[0].set_title(attTitle)

       

        axs[-1].set_xlabel(f'Time [{self._units_t}]')
        fig.suptitle(figureTitle, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{figureTitleOpt.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")


    def plot3DimVecError3Sigma(self,truth_v, truth_t, est_v, est_Pxx, est_t,
                            strFigureTitle='',
                            strXlabel='',
                            strYunits=''
                               ):

        ############################################
        # MEKF filter state
        # extraction, interpolation of truth, and error calculation
        ############################################
        
        P_diag = np.array([np.diag(P) for P in est_Pxx])
        # convert to deg
        P_diag = (P_diag)
        sigma3 = 3 * np.sqrt(P_diag)


        # interpolate truth solution
        truth_inter_obj = interp1d(truth_t, truth_v, axis=0)
        truth_interp_to_sol = truth_inter_obj(est_t)
        
        error = truth_interp_to_sol - est_v

        ############################################
        # Plot attitude estimation errors
        ############################################
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axis_labels = ['X', 'Y', 'Z']
        legend_label = 'Est. Error'

        # titles
        if self._NO_TITLE_FLAG:
            figureTitle = ''
        else:
            figureTitle = strFigureTitle
            
            
        
        # Position error plots
        for i in range(3):
            axs[i].plot(est_t, error[:,i], 'k-', linewidth=1.8, label=f'{legend_label}')
            axs[i].plot(est_t, (sigma3[:, i]), 'r--', linewidth=1)
            axs[i].plot(est_t, (-sigma3[:, i]), 'r--', linewidth=1, label='±3σ confidence')
            axs[i].set_ylabel(f'{axis_labels[i]} [{strYunits}]')
            axs[i].grid(True)
        axs[0].legend(loc='upper right')

        axs[-1].set_xlabel(strXlabel)
        fig.suptitle(figureTitle, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{self._solutionName}_{strFigureTitle.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")


    ##  MAKE THIS AN INHERITED CLASS METHOD
    def plot_landmark_innovations_lvlh(self,
    innArray,
    innTime_array,
    innCov,
    landmark_ids,
    est_r,
    est_v,
    est_q_array,
    lvlh_oneSigmaArrayInput,
    xLabel="Time [s]",
    title="Landmark Innovations LVLH Frame",
    unitString = "km",
    show_measurement_noise=True,
    show_confidence=True,
    figsize=(8,5),
    _PLOT_BODY=False
    ):
        
        # rotate inovations and covariance into truth lvlh frame
        innLvlhArray = np.zeros_like(innArray)
        innCovLvlhArray = np.zeros_like(innCov)
        for i,tk in enumerate(innTime_array):

            # find all est_time <= innovation time
            valid_idxs = np.where(self._est_t <= tk)[0]

            # choose last = posteriori
            last_idx = valid_idxs[-1]

            est_r_k = est_r[last_idx]
            est_v_k = est_v[last_idx]
    
            T_BodyToLvlh_i = PoseAnalyzer.ComputeResolvedToLVLH_DCM(r_BM_M_truth=est_r_k.flatten(),
                                v_BM_M_truth=est_v_k.flatten(),
                                q_BM_truth=Quaternion.from_array(est_q_array[i,:].flatten()))
            
            if _PLOT_BODY:
                innLvlhArray[i,:] = innArray[i,:]
                innCovLvlhArray[i,:,:] = innCov[i,:,:]
                oneSigmaLVLH = lvlh_oneSigmaArrayInput[i,:]        # shape (3,)
                PvvLVLH = np.diag(oneSigmaLVLH**2)
                PvvBodyFrame = T_BodyToLvlh_i.T @ PvvLVLH @ T_BodyToLvlh_i   
                lvlh_oneSigmaArrayInput[i,:] = np.sqrt(np.diag(PvvBodyFrame))  

            else:
                innLvlhArray[i,:] = (T_BodyToLvlh_i @ innArray[i,:].T).reshape(1,3)
                innCovLvlhArray[i,:] = T_BodyToLvlh_i @ innCov[i,:,:] @ T_BodyToLvlh_i.T
            
        
        # extract q 1 sigma Pzz
        Pzz_diag = np.diagonal(innCovLvlhArray, axis1=1, axis2=2)

        innSigmaLvlh_array = np.sqrt(Pzz_diag)

        # --- Plot ---
        fig, axs = plt.subplots(4, 1, figsize=(figsize[0], figsize[1]+2), sharex=True)
        labels = [
        fr"$\hat{{r}}$ {unitString}",
        fr"$\hat{{v}}$ {unitString}",
        fr"$\hat{{h}}$ {unitString}",
        ]
        

        # titles
        if self._NO_TITLE_FLAG:
            figureTitle = ''
        else:
            figureTitle = title
            
            

        for i in range(3):
            axs[i].scatter(innTime_array, innLvlhArray[:, i], marker='x', color='k', label=f'Innovation')

            # Optional measurement noise bounds
            if show_measurement_noise:
                measNoise_times = np.unique(innTime_array)
                axs[i].plot(measNoise_times ,3 * lvlh_oneSigmaArrayInput[:, i], color='gray', linestyle='--', label='Measurement Noise ±3σ')
                axs[i].plot(measNoise_times ,-3 * lvlh_oneSigmaArrayInput[:, i], color='gray', linestyle='--')

            # Optional ±3σ filter confidence bounds
            if show_confidence:
                axs[i].plot(innTime_array, 3*innSigmaLvlh_array[:, i], '-r', label='Innovation ±3σ confidence')
                axs[i].plot(innTime_array, -3*innSigmaLvlh_array[:, i], '-r')

            axs[i].grid(True)
            axs[i].set_ylabel(f'{labels[i]}')
        
        axs[0].legend(loc='upper right')        
        axs[-1].set_xlabel(xLabel)

        # --- 4th row: Landmark ID vs time ---
        axs[3].scatter(innTime_array, landmark_ids, marker='o', s=12, color='b')
        axs[3].set_ylabel("Landmark ID")
        axs[3].grid(True)
        axs[3].set_xlabel(xLabel)

        fig.suptitle(figureTitle)
        plt.tight_layout()

        # opt save
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{self._solutionName} {title.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")



    def extract_posterior(self, posteriori_time_array):
       

        unique_post_times = np.unique(posteriori_time_array)
        

        # get posteriori indicies
        posterior_indices = []

        for tk in unique_post_times:
            # find all indices where est_t == tk
            idxs = np.where(self._est_t == tk)[0]
            # choose the last one
            posterior_indices.append(idxs[-1])

        #
        posterior_indices = np.array(posterior_indices)
        self._logical_post_est = posterior_indices

        self._est_r_post = self._est_r[self._logical_post_est,:]
        self._est_v_post = self._est_v[self._logical_post_est,:]
        self._est_q_post = self._est_q[self._logical_post_est,:]
        self._est_Pxx_post = self._est_Pxx[self._logical_post_est,:]
        self._est_t_post = unique_post_times


    def compute_errors(self):
        """
        Compute RMSE and MAE for r, v, and q using only posterior solutions.
        Stores results internally as self._stats dict.
        """

        if (self._truth_r is None or
            self._truth_v is None or
            self._truth_q is None or
            self._est_r_post is None or
            self._est_v_post is None or
            self._est_q_post is None):
            raise ValueError("Truth or posterior estimate arrays missing. Run extract_posterior() first.")

        # get posteriori indicies from truth
        posterior_indices = []

        for tk in self._est_t_post:
            # find all indices where est_t == tk
            idxs = np.where(self._truth_t == tk)[0]
            # choose the last one
            posterior_indices.append(idxs[-1])

        posterior_indices = np.array(posterior_indices)
        self._logical_post_truth = posterior_indices

    
        truth_r = self._truth_r[self._logical_post_truth,:]
        truth_v = self._truth_v[self._logical_post_truth,:]
        truth_q = self._truth_q[self._logical_post_truth,:]

        est_r = self._est_r_post
        est_v = self._est_v_post
        est_q = self._est_q_post

        # -----------------------
        #  Position Errors
        # -----------------------
        e_r = truth_r - est_r         # (N,3)
        rmse_r = np.sqrt(np.mean(np.sum(e_r**2, axis=1)))
        mae_r = np.mean(np.linalg.norm(e_r, axis=1))

        # -----------------------
        #  Velocity Errors
        # -----------------------
        e_v = truth_v - est_v         # (N,3)
        rmse_v = np.sqrt(np.mean(np.sum(e_v**2, axis=1)))
        mae_v = np.mean(np.linalg.norm(e_v, axis=1))

        # -----------------------
        #  Attitude Errors (deg)
        # -----------------------
        # compute attitude error as principle rotation vector
        # body attitude error list
        PRV_BprimeB_list = []
        for i in range(self._est_q_post.shape[0]):
            # compute attitude error and store
            q_BM_true = Quaternion.from_array(truth_q[i,:]).normalize()
            q_BM_filt = Quaternion.from_array(self._est_q_post[i,:])
            prv_BprimeB = Quaternion.computeEulerVecAttErrorFromQuats(
                q_ref=q_BM_true,
                q_est=q_BM_filt
            )
            PRV_BprimeB_list.append(prv_BprimeB)

        PRV_BprimeB_array = np.rad2deg(np.array(PRV_BprimeB_list))

        rmse_q = np.sqrt(np.mean(PRV_BprimeB_array**2))
        mae_q = np.mean(np.abs(PRV_BprimeB_array))

        N = len(self._est_t_post)

        # -----------------------
        # Store results internally
        # -----------------------
        self._stats = {
            "rmse_r": rmse_r,
            "mae_r": mae_r,
            "rmse_v": rmse_v,
            "mae_v": mae_v,
            "rmse_q_deg": rmse_q,
            "mae_q_deg": mae_q,
            "num_samples": N,
        }

        return self._stats


    def write_stats_to_file(self):
        """
        Writes the statistics stored in self._stats to a stats.txt file in the data directory.
        """
        if not hasattr(self, "_stats"):
            raise ValueError("No statistics available. Run compute_errors() first.")

        stats_path = os.path.join(self._dataDir, f"{self._solutionName}_stats.txt")

        with open(stats_path, "w") as f:
            f.write(f"Solution: {self._solutionName}\n")
            f.write(f"Data Directory: {self._dataDir}\n\n")
            f.write("Posteriori State Error Statistics\n")
            f.write("=================================\n\n")

            f.write(f"Number of samples: {self._stats['num_samples']}\n\n")

            f.write("Position Error (" + self._units_r + "):\n")
            f.write(f"   RMSE = {self._stats['rmse_r']:.6f}\n")
            f.write(f"   MAE  = {self._stats['mae_r']:.6f}\n\n")

            f.write("Velocity Error (" + self._units_v + "):\n")
            f.write(f"   RMSE = {self._stats['rmse_v']:.6f}\n")
            f.write(f"   MAE  = {self._stats['mae_v']:.6f}\n\n")

            f.write("Attitude Error (" + self._units_att + "):\n")
            f.write(f"   RMSE = {self._stats['rmse_q_deg']:.6f}\n")
            f.write(f"   MAE  = {self._stats['mae_q_deg']:.6f} \n\n")

        print(f"Statistics saved to: {stats_path}")

    
    @staticmethod
    def ComputeResolvedToLVLH_DCM(r_BM_M_truth,v_BM_M_truth,q_BM_truth):
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

        return T_body_to_lvlh_truth

    
    @staticmethod
    def CreatePklFileDataDict(
                    input_truth_r = None,
                    input_truth_v = None,
                    input_truth_q = None,
                    input_truth_t = None,

                    input_est_r = None,
                    input_est_v = None,
                    input_est_q = None,
                    input_est_Pxx = None,
                    input_est_t = None,

                    input_ref_frame = "",
                    input_tgt_frame = "",
                    input_resolved_frame = ""
                    ):
        "input all matrices as (N solutions x __)"
        data_dict = {
            # Truth
            PoseAnalyzer.PKL_TRUTH_POS_KEY:          input_truth_r,
            PoseAnalyzer.PKL_TRUTH_VEL_KEY:          input_truth_v,
            PoseAnalyzer.PKL_TRUTH_ATT_KEY:          input_truth_q,
            PoseAnalyzer.PKL_TRUTH_TIME_KEY:           input_truth_t,

            # Estimation
            PoseAnalyzer.PKL_EST_POS_KEY:            input_est_r,
            PoseAnalyzer.PKL_EST_VEL_KEY:            input_est_v,
            PoseAnalyzer.PKL_EST_ATT_KEY:            input_est_q,
            PoseAnalyzer.PKL_EST_PXX_KEY:           input_est_Pxx,
            PoseAnalyzer.PKL_EST_TIME_KEY:           input_est_t,

            # Frames
            PoseAnalyzer.PKL_REF_FRAME_KEY:          input_ref_frame,
            PoseAnalyzer.PKL_TGT_FRAME_KEY:          input_tgt_frame,
            PoseAnalyzer.PKL_RESOLVED_FRAME_KEY:     input_resolved_frame,

        }
        return data_dict
    
    
    
    

