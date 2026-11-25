
import sys, os
import numpy as np
from numpy import linalg
import copy
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

# Add the basilisk root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import numpy as np
from helpers.attitude.Quaternion import Quaternion



## Function to facilitate EGMF impl
def gaussian_pdf(x, mean, cov):
    """
    Evaluate a multivariate Gaussian pdf N(x; mean, cov).
    
    Args:
        x (scalar or np.ndarray): Evaluation point (scalar for 1D, array for nD).
        mean (scalar or np.ndarray): Mean value/vector.
        cov (scalar or np.ndarray): Covariance value/matrix.
    
    Returns:
        float: pdf value at x
    """
    
    # Convert scalar inputs to 1D numpy arrays if necessary, 
    # and reshape to column vectors (n, 1) for consistent matrix math.
    x = np.atleast_1d(x).reshape(-1, 1)
    mean = np.atleast_1d(mean).reshape(-1, 1)
    cov = np.atleast_1d(cov) # Keep cov as (n,) for scalar case or (n,n) for matrix

    # Determine dimension 'n' from mean shape
    n = mean.shape[0]

    # Handle the 1D covariance as a 1x1 matrix for linalg functions
    if n == 1 and cov.ndim == 1:
        cov = cov.reshape(1, 1)

    # Compute normalization constant
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * det_cov)

    # Compute exponent
    diff = x - mean
    # Use np.squeeze to turn the (1, 1) matrix result into a scalar for the exp function
    exponent = -0.5 * diff.T @ inv_cov @ diff
    
    return float(norm_const * np.exp(exponent))


############################################
#       Dynamics functions to be
#           called by solve_ivp
############################################

## reference quaternion dynamics wrapper
def dqdt_wrapper(t, q, w_BM_B):
    return Quaternion.dqdt(t, w_BM_B, q)



def posAttFullStateProp(t, x_aug, MU_MOON, w_MN_M, w_BM_B_corrected, Fw, Qww, nx):
    """
    Coupled propagation of mean and covariance for EKF in MCMF frame.
    x_aug = [x, P_flat] 
    where x = [r, rdot]

    Fw is the determinstic process noise mapping matrix and is not a function of the mean state
    Qww is the process noise PSD
    """

    # --- Unpack state ---
    x = x_aug[:nx]
    P_flat = x_aug[nx:]
    P = P_flat.reshape((nx, nx))

    # --- Unpack mean state ---
    # translational
    r = x[:3]
    rdot = x[3:6]
    rnorm = np.linalg.norm(r)
    # attitude
    alpha = x[6:9]
    erwbias = x[9:]

    # --- Mean dynamics ---
    fgrav = -MU_MOON * r / rnorm**3
    coriolis = 2 * np.cross(w_MN_M, rdot)
    centripetal = np.cross(w_MN_M, np.cross(w_MN_M, r))
    dr2dt2 = fgrav - coriolis - centripetal

    xdot_trans = np.concatenate((rdot.flatten(), dr2dt2.flatten()), axis=0)

    # --- Compute dynamics Jacobian Fx ---
    I3 = np.eye(3)
    w_skew = np.array([
        [0, -w_MN_M[2], w_MN_M[1]],
        [w_MN_M[2], 0, -w_MN_M[0]],
        [-w_MN_M[1], w_MN_M[0], 0]
    ])
    F11 = np.zeros((3, 3))
    F12 = np.eye(3)
    F22 = -2 * w_skew
    F21 = -MU_MOON * ((I3 / rnorm**3) - ((3 * r @ r.T) / rnorm**5))
    FxTrans = np.block([[F11, F12],
                   [F21, F22]])

    # propagate error covarance
    wx,wy,wz = w_BM_B_corrected
    omega_skew = np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
    
    # MEKF dynamics Jacobian
    FxMekf = np.hstack((-omega_skew,-np.eye(3)))
    FxMekf = np.vstack((FxMekf,np.zeros((3,6))))

    # Formulate full state Jacobian and derivatives
    Fx = np.zeros((nx,nx))
    Fx[:6,:6] = FxTrans
    Fx[6:,6:] = FxMekf

    xdot = np.zeros((nx,1)).flatten()
    xdot[:6] = xdot_trans



    # Defensive shape checks (will raise helpful errors if wrong)
    if Fw.ndim != 2:
        raise ValueError("Fw must be 2D; got shape {}".format(Fw.shape))
    if Qww.shape[0] != Qww.shape[1]:
        raise ValueError("Qww must be square")
    Qterm = Fw @ Qww @ Fw.T
    if Qterm.shape != (nx, nx):
        raise ValueError("Process noise term shape mismatch: expected ({},{}) got {}".format(nx, nx, Qterm.shape))

    # --- Covariance dynamics ---
    Pdot = Fx @ P + P @ Fx.T + Fw @ Qww @ Fw.T

    # --- Stack mean and covariance derivatives ---
    x_aug_dot = np.concatenate((xdot.flatten(), Pdot.flatten()),axis=0)

    return x_aug_dot


    


############################################
# Classes to hold Different Filter States
############################################
class FullFilterState():
    def __init__(self,nx):
        self.mx = np.zeros((nx, 1))
        self.Pxx = np.eye(nx)
        self.q_BMref = Quaternion.identity()
        self.gyroBiasRef = np.zeros((3,1))
        self.w = 0.0
        self.k = 0.0
        self.t = 0.0
    
class EkfPosVelState():
    def __init__(self,nx):
        self.r_BM_M_mean = np.zeros((3,1))
        self.Mdrdt_BM_M_mean = np.zeros((3,1))
        self.Pxx = np.eye(nx)

        self.t = 0.

class MekfState():
    def __init__(self,nx):
        # reference states
        self.q_BMref = Quaternion.identity()
        self.gyroBiasRef = np.zeros((3,1))
        
        # error states
        self.angleError_mean = np.zeros((3,1))
        self.gyroBiasError_mean = np.zeros((3,1))

        # error cov
        self.Pxx = np.eye(nx)

        self.t = 0.



class LandMarkInnovation():
    def __init__(self):
        self.t=0.0
        self.innovation = np.zeros((3,1))
        self.innovationCov = np.zeros((3,3))
        self.landmarkId = -1



############################################
#           Main Estimator class
############################################

class EkfPoseEstimator():
    def __init__(self):

        ## flag for EGMF assistance
        self._GMF_FLAG = False
        self._DECOUPLED_FLAG = False
        
        # define state sizes
        self.nx_full = 12
        self.nz = 3

        self.nx_posVel = 6
        self.nx_mekf = 6

        # full filter state
        self.mx_full = FullFilterState(self.nx_full) 
    
        # position and velocity state
        self.mx_posVel_prior_tk_ = EkfPosVelState(self.nx_posVel)
        self.mx_posVel_prior_tk = EkfPosVelState(self.nx_posVel)
        self.mx_posVel_post_tk = EkfPosVelState(self.nx_posVel)

        # MEKF state
        self.mx_mekf_prior_tk_ = MekfState(self.nx_mekf)
        self.mx_mekf_prior_tk = MekfState(self.nx_mekf)
        self.mx_mekf_post_tk = MekfState(self.nx_mekf)

        # Process Noise Shaping 
        self.Fw_posVel = np.eye(self.nx_posVel)
        self.Fw_mekf = block_diag(-np.eye(self.nx_mekf//2),np.eye(self.nx_mekf//2))

        # PSD
        self.QPosVel = np.zeros((self.nx_posVel,self.nx_posVel))
        self.QMekf = np.zeros((self.nx_mekf,self.nx_mekf))

        # --- Measurement Update Related --- #
        self.Hv = np.eye(self.nz)
        self.Pvv = np.eye(self.nz)
        self.landmarkMap = None



        


        # Moon angular rate
        self.w_MN_M = np.array([0.0, 0.0, 2*np.pi/27.322/24/3600])
        wx,wy,wz = self.w_MN_M
        self.w_MN_M_skew = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        # Moon gravitational const
        self.MU_MOON = 4902.799 # km^3/kg/s^2




        # --- Logging containers ---
        self.posVelState_log = []
        self.mekfState_log = []
        self.innovation_log = []       
        self.outlier_log = []



    def initialize(self,initPosVelState: "EkfPosVelState" ,initMekfState: "MekfState",
                   QPosVel, Qmekf):
        '''
            Funtion to initialize mean position / velocity state, MEKF state, and PSD
        '''
        # initalize prior
        self.mx_posVel_prior_tk_ = initPosVelState
        self.mx_mekf_prior_tk_ = initMekfState
        
        ## initialize full state
        # mean
        self.mx_full.mx[:6] = np.concatenate((
            initPosVelState.r_BM_M_mean,
            initPosVelState.Mdrdt_BM_M_mean),axis=0).reshape(-1,1)
        
        self.mx_full.q_BMref = initMekfState.q_BMref
        self.mx_full.gyroBiasRef = initMekfState.gyroBiasRef
        # covariance
        self.mx_full.Pxx = np.zeros((self.nx_full,self.nx_full))
        self.mx_full.Pxx[:6,:6] = initPosVelState.Pxx   
        self.mx_full.Pxx[6:,6:] = initMekfState.Pxx
        self.mx_full.t = initPosVelState.t

        # PSD
        self.QPosVel = QPosVel
        self.QMekf = Qmekf


    def loadLandmarkMap(self,landmarkMapMCMF):
        self.landmarkMap = landmarkMapMCMF


    def propagate(self,toTime,w_BN_B_meas):
        ''' 
            tk = toTime

            This function expects: 
                self.mx_prior_tk_ and self.xref_tk_
            to be set outside of this function           
            
            This function will set:
                self.xref_tk and self.mx_prior_tk 
        '''
        
        # set time 
        tk = toTime
        
        # log error state and reference state before propagation
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_prior_tk_))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_prior_tk_))

        # grab prior error covariance, reference state, and time

        ## If decoupled, diagonal full state covariance
        if self._DECOUPLED_FLAG:
            # Upper-right block
            self.mx_full.Pxx[0:self.nx_posVel, self.nx_posVel:self.nx_posVel+self.nx_mekf] = 0.0

            # Lower-left block
            self.mx_full.Pxx[self.nx_posVel:self.nx_posVel+self.nx_mekf, 0:self.nx_posVel] = 0.0


        # --- Full State Propagation --- #
        tkm = self.mx_full.t
        Pxx_prior_tk_ = self.mx_full.Pxx

        ## side step odd ekf state handling
        if self._GMF_FLAG:
            xRef_tk_ = self.mx_full.mx
            gyroBiasRef = self.mx_full.gyroBiasRef
            q_BMref_tk_ = self.mx_full.q_BMref
        else:
            xRef_tk_ = np.concatenate((
                self.mx_posVel_prior_tk_.r_BM_M_mean.flatten(),
                self.mx_posVel_prior_tk_.Mdrdt_BM_M_mean.flatten(),
                np.zeros((self.nx_mekf,1)).flatten()
                ),axis=0)
            gyroBiasRef = self.mx_mekf_prior_tk_.gyroBiasRef
            q_BMref_tk_ = self.mx_mekf_prior_tk_.q_BMref

        x_aug0 = np.concatenate((
        xRef_tk_.flatten(),                
        Pxx_prior_tk_.flatten()  
        ),axis=0)

        # compute corrected gyro output 
        # collect gyro measurement and correct with current bias est
        w_BN_B_corrected = (w_BN_B_meas - gyroBiasRef.flatten()).flatten()
        # rotate MCMF angular velocity into body frame
        w_MN_B = q_BMref_tk_.rotate(self.w_MN_M)
        w_BM_B_corrected = w_BN_B_corrected - w_MN_B

        # propagate the coupled mean and covariance dynamics
        # construct full state noise mapping and psd matrices
        Qfull = np.zeros((self.nx_full,self.nx_full))
        Fwwfull = np.zeros((self.nx_full,self.nx_full))

        ## manually add process noise in EGMF case, 'discritized dynamics'
        if not self._GMF_FLAG:
            Qfull[:6,:6] = self.QPosVel
            Qfull[6:,6:] = self.QMekf
            Fwwfull[:6,:6] = self.Fw_posVel
            Fwwfull[6:,6:] = self.Fw_mekf

        sol = solve_ivp(
        fun=lambda t, x_aug: posAttFullStateProp(
            t,
            x_aug, 
            self.MU_MOON, 
            self.w_MN_M, 
            w_BM_B_corrected,
            Fwwfull, 
            Qfull, 
            self.nx_full
            ),
        t_span=[tkm, tk],
        y0=x_aug0,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
        )

        # reconstruct reference state and error covariance
        x_aug_sol = sol.y 
        x_sol_tk = x_aug_sol[:self.nx_full, -1]
        Pxx_sol_tk = x_aug_sol[self.nx_full:, -1].reshape(self.nx_full,self.nx_full)


        # if any non-finite values, thrown an exception
        if np.any(Pxx_sol_tk[np.diag_indices_from(Pxx_sol_tk)] < 0.0):
            raise ValueError(
                f"Invalid covariance: Pxx has negative diagonal entries after translational propagation at time {tk}"
            )
            

        # --- MEKF Quaternion Propagation --- #
        # ensure time is aligned with pos/vel
        if self._GMF_FLAG:
            tkm_mekf = self.mx_full.t
        else:
            tkm_mekf = self.mx_mekf_prior_tk_.t
        if not np.isclose(tkm,tkm_mekf,1e-8):
            raise ValueError("Propagation tk minus for pos/vel and MEKF states do not align.")

        ## --- propagate quaternion --- ##
        
        q_BM_tk_ = q_BMref_tk_.as_array()

        sol = solve_ivp(
            fun=lambda t,x : dqdt_wrapper(t,x,w_BM_B=w_BM_B_corrected),
            t_span=[tkm, tk],
            y0=q_BM_tk_,
            method='RK45',
            rtol=1e-9,
            atol=1e-9
        )

        q_BM_tk = sol.y[:, -1]
        q_BM_tk_obj = Quaternion.from_array(q_BM_tk)
        if np.abs(1. - q_BM_tk_obj.norm()) > 1e-7:
            q_BM_tk_obj.normalize()


        # update pos vel state obj post propagation
        self.unpackFullState(
            xin=x_sol_tk,
            Pin=Pxx_sol_tk,
            t=tk,
            transStateObj=self.mx_posVel_prior_tk,
            mekfStateObj=self.mx_mekf_prior_tk
        )

        # update mekf prior state obj at end of prop
        self.mx_mekf_prior_tk.t = tk
        self.mx_mekf_prior_tk.q_BMref = copy.deepcopy(q_BM_tk_obj)
        self.mx_mekf_prior_tk.q_BMref.ensureScalarPos()
        self.mx_mekf_prior_tk.gyroBiasRef = self.mx_mekf_prior_tk_.gyroBiasRef

        ## full state
        self.mx_full.t = tk
        self.mx_full.q_BMref = copy.deepcopy(q_BM_tk_obj)
        self.mx_full.q_BMref.ensureScalarPos()



        # log states after propagation
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_prior_tk))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_prior_tk))



    def updateWithLandmarks(self, z_meas_matrix, PvvBodyFrame, measTime):       

        self.Pvv = PvvBodyFrame
        # get column of id's
        landmarkIds = z_meas_matrix[:,-1].astype(int)
        landmarkMeas = z_meas_matrix[:,:3]

        ## handle GMF vs EKF impl
        if self._GMF_FLAG:
            mr_BM_M_tk = self.mx_full.mx[0:3].flatten()
            q_BM_ref = self.mx_full.q_BMref

        else:
            # get position at measurement time
            mr_BM_M_tk = self.mx_posVel_prior_tk.r_BM_M_mean.flatten()
            q_BM_ref = self.mx_mekf_prior_tk.q_BMref

        # construct measurement matrix
        HxStack = None
        PvvStack = None
        mzkStack = None
        innovationsVec = None

        # create innovation objects list
        innObjList = []
        for i in range(z_meas_matrix.shape[0]):
            innObjList.append(copy.deepcopy(LandMarkInnovation()))



        for i in range(z_meas_matrix.shape[0]):

            # get map landmark position
            q_BM_ref.ensureScalarPos()
            map_r_LM_M = self.landmarkMap[landmarkIds[i],:].flatten()
            map_r_LM_B= q_BM_ref.rotate(
                map_r_LM_M
            ) 

            # compute mean of measurement model
            mzk = q_BM_ref.rotate(map_r_LM_M - mr_BM_M_tk)
            if mzkStack is None:
                mzkStack = mzk.reshape(-1,1)
            else:
                mzkStack = np.vstack((mzkStack.reshape(-1,1), mzk.reshape(-1,1)))

            innovation = (landmarkMeas[i,:].flatten() - mzk).reshape((3,1))
            # log innovation
            innObj = innObjList[i]
            innObj.t = measTime
            innObj.innovation = innovation.reshape((3,1))
            innObj.landmarkId = landmarkIds[i]

            # translation Hx with nx_fullstate columns
            TBMhat = q_BM_ref.to_dcm()
            HxTrans = np.hstack((-TBMhat, np.zeros((self.nz, 3))))

            # MEKF Hx with nx_fullstate columns
            # skew of mean of mcmf to body frame position
            HxMekf = np.zeros((self.nz,self.nx_mekf))
            rx,ry,rz = TBMhat@map_r_LM_M
            map_r_LM_B_skew = np.array([
                [0, -rz, ry],
                [rz, 0, -rx],
                [-ry, rx, 0]
            ])
            rx,ry,rz = TBMhat@mr_BM_M_tk
            mr_BM_B_tk_skew = np.array([
                [0, -rz, ry],
                [rz, 0, -rx],
                [-ry, rx, 0]
            ])
            HxMekf[:,:3] = map_r_LM_B_skew - mr_BM_B_tk_skew

            

            if HxStack is not None:
                Hxi = np.concatenate((HxTrans,HxMekf),axis=1)
                HxStack = np.concatenate((HxStack,Hxi),axis=0)
                PvvStack = block_diag(
                    PvvStack,
                    self.Pvv)
                innovationsVec = np.vstack((innovationsVec, innovation.reshape(-1,1)))
            else:
                HxStack = np.concatenate((HxTrans,HxMekf),axis=1)
                # HxStack = HxTrans 
                # HxStack = HxMekf
                PvvStack = self.Pvv
                innovationsVec = innovation.reshape(-1,1)
                
        # prepare for kalman update
        Pxxk_prior = self.mx_full.Pxx
        # Pxxk_prior = self.mx_mekf_prior_tk.Pxx
        Pxzk = Pxxk_prior @ HxStack.T
        Pzzk = (HxStack @ Pxxk_prior @ (HxStack.T)) + PvvStack
        Kk = Pxzk @ linalg.inv(Pzzk)

        ## EGMF weights case
        # note an implicit assumption of a single measurement here
        if self._GMF_FLAG:
            self.mx_full.k = gaussian_pdf(
                x=landmarkMeas.flatten(),
                mean=mzkStack,
                cov=Pzzk)

        # store innovation covariance
        block_size = self.nz
        for i, innObj in enumerate(innObjList):
            start = i * block_size
            stop = start + block_size
            innObj.innovationCov = Pzzk[start:stop, start:stop]
            if not self._GMF_FLAG:
                self.innovation_log.append(copy.deepcopy(innObj))

        # create full state 
        if self._GMF_FLAG:
            mxk_prior = self.mx_full.mx.reshape((-1,1))
        else:
            mxk_prior = np.concatenate((
            self.mx_posVel_prior_tk.r_BM_M_mean.flatten(),
            self.mx_posVel_prior_tk.Mdrdt_BM_M_mean.flatten(),
            self.mx_mekf_prior_tk.angleError_mean.flatten(),
            self.mx_mekf_prior_tk.gyroBiasError_mean.flatten()
        ),axis=0).reshape((-1,1))
        
        # kalman update
        mxk_post = mxk_prior + Kk @ innovationsVec

        # try Joseph's Formulation of the covariance update eq
        I12 = np.eye(self.nx_full)

        # Kalman Gain with non-linear measurements
        Pxxk_post = (I12 - Kk@HxStack)@Pxxk_prior

        # Joseph's
        # Pxxk_post = (I12 - Kk@HxStack) @ Pxxk_prior @ (I12 - Kk@HxStack).T + Kk@PvvStack@Kk.T
        # Basic Covariance update
        # Pxxk_post = Pxxk_prior - Pxzk@Kk.T - Kk@Pxzk.T + Kk @ Pzzk @ Kk.T

        # unpack posterior
        self.unpackFullState(
            xin=mxk_post,
            Pin=Pxxk_post,
            t=measTime,
            transStateObj=self.mx_posVel_post_tk,
            mekfStateObj=self.mx_mekf_post_tk
        )

         # if any negative diagonal entries, stop the update
        if np.any(self.mx_full.Pxx[np.diag_indices_from(self.mx_full.Pxx)] < 0.0):
            raise ValueError(
                f"Invalid covariance: Pxx has negative diagonal entries after measurement update at time {measTime}"
            )


        # add attitude error correction to nominal quaternion
        q_err = Quaternion(qv=self.mx_mekf_post_tk.angleError_mean.flatten()/2,
                           q0=1.0)
        q_BM_post = (q_err*q_BM_ref)
        q_BM_post.ensureScalarPos()
        if np.abs(q_BM_post.scalar()-1.)>1e-7:
            q_BM_post.normalize()
        
        ## handle EGMF impl
        if self._GMF_FLAG:
            self.mx_full.q_BMref = copy.deepcopy(q_BM_post)
        else:    
            self.mx_mekf_post_tk.q_BMref = copy.deepcopy(q_BM_post)

        # add gyro bias error correction to nominal bias
        if self._GMF_FLAG:
            gyroBiasRef = self.mx_full.gyroBiasRef.flatten()
            gyroBiasError_mean =  mxk_post[9:].flatten()
        else:
            gyroBiasRef = self.mx_mekf_prior_tk.gyroBiasRef.flatten()
            gyroBiasError_mean = self.mx_mekf_post_tk.gyroBiasError_mean.flatten()
        gyroBias_post = (gyroBiasRef +
                         gyroBiasError_mean)
        
        if self._GMF_FLAG:
            self.mx_full.gyroBiasRef = gyroBias_post.flatten()
        else:
            self.mx_mekf_post_tk.gyroBiasRef = gyroBias_post.flatten()


        # set error state to 0
        self.mx_mekf_post_tk.angleError_mean = np.zeros((3,1))
        self.mx_mekf_post_tk.gyroBiasError_mean = np.zeros((3,1))
        
        self.mx_full.mx[6:] = np.zeros((6,1))

         ## If decoupled, diagonal full state covariance
        if self._DECOUPLED_FLAG:
            # Upper-right block
            self.mx_full.Pxx[0:self.nx_posVel, self.nx_posVel:self.nx_posVel+self.nx_mekf] = 0.0

            # Lower-left block
            self.mx_full.Pxx[self.nx_posVel:self.nx_posVel+self.nx_mekf, 0:self.nx_posVel] = 0.0


        # log updated states
        self.posVelState_log.append(copy.deepcopy(self.mx_posVel_post_tk))
        self.mekfState_log.append(copy.deepcopy(self.mx_mekf_post_tk))


    def unpackFullState(self, xin, Pin, t, transStateObj, mekfStateObj):
        # unpack states
        transStateObj.r_BM_M_mean = xin[:3].flatten()
        transStateObj.Mdrdt_BM_M_mean = xin[3:6].flatten()
        mekfStateObj.angleError_mean = xin[6:9].flatten()
        mekfStateObj.gyroBiasError_mean = xin[9:].flatten()

        # unpack covariance
        transStateObj.Pxx = Pin[:6,:6]
        mekfStateObj.Pxx = Pin[6:,6:]

        # update times
        transStateObj.t = t
        mekfStateObj.t = t
        
        # construct full state error covariance post propagation
        self.mx_full.Pxx = Pin
        self.mx_full.mx = xin
        self.mx_full.t = t 




