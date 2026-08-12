import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from prjs.aero626.ekfMoon.EkfPoseEstimator import(
    EkfPoseEstimator,
    FullFilterState
) 
from helpers.attitude.Quaternion import Quaternion



class MoonEGMF():
    def __init__(self):
        
        ## list of gaussian pdfs
        self.gaussianPdfList_ = []

        ## pull in ekf for pdf propgation and update utility
        self._ekf = None
        self.Pwwkm1 = None

        

        
        #### STORAGE ####
        ## list to store gmLists at each epoch
        self.storeGmList_ = []

        ## list to store GM mean, cov at each up
        self.storeGmBestGuess_ = []

        ## list to store residual and residual


    def PropagateMixtureEkf(self,toTime,w_BN_B_meas):
        for k,g in enumerate(self.gaussianPdfList_):
            self._ekf.mx_full = g
            
            self._ekf.propagate(toTime=toTime,w_BN_B_meas=w_BN_B_meas)
            ## unique add discrete process noise
            g.Pxx = g.Pxx + self.Pwwkm1
            
            

        # compute some GM stats
        mean,cov = self.computeBestEstMeanAndCovAtEpoch()
        gyroBiasEst,q_BM_est= self.computeBestEstQuaternionAndGryoBias()
        GMstate = FullFilterState(nx=len(self._ekf.mx_full.mx))
        GMstate.mx = mean
        GMstate.Pxx = cov
        GMstate.t = toTime
        GMstate.gyroBiasRef = gyroBiasEst
        GMstate.q_BMref = q_BM_est
        self.storeGmBestGuess_.append(copy.deepcopy(GMstate))
   

    def LandmarkMeasUpdateEkf(self, z_meas_matrix, PvvBodyFrame, measTime):
        for k,g in enumerate(self.gaussianPdfList_):
            self._ekf.mx_full = g
            self._ekf.updateWithLandmarks(z_meas_matrix, PvvBodyFrame, measTime)

        # update weights 
        self.UpdateWeights()

        # compute some GM stats
        mean,cov = self.computeBestEstMeanAndCovAtEpoch()
        gyroBiasEst,q_BM_est= self.computeBestEstQuaternionAndGryoBias()
        GMstate = FullFilterState(nx=len(self._ekf.mx_full.mx))
        GMstate.mx = mean
        GMstate.Pxx = cov
        GMstate.t = measTime
        GMstate.gyroBiasRef = gyroBiasEst
        GMstate.q_BMref = q_BM_est
        self.storeGmBestGuess_.append(copy.deepcopy(GMstate))
    
   
    def UpdateWeights(self):
        # # compute normalization factor
        # denom = 0.
        # for i, gm in enumerate(self.gaussianPdfList_):
        #     denom += gm.k*gm.w

        # # normalize posterior weights
        # for i, gm in enumerate(self.gaussianPdfList_):
        #     gm.w = gm.k*gm.w/denom
        # compute unnormalized weights safely
        unnorm = np.array([gm.k * gm.w for gm in self.gaussianPdfList_], dtype=np.float64)

        # avoid underflow: floor tiny weights
        eps = 1e-300
        unnorm = np.clip(unnorm, eps, None)

        denom = np.sum(unnorm)

        # protect denominator from underflow
        if denom < eps or np.isnan(denom):
            # fallback: assign equal weights
            n = len(self.gaussianPdfList_)
            for gm in self.gaussianPdfList_:
                gm.w = 1.0 / n
            return

        # normalize
        for gm, u in zip(self.gaussianPdfList_, unnorm):
            gm.w = u / denom
    

    def computeBestEstMeanAndCovAtEpoch(self): 
        mean = np.zeros_like(self.gaussianPdfList_[0].mx)
        for i, gm in enumerate(self.gaussianPdfList_):
            mean += gm.w*gm.mx

        cov = np.zeros_like(self.gaussianPdfList_[0].Pxx)
        for i, gm in enumerate(self.gaussianPdfList_):
            cov += gm.w*(gm.Pxx + (gm.mx - mean)*(gm.mx - mean).T)
            
        return mean, cov
    
    def computeBestEstQuaternionAndGryoBias(self):
        # weighted average for gryo bias estimate
        gyroBiasEst = np.zeros_like(self.gaussianPdfList_[0].gyroBiasRef)
        for i, gm in enumerate(self.gaussianPdfList_):
            gyroBiasEst += gm.w*gm.gyroBiasRef
        # take most highest weighted est for now 
        # q_BM_est = Quaternion()
        maxWeight = 0.
        for i, gm in enumerate(self.gaussianPdfList_):
            if gm.w > maxWeight:
                q_BM_est = copy.deepcopy(gm.q_BMref)
                maxWeight = gm.w

        return gyroBiasEst, q_BM_est

    
    def sampleFromThisGaussianMixList(self,seed=None):
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)
        # Extract weights in the same order as gaussianPdfList_
        weights = np.array([g.w for g in self.gaussianPdfList_], dtype=float)

        # Draw a component index according to mixture weights
        intArray =np.arange(0,len(self.gaussianPdfList_),1,dtype=int)
        l = rng.choice(a=intArray, p=weights)

        # Pull out mean and covariance
        mean = self.gaussianPdfList_[l].mx
        cov  = self.gaussianPdfList_[l].Pxx

        # Sample from the selected Gaussian
        sample = rng.normal(loc=mean, scale=np.sqrt(cov))

        return sample
    



class MoonGaussianMixtureModel():
    def __init__(self, Lx_input):
        self._nx = 12
        self.Lx = Lx_input
        ## list of full state objs
        self._gaussianPdfList = []

        ## saving figures
        self._EXPORT_FIGURES_FLAG = False
        self._dataDir = ""


    def pdf_based_weights(self, referenceGaussian):
        """
        mxs     : list/array of component means, shape (Lx, nx)
        ref_mx  : reference Gaussian mean
        ref_Pxx : reference Gaussian covariance
        """
        Lx = self.Lx
        pdf_vals = np.zeros(Lx)

        for k in range(Lx):
            pdf_vals[k] = MoonGaussianMixtureModel.gaussian_pdf(
                x=np.asarray(self._gaussianPdfList[k].mx), 
                mean=np.asarray(referenceGaussian.mx), 
                cov=np.asarray(referenceGaussian.Pxx)
            )

        # normalize
        weights = pdf_vals / pdf_vals.sum()

        for k, g in enumerate(self._gaussianPdfList):
            g.w = weights[k]

    def visualizeGmm(self, stateIdxOfInterest, referenceGaussian, numSigma=4, numPoints=800):
        """
        Visualize the 1D marginal pdf of a GMM along a chosen state index,
        including the reference Gaussian, all mixture components, and the final mixture pdf.

        Parameters
        ----------
        stateIdxOfInterest : int
            Index of the state variable to visualize.
        referenceGaussian : object with .mx (mean vector) and .Pxx (covariance matrix)
        numSigma : float
            Range for x-axis = mean ± numSigma * sigma
        numPoints : int
            Number of points in xEval
        """

        # ---------------------------------------------
        # Extract reference mean & sigma for this 1D component
        # ---------------------------------------------
        mx_ref = referenceGaussian.mx[stateIdxOfInterest]
        std_ref = np.sqrt(referenceGaussian.Pxx[stateIdxOfInterest, stateIdxOfInterest])

        # intelligent x-range: mean ± numSigma * std
        x_min = mx_ref - numSigma * std_ref
        x_max = mx_ref + numSigma * std_ref
        xEval = np.linspace(x_min, x_max, numPoints)

        # ---------------------------------------------
        # Evaluate reference Gaussian in 1D
        # ---------------------------------------------
        def gaussian_1d(x, mean, var):
            return 1/np.sqrt(2*np.pi*var) * np.exp(-(x-mean)**2/(2*var))

        ref_pdf = gaussian_1d(xEval, mx_ref, std_ref**2)

        # ---------------------------------------------
        # Evaluate GMM components + mixture pdf
        # ---------------------------------------------
        gmm = self._gaussianPdfList

        component_pdfs = []
        mixture_pdf = np.zeros_like(xEval)

        for g in gmm:
            mx_i = g.mx[stateIdxOfInterest]
            var_i = g.Pxx[stateIdxOfInterest, stateIdxOfInterest]

            pdf_i = gaussian_1d(xEval, mx_i, var_i)
            component_pdfs.append(pdf_i)

            mixture_pdf += g.w * pdf_i

        # ---------------------------------------------
        # Plot results
        # ---------------------------------------------
        plt.figure(figsize=(9,6))

        # Plot reference Gaussian
        plt.plot(xEval, ref_pdf, 'k-', linewidth=2, label="Reference Gaussian")

        # Plot each Gaussian component
        colors = plt.cm.viridis(np.linspace(0, 1, len(component_pdfs)))

        for idx, pdf_i in enumerate(component_pdfs):
            plt.plot(
                xEval,
                pdf_i,
                '--',
                alpha=0.9,
                color=colors[idx],
                label=f"Component {idx+1}"
            )

        # Plot combined GMM pdf
        plt.plot(xEval, mixture_pdf, 'r-', linewidth=2.5, label="GMM Mixture PDF")

       

        plt.grid(True)
        plt.xlabel(f"x")
        plt.ylabel("pdf(x)")
        # plt.title("Gaussian Mixture Model – 1D PDF")
        plt.legend()
        plt.tight_layout()

        figTitleSave =f"GMM_state_{stateIdxOfInterest}"
        if self._EXPORT_FIGURES_FLAG:
            fig_path = os.path.join(self._dataDir,'figures')
            out_path = os.path.join(fig_path,f"{figTitleSave}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")


    @ staticmethod
    def gaussian_to_gmm(mx, Pxx, Lx=5, spread_sigma=3.0):
        mx = np.atleast_1d(mx)
        n = mx.shape[0]

        # --- Compute eigen decomposition of the covariance ---
        vals, vecs = np.linalg.eigh(Pxx)  # vals = eigenvalues (variances), vecs = eigenvectors

        stds = np.sqrt(vals)               # standard deviations along principal axes

        # --- Select K points evenly spread across eigen-directions ---
        # Example: K=5 produces positions [-3σ, -1.5σ, 0, 1.5σ, 3σ]
        a = np.linspace(-spread_sigma, spread_sigma, Lx)

        mxs = []
        for i in range(Lx):
            # Offset in principal-axis coordinates
            offset = (a[i] * stds)
            # Transform back to original coordinates
            new_mx = mx.flatten() + vecs @ offset
            mxs.append(new_mx)

        # All components share the original covariance (adjustable)
        Pxx = [Pxx.copy() for _ in range(Lx)]

        # Equal weights
        weights = np.ones(Lx) / Lx

        return weights, mxs, Pxx
    

    ## Function to facilitate EGMF impl
    @staticmethod
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
        





        