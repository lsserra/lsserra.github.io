import numpy as np

class Quaternion:
    """
    Vector-first quaternion class: q = [qv; q0]
    where qv = (3,) vector part, q0 = scalar part.
    """

    def __init__(self, qv=None, q0=None):
        if qv is None and q0 is None:
            # identity quaternion
            self.q = np.array([0., 0., 0., 1.])
        elif qv is not None and q0 is not None:
            self.q = np.hstack((np.array(qv, dtype=float), float(q0)))
        else:
            raise ValueError("Provide both qv and q0 or neither.")

    @classmethod
    def from_array(cls, arr):
        """Initialize from a 4-element array [q1, q2, q3, q0]."""
        return cls(arr[:3], arr[3])

    def as_array(self):
        return self.q.copy()

    def vector(self):
        return self.q[:3]

    def scalar(self):
        return self.q[3]

    # --- Core Quaternion Operations ---

    def conj(self):
        """Quaternion conjugate (inverse for unit quaternion)."""
        qv, q0 = self.vector(), self.scalar()
        return Quaternion(-qv, q0)

    def norm(self):
        return np.linalg.norm(self.q)

    def normalize(self):
        self.q /= self.norm()
        return self
    
    def ensureScalarPos(self):
        if self.scalar() < 1e-16:
            self.q *= -1.0    # flips all 4 components in place
        return self


    def inverse(self):
        """Inverse quaternion (for unit q, same as conjugate)."""
        qv, q0 = self.vector(), self.scalar()
        norm2 = np.dot(self.q, self.q)
        return Quaternion(-qv / norm2, q0 / norm2)

    # --- Quaternion Multiplication ---

    def __mul__(self, other):
        """Quaternion multiplication (⊗)"""
        if not isinstance(other, Quaternion):
            raise TypeError("Quaternion can only multiply another Quaternion.")
        p, q = self.q, other.q
        p_v, p_0 = p[:3], p[3]
        q_v, q_0 = q[:3], q[3]
        v = p_0 * q_v + q_0 * p_v - np.cross(p_v, q_v)
        s = p_0 * q_0 - np.dot(p_v, q_v)
        return Quaternion(v, s)

    # --- Rotation Matrix Conversion ---

    def to_dcm(self):
        """Convert to passive DCM (frame rotation)."""
        qv = self.vector()
        q0 = self.scalar()
        qx, qy, qz = qv
        I = np.eye(3)
        qx_skew = np.array([
            [0, -qz, qy],
            [qz, 0, -qx],
            [-qy, qx, 0]
        ])
        return I - 2*q0*qx_skew + 2*np.linalg.matrix_power(qx_skew, 2)

    # --- Rotation of Vectors ---

    def rotate(self, v):
        """Rotate a 3-vector v using this quaternion (passive rotation) q v x q^-1."""
        v = np.array(v, dtype=float)
        q_inv = self.inverse()
        v_pure = Quaternion(v, 0.0)
        rot_pure = self * v_pure * q_inv 
        return rot_pure.vector()
    

    # --- Return q dot given angluar rate, as an np array --
    @staticmethod
    def dqdt(t,omega,q):
        qv = q[:3]
        q0 = q[-1]
        wx, wy, wz = omega

        qx, qy, qz = qv
        omega_skew = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        Omega = np.zeros((4,4))
        Omega[:3,:3] = -omega_skew
        Omega[:3,3] = omega
        Omega[3,:3] = -omega
        qdot = 0.5 * Omega@q
        
        return qdot

    # --- Static Constructors ---

    @staticmethod
    def from_DCM(T):
        """
        Create a vector-first quaternion (passive rotation) from a 3x3 DCM.
        The DCM maps coordinates from frame N to frame B: v_B = T * v_N
        """
        T = np.array(T, dtype=float)
        tr = np.trace(T)

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # 4*q0
            q0 = 0.25 * S
            q1 = (T[2, 1] - T[1, 2]) / S
            q2 = (T[0, 2] - T[2, 0]) / S
            q3 = (T[1, 0] - T[0, 1]) / S
        else:
            # find major diagonal element and compute accordingly
            if (T[0, 0] > T[1, 1]) and (T[0, 0] > T[2, 2]):
                S = np.sqrt(1.0 + T[0, 0] - T[1, 1] - T[2, 2]) * 2
                q0 = (T[2, 1] - T[1, 2]) / S
                q1 = 0.25 * S
                q2 = (T[0, 1] + T[1, 0]) / S
                q3 = (T[0, 2] + T[2, 0]) / S
            elif T[1, 1] > T[2, 2]:
                S = np.sqrt(1.0 + T[1, 1] - T[0, 0] - T[2, 2]) * 2
                q0 = (T[0, 2] - T[2, 0]) / S
                q1 = (T[0, 1] + T[1, 0]) / S
                q2 = 0.25 * S
                q3 = (T[1, 2] + T[2, 1]) / S
            else:
                S = np.sqrt(1.0 + T[2, 2] - T[0, 0] - T[1, 1]) * 2
                q0 = (T[1, 0] - T[0, 1]) / S
                q1 = (T[0, 2] + T[2, 0]) / S
                q2 = (T[1, 2] + T[2, 1]) / S
                q3 = 0.25 * S

        q = Quaternion([q1, q2, q3], q0)
        return q.normalize()


    @staticmethod
    def from_axis_angle(axis, angle):
        """Create quaternion from unit axis and rotation angle (rad)."""
        axis = np.array(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        qv = np.sin(angle / 2) * axis
        q0 = np.cos(angle / 2)
        return Quaternion(qv, q0)

    @staticmethod
    def identity():
        return Quaternion([0, 0, 0], 1.0)
    
    @staticmethod
    def computeEulerVecAttErrorFromQuats(q_ref: "Quaternion", q_est: "Quaternion"):
        """
        
        given these are attitde quaternions (rotation from Ref Frame to Tgt Frame)
        this function returns the Euler Vector of the error in the Tgt frame in radians
        
        """
        # Ensure both quaternions have positive scalar parts
        q_ref.ensureScalarPos()
        q_est.ensureScalarPos()

        # Quaternion attitude error (reference → estimated) expressed in object frame
        q_err = q_ref * q_est.inverse()

        # Convert to array form [qx, qy, qz, q0]
        q_err_arr = q_err.as_array()

        # Ensure positive scalar part for uniqueness
        if q_err_arr[-1] < 0:
            q_err_arr = -q_err_arr

        q0 = q_err_arr[-1]

        # Handle small angle numerically
        if (np.abs((q0 - 1.0)) < 1e-12):
            return np.zeros(3)

        # Principal rotation angle and axis
        phi = 2 * np.arccos(q0)
        ehat = q_err_arr[:3] / np.sin(phi/2)

        # Euler (principal) rotation vector
        return phi * ehat


    # --- Representation ---

    def __repr__(self):
        qv, q0 = self.vector(), self.scalar()
        return f"Quaternion(qv={qv}, q0={q0:.6f})"
