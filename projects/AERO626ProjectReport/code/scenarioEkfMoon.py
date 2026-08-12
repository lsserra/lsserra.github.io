
import os, sys
from copy import copy
import pickle
from Basilisk.topLevelModules import pyswice
from datetime import datetime, timedelta
from Basilisk.simulation import imuSensor
from Basilisk.utilities import RigidBodyKinematics as rbk

# import message declarations
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D
# import FSW Algorithm related support
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav

from Basilisk.utilities.pyswice_spk_utilities import spkRead


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from helpers.attitude.Quaternion import Quaternion



import matplotlib.pyplot as plt
import numpy as np
# To play with any scenario scripts as tutorials, you should make a copy of them into a custom folder
# outside of the Basilisk directory.
#
# To copy them, first find the location of the Basilisk installation.
# After installing, you can find the installed location of Basilisk by opening a python interpreter and
# running the commands:
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# Copy the folder `{basiliskPath}/examples` into a new folder in a different directory.
# Now, when you want to use a tutorial, navigate inside that folder, and edit and execute the *copied* integrated tests.


# import simulation related support
from Basilisk.simulation import spacecraft
# general support file with common unit test functions
# import general simulation support files
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport, vizSupport)

# always import the Basilisk messaging support

def run(showPlots, savePkl, EarthAndMoonGrav):
   
    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # (Optional) If you want to see a simulation progress bar in the terminal window, the
    # use the following SetProgressBar(True) statement
    scSim.SetProgressBar(True)

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(.1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # setup the simulation tasks/objects
    # initialize spacecraft object and set properties
    # The dynamics simulation is setup using a Spacecraft() module.
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # setup Gravity Body
    # The first step to adding gravity objects is to create the gravity body factor class.  Note that
    # this call will create an empty gravitational body list each time this script is called.  Thus, there
    # is not need to clear any prior list of gravitational bodies.
    gravFactory = simIncludeGravBody.gravBodyFactory()

     # Setup gravity factory and gravity bodies
    # Include bodies as a list of SPICE names
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = None
    if EarthAndMoonGrav:
        gravBodies = gravFactory.createBodies('moon','earth')

    else:
        gravBodies = gravFactory.createBodies('moon')
    
    gravBodies['moon'].isCentralBody = True
    moonBody = gravBodies.get('moon')

    # Add gravity bodies to the spacecraft dynamics
    gravFactory.addBodiesTo(scObject)

    # Create default SPICE module, specify start date/time.
    timeInitString = "2022 August 31 15:00:00.0"
    spiceTimeStringFormat = '%Y %B %d %H:%M:%S.%f'
    timeInit = datetime.strptime(timeInitString, spiceTimeStringFormat)
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    # spiceObject.zeroBase = 'Earth'

    print(spiceObject.planetFrames)

    #
    #   setup orbit and simulation time
    #
    # setup the orbit using classical orbit elements
    
    oe = orbitalMotion.ClassicElements()
    rLLO = moonBody.radEquator + 2000       # meters
    oe.a = rLLO
    oe.e = 0.00001
    oe.i = 30.0 * macros.D2R
    oe.Omega = 30.0 * macros.D2R
    oe.omega = 0.0 * macros.D2R
    oe.f = 0.0 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(moonBody.mu, oe)
    oe = orbitalMotion.rv2elem(moonBody.mu, rN, vN)   

    # Add SPICE object to the simulation task list
    scSim.AddModelToTask(simTaskName, spiceObject, 1)

    # Import SPICE ephemeris data into the python environment
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'de430.bsp')  # solar system bodies
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'naif0012.tls')  # leap second file
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'de-403-masses.tpc')  # solar system masses
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'pck00010.tpc')  # generic Planetary Constants Kernel


    # define the simulation inertia
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    # I = [1., 0., 0.,
    #      0., 1., 0.,
    #      0., 0., 1.]

    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # initial tip off
    ### SPACECRAFT
    scObject.hub.sigma_BNInit = rbk.PRV2MRP([macros.D2R*0.0, 0.0, macros.D2R*0.0]) # rbk.C2MRP(np.identity(3))  # sigma_BN_B
    # scObject.hub.omega_BN_BInit = [macros.D2R*5.0, macros.D2R*20.0, macros.D2R*1.0]  # rad/s - omega_BN_B
    scObject.hub.omega_BN_BInit = [macros.D2R*5.0, macros.D2R*2.0, macros.D2R*1.0]  # rad/s - omega_BN_B
    

    # set the simulation time
    n = np.sqrt(moonBody.mu / oe.a / oe.a / oe.a)
    P = 2. * np.pi / n
    simulationTime = macros.sec2nano(1.*P)
    # Setup data logging
    numDataPoints = np.round(simulationTime/simulationTimeStep)
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)



    # --- add imu --- #
    imu = imuSensor.ImuSensor()
    imu.ModelTag = "imu"    
    
    # Configure gyro noise (rad/s)
    gryoBias = np.deg2rad(0.1) / 3600 # deg/hr to rad/s
    # imu.senRotBias = np.array([0.,0.,0.])
    senRotNoiseStd = np.sqrt(10)*10**(-7) # rad/sec^(1/2)
    # walkBound = 0.01 # rad/s
    PMatrix = np.eye(3)* senRotNoiseStd**2 # cholesky defactorization of noise covariance matrix, drives Gauss Markov Process
    L = np.linalg.cholesky(PMatrix)
    # L = np.zeros((3,3))
    imu.PMatrixGyro = L
    AMatrixGyro = np.zeros((3,3))

    senWalkNoiseStd = np.sqrt(10)*10**(-10) # rad/sec^(3/2)
    AMatrixGyro = np.eye(3)* senWalkNoiseStd**2 # cholesky defactorization of noise covariance matrix, drives Gauss Markov Process
    L = np.linalg.cholesky(AMatrixGyro)
    imu.AMatrixGyro = L
    # imu.AMatrixGyro = np.zeros((3,3))
    # imu.setErrorBoundsGyro([walkBound, walkBound, walkBound])
    imu.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    # Add IMU to simulation
    scSim.AddModelToTask(simTaskName, imu)

    
    # Setup spacecraft data recorder
    scDataRec = scObject.scStateOutMsg.recorder(samplingTime)
    MoonDataRec = spiceObject.planetStateOutMsgs[0].recorder(samplingTime)
    EarthDataRec = None
    if EarthAndMoonGrav:
        EarthDataRec = spiceObject.planetStateOutMsgs[1].recorder(samplingTime)
        scSim.AddModelToTask(simTaskName, EarthDataRec)

    scSim.AddModelToTask(simTaskName, scDataRec,ModelPriority=5)
    scSim.AddModelToTask(simTaskName, MoonDataRec,ModelPriority=4)
    # Set up messages for both IMU's
    imuDataRec = imu.sensorOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, imuDataRec)
    
#######################
# CONTROLLER SETUP
#######################
   # setup extForceTorque module
    # the control torque is read in through the messaging system
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject)

    # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)

    #
    #   setup the FSW algorithm tasks
    #

    # setup inertial3D guidance module
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DObj)
    inertial3DObj.sigma_R0N = [0., 0., 0.]  # set the desired inertial orientation

    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)

    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    # scSim.AddModelToTask(simTaskName, mrpControl)
    mrpControl.K = 3.5
    mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 30.0

    # mrpControl.K = 0.1         # Reduced Gain for stability testing
    # mrpControl.Ki = 0.1        # Set to 0.0 to turn off integral feedback
    # mrpControl.P = 0.1    
    # mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

    #
    #   Setup data logging before the simulation is initialized
    #
    # numDataPoints = 50
    # samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    mrpLog = mrpControl.cmdTorqueOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, mrpLog)

    #
    # create simulation messages
    #

    # create the FSW vehicle configuration message
    # use the same inertia in the FSW algorithm as in the simulation
    vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
    configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    #
    # connect the messages to the modules
    #
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    extFTObject.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)

    # if this scenario is to interface with the BSK Viz, uncomment the following lines
    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                         , saveFile=fileName
                                        )
    viz.settings.showVelocityFrame = 1
    viz.settings.spacecraftCSon = 1
    viz.settings.planetCSon = 1
    

    # Initialize simulation
    scSim.InitializeSimulation()

    # Execute simulation
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # toss away first data point
    # Retrieve logged data
    posData = scDataRec.r_BN_N[1:,:]
    velData = scDataRec.v_BN_N[1:,:]
    mrpBN = scDataRec.sigma_BN[1:,:]
    timeData = scDataRec.times()[1:]
    moonPos = MoonDataRec.PositionVector[1:,:]
    moonVel = MoonDataRec.VelocityVector[1:,:]

    gryoAngVel = imuDataRec.AngVelPlatform[1:,:]
    gyroTime = imuDataRec.times()[1:]

    earthPos = None
    earthVel = None
    if EarthAndMoonGrav:
        earthPos = EarthDataRec.PositionVector
        earthVel = EarthDataRec.VelocityVector



    posData[:] -= moonPos[:]
    velData[:] -= moonVel[:]


    ## convert true s/c MRP N to B attitude to quaternion representation
    q_BN_truth_list = []
    for att_i in range(mrpBN.shape[0]):
        # mrp to quaternion
        q_BN_i = rbk.MRP2EP(mrpBN[att_i,:])
        # vector first quaternion object
        q_BN_truth = Quaternion(q0=q_BN_i[0],qv=q_BN_i[1:]).normalize()
        q_BN_truth_list.append(q_BN_truth)
    
    

    # Bundle everything in one dictionary
    if savePkl:
        data_bundle = {
            "time": timeData,
            "sc_pos": posData,
            "sc_vel": velData,
            "q_BN_truth":q_BN_truth_list,
            "moon_pos": moonPos,
            "moon_vel": moonVel,
            "earth_pos": earthPos,
            "earth_vel": earthVel,
            "gryoAngVel": gryoAngVel,
            "timeGyro": gyroTime
        }
        
        
        if EarthAndMoonGrav:
            # Write to pickle file
            with open("data/MoonCentralBody_MoonEarthGrav.pkl", "wb") as f:
                pickle.dump(data_bundle, f)
            print("Simulation data boxed up into data/MoonCentralBody_MoonEarthGrav.pkl")
        else:
            # Write to pickle file
            with open("data/MoonCentralBody_MoonGrav.pkl", "wb") as f:
                pickle.dump(data_bundle, f)
            print("Simulation data boxed up into data/MoonCentralBody_MoonGrav.pkl")
            

        

    

    #   plot the results
    #
    # draw the inertial position vector components
    plt.close("all")  # clears out plots from earlier test runs
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    for idx in range(3):
        plt.plot(timeData * macros.NANO2SEC / P, posData[:, idx] / 1000.,
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$r_{BN,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [orbits]')
    plt.ylabel('Inertial Position [km]')
    figureList = {}
    pltName = fileName + "1"
    figureList[pltName] = plt.figure(1)


    # draw the planet
    plt.figure(2)
    fig = plt.gcf()
    ax = fig.gca()
    planetColor = '#555555'
    planetRadius = moonBody.radEquator / 1000
    ax.add_artist(plt.Circle((0, 0), planetRadius, color=planetColor))
    # draw the actual orbit
    rData = []
    fData = []
    for idx in range(0, len(posData)):
        oeData = orbitalMotion.rv2elem(moonBody.mu, posData[idx], velData[idx])
        rData.append(oeData.rmag)
        fData.append(oeData.f + oeData.omega - oe.omega)
    plt.plot(posData[:,0] / 1000, posData[:,1] / 1000, color='#aa0000', linewidth=3.0)
    plt.xlabel("X-Intertial [km]")
    plt.ylabel("Y-Intertial [km]")
    
    pltName = fileName + "2"
    figureList[pltName] = plt.figure(2)
    


    plt.figure(3)
    alt = np.linalg.norm(posData,axis=1) - moonBody.radEquator
    plt.plot(timeData * macros.NANO2SEC / P,alt/1000.)
    plt.ylabel('Altitude [km]')
    plt.xlabel('Time [orbits]')
    # disable scientific notation on y-axis
    plt.gca().ticklabel_format(style='plain', axis='y')



    gryoAngVel_array = np.array(gryoAngVel)
    stdGryo = np.std(gryoAngVel_array[1:,:],axis=0)
    print(f"Gryo STD (rad/s): ")
    print(stdGryo)
    plt.figure(4, figsize=(12, 8))
    plt.plot(gyroTime * macros.NANO2SEC,
         macros.R2D * gryoAngVel_array[:, 0],
            label='X', alpha=0.7)

    plt.plot(gyroTime * macros.NANO2SEC,
            macros.R2D * gryoAngVel_array[:, 1],
            label='Y', alpha=0.7)

    plt.plot(gyroTime * macros.NANO2SEC,
            macros.R2D * gryoAngVel_array[:, 2],
            label='Z', alpha=0.7)

    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [deg/s]')
    # plt.title('IMU Gyro Measurements')
    plt.legend()
    plt.grid(True)


      #
    #   retrieve the logged data
    #
    dataLr = mrpLog.torqueRequestBody
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    timeAxis = attErrorLog.times()

    plt.figure(5)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')
    figureList = {}
    pltName = fileName + "1" 
    figureList[pltName] = plt.figure(5)

    plt.figure(6)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataLr[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$L_{r,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Control Torque $L_r$ [Nm]')
    pltName = fileName + "2"
    figureList[pltName] = plt.figure(6)

    plt.figure(7)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error [rad/s] ')


    if showPlots:
        plt.show()

    plt.close("all")

    # Unload spice libraries
    gravFactory.unloadSpiceKernels()
    pyswice.unload_c(spiceObject.SPICEDataPath + 'de430.bsp')  # solar system bodies
    pyswice.unload_c(spiceObject.SPICEDataPath + 'naif0012.tls')  # leap second file
    pyswice.unload_c(spiceObject.SPICEDataPath + 'de-403-masses.tpc')  # solar system masses
    pyswice.unload_c(spiceObject.SPICEDataPath + 'pck00010.tpc')  # generic Planetary Constants Kernel


if __name__ == "__main__":

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) 
    datadir = os.path.join(datadir,'data')
    run(
        True,        # show_plots
        True,      # save pkl file
        True, # EarthAndMoonGrav
    )
