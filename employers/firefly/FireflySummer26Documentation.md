$\gdef\vect#1{\mathbf{#1}}$
$\gdef\mat#1{\mathbf{#1}}$
$\gdef\rBM{\vect{r}^M_{B/M}}$
$\gdef\vBM{\vect{v}^M_{B/M}}$
$\gdef\qBM{\hat{\Bar{\mathbf{q}}}^B_{M}}$

$\gdef\rhatRsoWrtCamInI{\hat{\vect{r}}^I_{RSO/C}}$

$\gdef\qIB{\hat{\bar{\mathbf{q}}}^I_{B}}$

$\gdef\qImageGenI{\hat{\bar{\mathbf{q}}}^{ImageGen}_{I}}$

<!-- Here is an inline vector: $\rBM$ and a matrix: $\mat{A}$. -->
# Documentation of my summer 2026 internship at Firefly Aerospace

Team: Spacecraft GNC - Elytra \
Members: Joshua Wheeler, John Beavers, Will MacCormack \
Manager: Forrest Ward 

## Code bases / software architecture
**General Configuration:**
* Sim: NASA-Trick Engine + JEOD modeling utils \
    A set of .yml files of .py input files that configure the simulation by 
* Fsw: NASA Fprime
* Vehicle Ground Comms: F Prime Ground Data System (GDS)
* Analysis: Trick Verification / Criterion Configuration

**Repositories:**
* trick-sim
* sensor / actuator models
* fsw component / topology 
* fsw algorithms
* common data (.ymls)

## Mission Specifics
[Elytra DIU Mission](https://fireflyspace.com/missions/elytra-diu-sinequone/)

* Mission Type: Responsive In-Space Demonstration
* Customer: Defense Innovation Unit (DIU)
* Spacecraft: Elytra Orbital Vehicle
* Payloads: U.S. Government 

<u>Payloads:</u>
Elytra will host a suite of U.S. government payloads, including optical visible and infrared cameras and a universal electrical bus with a payload interface module.

<u>Vehicle:</u>
Firefly’s Elytra Dawn configuration will utilize common components from the company’s launch vehicles and lunar landers, including the avionics, composite structures, and propulsion systems, to enable on-demand mobility, plane changes, and maneuvers with high delta-V capabilities and reliability. Elytra’s main engine, called Spectre, was flight proven on Blue Ghost Mission 1 as the reaction control system thrusters that successfully performed Firefly’s final descent on the Moon.

<u>Mission Summary:</u>
Firefly was awarded a contract from the U.S. DoW’s Defense Innovation Unit (DIU) to perform a responsive in-space mission with its Elytra orbital vehicle. During the mission, Elytra will serve as a space maneuver vehicle to perform a series of responsive tasks, including space domain awareness operations in low Earth orbit. This award supports DIU’s Sinequone Project as the first step to enable future access and operations in xGEO on a responsive timeline.

# Tasks

## SensorQuality Class implementation
**Problem:** 
* Our FSW should be robust to sensor anomalies but we have no way to simulate sensor anamolies in our simulation.
**Action:** 
* Implement a Sensor Quality class to accelerometer, gyroscope, Position Velocity Timing sensor, magnetometer, reaction wheel momentum telemetry, sun sensor, and star tracker.
* This class enabled NOMINAL, DEGRADED, NONSENSE, STALE, and RANDOMIZED degradation modes that acted on the sensor model's ouput measurement.


**Result:** 
* This functionality was used in our Monarch 0.1 FSW release to verify that our FSW was able to detect stale and nonsensical measurements.
* I familiarized myself with the TRICK / JEOD simulation framework.


## Increased RSO
Sperhical SRP force model, Cannon ball aerodynamic drag, point mass Sun and Moon third body gravity effects
**Problem:** 

**Action:** 

**Result:** 

## Attitude Initialization Tool
Wrapped TRIAD algorithm in C++ `InitializeAttitudeConstrainedAlign` class.
Configure in .yml utilizing knowledge of TRICK sim variables. \
**Problem:** 

**Action:** 

**Result:** 



## Simple Attitude Determination
Propagate update algorithm. Propagate quaternion kinematics with angular rate source. Brute force overwrite attitude estimate with updated input.
**Problem:** 

**Action:** 

**Result:** 



## Optical Payload GNC Preparations
* <u>Target Track CONOP:</u> Open loop slew attitude that rotates camera boresight to relative position vector from ephemeris data. Then close the loop with AON measurements from wide FOV camera.
* Payload that produces an Angles only navigation (AON) measurement. A unit vector of the position of the RSO wrt the camera coordinatized in the inertial frame ($\rhatRsoWrtCamInI$). Closes the loop on RSO attitude tracking.
* Therefore, the payload must be computing the body to inertial frame rotation $\qIB$. Let's process this in our atttiude estimation filter!

The payload producing AON measurements has it's own hardware therfore the need to test the AON measurement generation and processing in-the-loop on the FlatSat is extremely desireable.

**Goal:** Test the AON measurement generation and processing in-the-loop on the FlatSat.


### Optical Processing Delay Simulation Modeling
**Problem:** 
* Optical payload has ~0.5s of latency from image epoch to AON reception epoch. 

**Action:** 
* Integrate Delay model in measurement model.

**Result:** 
* Increased simulation fidelity.


### Synthetic Starfield Generation
**Brief**: Export Data Attitude Interpretation / Creation of Image Generation Frame \
**Problem:** 
* Starfield emulators desire "ra and dec of boresight from Intertial, roll about boresight" attitude information.
* We are currently only modeling our "camera" as a body frame boresight direction
* The necessity to define a "camera frame" has arrised.

**Action:** 
* I created the $\qImageGenI$ frame with an associated structure that holds synthetic starfield emulation ra, dec, and roll values, which were obtained from computing the 3-2-1 Euler angle sequence through angles $\psi$ (psi), $\theta$ (theta), and $\phi$ (phi).\
The final orientation values are mapped as follows:
    * ra=$\psi$, dec = -$\theta$, roll = $\phi$


**Result:** 
* I sucessfully interpreted the requested attitude information, created a ImageGen frame to easily extract the requested angles, and added this to the camera model so that we can easily log and export true camera orientation for future image generation needs.


### Synthetic Starfield Generation HIL in-the-loop

**Problem:** 
* We need to use our TRICK HWIL simulation to generate synthetic images in real time.
* Our internal synthetic starfield emulation software inputs two epochs of information: the beginning and end of camera exposure. 

**Action:** 
* Note the attitude informatino is now easily acessible and contained in the AON model.
* Developed a Starfield Emulator ICD class that contains a pointer to a AON model, handles the beginning end of exposure data aquisitions, and pushes this information from TRICK binary to the starfield emulation software client.

**Result:** 
* `--starfield_emulator_icd_active` simulation flag that updates the ICD class to extract data at the beginning and end of exposure epochs.
* `--starfield_emulator_socket_active` simulation flag that automatically establishes parent - client socket, pushes beginning and end of exposure epoch data to client, and generates synthetic starfield images in real time (~60ms from client information reception to output image).
* Testing culminated in successful -RT = 1. SIL with starfield emulation running locally. 


### Automated Synthetic Starfield Generation SIL & TLM Output
**Problem:** 
* The to payload simulation data extraction was extremely manual from our simulation products. We had many troubles of data containing duplicates, not being logged at the same rate, and time tagging confusion. 

**Action:** 
* I utilized the Starfield Emulator ICD class to log neccesary starfield generation data at the correct simulation epochs.
* Created a Python script to automatically generate Starfield images from a simulation run data directory for certain simulation windows.
* Created a automated script that took in a simulation run data directory and a viewing window .csv to export "to payload" FSW information and generated sim_time tagged sythetic images from the Starfield Emulator ICD class.


**Result:** 
*  I turned an extremely manual data collection process into an extremely reliable automated process.
* The simulation scenario, FSW outputs, image viewing windows, and sythnetic images are all coupled and unique to our . The automated process unsures minimal opportunities for misalignment of these critical pieces of information that help verify payload algorithms.