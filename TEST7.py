import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import pyzed.sl as sl
import time
import math 

x=0
pt=0
pttt=0
ptttinT=[]
ptinT=[]
init_params=sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 15
zed=sl.Camera() 
status1=zed.open(init_params)
mat1=sl.Mat()
mat2=sl.Mat()			   
win_name2="live CAM"       # Right cam(according to calibration)    Left cam(according to calibration)
win_name="match Live Cam"          # fx=524.301 cx=672.224            fx=524.284  cx=680.445     
cv.namedWindow(win_name)	   # fy=523.312  cy=367.930            fy=524.740   cy=370.417
cv.namedWindow(win_name2)
Key=''
runtime=sl.RuntimeParameters()
sensors_data=sl.SensorsData()
K=np.array([[524.284,0,680.445],
			 [0,524.740,370.417],
			 [0,0,1]])
KR=np.array([[524.301,0,672.224 ],
			 [0,523.312 ,367.930],
			 [0,0,1]])
P=np.array([[524.301,0,672.224 ,0],[0,523.312,367.930,0],[0,0,1,0]])
PL=np.array([[524.284,0,680.445 ,0],
			 [0,524.740,370.417,0],
			 [0,0,1,0]])
PR=np.array([[524.301,0,672.224 ,0],
			 [0,523.312,367.930,0],
			 [0,0,1,0]])
baseline = -0.012  # -12 cm baseline (negative)
R_right = np.eye(3)
t_right = np.array([[baseline], [0], [0]], dtype=float)
PRR= np.dot(KR, np.hstack((R_right, t_right)))
Ident=np.identity(3)
timedif=0
sampling=0
sampling_minus1=0
x=0
Ciclos=0
g=9.81
Rt=np.array([[0.011,0,0],
		[0,0.011,0],
		[0,0,0.011]])
Qt2=np.array([[1,0,0],
		[0,1,0],
		[0,0,1]])
Qt=np.array([[0.001,0],
		[0,0.001]])
phi=0
theta=0
psi=0
theta_dot=0
phi_dot=0
psi_dot=0
p=0
q=0
r=0
ax=0
ay=0
az=0
Identity=np.array([[1,0],
			[0,1]])
Pt_minus1=np.array([[0.1,0,0],
			[0,0.1,0],
			[0,0,0.1]])
Pt2_minus11=np.array([[0.1,0],
			[0,0.1]])
class KalmanFilter:
	def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
		self.state = initial_state  # Initial state estimate
		self.covariance = initial_covariance  # Initial covariance matrix
		self.process_noise = process_noise  # Process noise covariance matrix
		self.measurement_noise = measurement_noise  # Measurement noise covariance matrix

	def predict(self, dt):
		# State transition model: constant angular velocity
		F = np.array([[1, dt],
					  [0, 1]])

		# Predict state and covariance
		self.state = np.dot(F, self.state)
		self.covariance = np.dot(np.dot(F, self.covariance), F.T) + self.process_noise

	def update(self, measurement):
		# Measurement model: directly measures angular position
		H = np.array([[1, 0], [1, 0]])  # Assuming both sensors directly measure angular position

		# Kalman gain calculation
		S = np.dot(np.dot(H, self.covariance), H.T) + self.measurement_noise
		K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))

		# Update state and covariance 
		self.state = self.state + np.dot(K, (measurement - np.dot(H, self.state)))
		self.covariance = np.dot((np.eye(len(self.state)) - np.dot(K, H)), self.covariance)
class DepthMap:
	def __init__(self,showImages):
		self.cccc=1
		
		if showImages:
			plt.figure()
			plt.subplot(121)
			plt.imshow(self.imgRight)
			plt.subplot(122)
			plt.imshow(self.imgLeft)
			plt.show()
	def computeDepthMapBM(self):
		nDispFactor=6
				 #to adjust
		stereo=cv.StereoBM.create(numDisparities=16*nDispFactor,blockSize=21)
		disparity=stereo.compute(self.imgLeft,self.imgRight)
		plt.imshow(disparity,'gray')
		plt.show()

	def computeDepthMapSGBM(self,imgLeftx,imgRightx,showImages):
		imgLeftx=cv.cvtColor(imgLeftx,cv.COLOR_BGR2GRAY)
		imgRightx=cv.cvtColor(imgRightx,cv.COLOR_BGR2GRAY)
		imgLeft=cv.resize(imgLeftx,(520, 520))
		imgRight=cv.resize(imgRightx,(520, 520))
		window_size=12
		Min_disp=4
		nDispFactor=10
		num_disp=16*nDispFactor-Min_disp
		if showImages:
			plt.figure()
			plt.subplot(121)
			plt.imshow(imgLeft)
			plt.subplot(122)
			plt.imshow(imgRight)
			plt.show()
		
		stereo=cv.StereoSGBM_create(minDisparity=Min_disp,numDisparities=num_disp,blockSize=window_size,P1=8*4*window_size**2,P2=32*4*window_size**2,
											disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2,preFilterCap=63,mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
		
		right_matcher = cv.ximgproc.createRightMatcher(stereo)

		disparity=stereo.compute(imgLeft,imgRight).astype(np.float32)/16
		
		rightD=right_matcher.compute(imgRight,imgLeft).astype(np.float32)/16


		lmbda=6000.0
		sigma=3

		wls=cv.ximgproc.createDisparityWLSFilter(stereo)

		wls.setLambda(lmbda)
		wls.setSigmaColor(sigma)

		filteredDis=wls.filter(disparity,imgLeft,disparity_map_right=rightD)

		#plt.imshow(filteredDis)
		#plt.colorbar()
		#plt.show()
		#plt.imshow(filteredDis)
		#depth=(2.12*120)/(filteredDis*0.002)
		#plt.imshow(disparity)
		#plt.colorbar()
		
		return filteredDis		
		
def ORBDiferent(cvImage,cvImage_minus1):
	#root=os.getcwd()
	#imgPath1=os.path.join(root,'test_ORBImages//bureau1.jpeg')#BMW1.png bureau1.jpeg saved.jpeg
	#imgPath2=os.path.join(root,'test_ORBImages//saved2.jpeg')#BMW2.png bureau2.jpeg saved2.jpeg
	imgGray1=cv.cvtColor(cvImage_minus1,cv.COLOR_BGR2GRAY)#cv.imread(imgPath1,cv.IMREAD_GRAYSCALE)
	imgGray2=cv.cvtColor(cvImage,cv.COLOR_BGR2GRAY)#cv.imread(imgPath2,cv.IMREAD_GRAYSCALE)
	imgGray1=cv.resize(imgGray1,(520, 520))
	imgGray2=cv.resize(imgGray2,(520, 520))

	orb=cv.ORB_create(5000)
	FLANN_INDEX_LSH=6	
	index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12,multi_probe_level=1)
	search_params=dict(checks=60)
	flann=cv.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)	
	
	keypoints1,descriptor1=orb.detectAndCompute(imgGray1,None)
	keypoints2,descriptor2=orb.detectAndCompute(imgGray2,None)
	matches=flann.knnMatch(descriptor1,descriptor2,k=2)
	goodMatches=[]
	print("______")
	print(len(matches))
	print("----------")
	for m,n in matches:
		if m.distance<0.8*n.distance:
			goodMatches.append(m)
	MinM=20	
	if len(goodMatches)>MinM:
		srcPts=np.float32([keypoints1[m.queryIdx].pt for m in goodMatches])#.reshape(-1,1,2)
		dstPts=np.float32([keypoints2[m.trainIdx].pt for m in goodMatches])#.reshape(-1,1,2)
		errorThreshold=6
		#print(dstPts)
		#print("_____________")
		E,maskE=cv.findEssentialMat(srcPts,dstPts,K,cv.RANSAC,0.999,errorThreshold)
		matchesMask=maskE.ravel().tolist()
		h,w=imgGray1.shape
		R1,R2,t=cv.decomposeEssentialMat(E)
		T1=TransformMatrix(R1,np.ndarray.flatten(t))
		T2=TransformMatrix(R2,np.ndarray.flatten(t))
		T3=TransformMatrix(R1,np.ndarray.flatten(-t))
		T4=TransformMatrix(R2,np.ndarray.flatten(-t))
		Transformations=[T1,T2,T3,T4]
		Kc=np.concatenate((K,np.zeros((3,1)) ),axis=1) 
		IdentC=np.concatenate((Ident,np.zeros((3,1)) ),axis=1)
		Projections=[Kc @ T1, Kc @ T2 , Kc @ T3  , Kc @ T4 ] #4x3
		Rs,ts,vs,_=cv.recoverPose(E,srcPts,dstPts)
		#print("recovered T")
		T0=np.concatenate((ts,vs),axis=1)
		T0=np.row_stack([T0,np.array([0,0,0,1])])
		#print(T0)
		#print("recovered T")
		positives=[]
		for Pc , T in zip(Projections,Transformations):
			HQ1=cv.triangulatePoints(P,Pc,srcPts.T,dstPts.T)
			HQ2=T @ HQ1
			Q1=HQ1[:3,:]/HQ1[3,:]
			Q2=HQ2[:3,:]/HQ2[3,:]
			total_sum=sum(Q2[2,:]>0)+sum(Q1[2,:]>0)
			relative_scale = np.mean(np.linalg.norm(Q1.T[:-1]-Q1.T[1:],axis=1)/np.linalg.norm(Q2.T[:-1]-Q2.T[1:],axis=1))
			positives.append(total_sum+relative_scale)
		#print(positives)
		max=np.argmax(positives)
		if(max==2):
			print("2")
			R=R1
			tt=-t
			print(R)
			print(tt)
		elif(max==3):
			print("3")
			R=R2
			tt=-t
			print(R)
			print(tt)
		elif(max==0):
			print("0")
			R=R1
			tt=t
			print(R)
			print(tt)
		elif(max==1):
			print("1")
			R=R2
			tt=t
			print(R)
			print(tt)
		DrawParams=dict(matchColor=-1,singlePointColor=None,matchesMask=matchesMask,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
		imgMatches=cv.drawMatches(imgGray1,keypoints1,imgGray2,keypoints2,goodMatches,None,**DrawParams) 
		TransformM=TransformMatrix(R,np.ndarray.flatten(tt))
		print("Enough matches",len(goodMatches))
	else: 
		print("Not enough matches")
		imgMatches=imgGray1	
	
	#plt.figure()
	#plt.imshow(imgMatches)
	#plt.show()
	return imgMatches,TransformM,srcPts,dstPts,goodMatches

def TransformMatrix(R,t):
	T=np.eye(4,dtype=np.float64)
	T[:3,:3]=R
	T[:3,3]=t
	return T
X=0
V=0
def TtoEuler(R):

	
	 # Roll (x-axis rotation)
	theta_x = np.arctan2(R[2, 1], R[2, 2])*180/np.pi
		# Pitch (y-axis rotation)
	theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))*180/np.pi
		# Yaw (z-axis rotation)
	theta_z = np.arctan2(R[1, 0], R[0, 0])*180/np.pi
	r,_=cv.Rodrigues(R)
	print('rodriguez')
	print(r*180/np.pi)
	print('rodriguez')
	print('mymethod')
	print([theta_x,theta_y,theta_z])
	print('mymethod')
	return theta_x, theta_y, theta_z
def Depth_for_points(srcPts,dstPts,disparity1,disparity2,goodMatches):
	left_y1, left_x1=srcPts[:,0],srcPts[:,1]
	left_y2, left_x2=dstPts[:,0],dstPts[:,1]
	disparityN1=[]
	disparityN2=[]
	depth_values1=[]
	depth_values2=[]

	for i in range(len(goodMatches)):
		#print(i)
		#print(left_x[i])
		disparity_value1 = disparity1[int(left_x1[i]), int(left_y1[i])]
		disparity_value2 = disparity2[int(left_x2[i]), int(left_y2[i])]
		depth1 =  0.12 * 529 / disparity_value1 #depth=(2.12*120)/(filteredDis*0.002)
		depth2 =  0.12 * 529 / disparity_value2 #depth=(2.12*120)/(filteredDis*0.002)
		disparityN1.append(disparity_value1)
		disparityN2.append(disparity_value2)

		depth_values1.append(depth1)
		depth_values2.append(depth2)
	return depth_values1,depth_values2,disparityN1,disparityN2
def delete_outliers(depth_values,Pts,disparityNs,max_depth,min_depth):
	left_y, left_x=Pts[:,0],Pts[:,1]
	rightx=left_x-disparityNs
	depth_valuesA= np.array(depth_values)/depth_values[3]
	
	DepthMASK1=np.where( depth_valuesA > max_depth)[0]
	
	
	depth_valuesA=np.delete(depth_valuesA,DepthMASK1)
	DepthMASK2=np.where( depth_valuesA < min_depth)[0]
	depth_valuesA=np.delete(depth_valuesA,DepthMASK2)
	NANMASK=np.isnan(depth_valuesA)
	left_xA=np.array(left_x)
	left_xA=np.delete(left_xA,DepthMASK1)
	left_xA=np.delete(left_xA,DepthMASK2)
	left_xA=left_xA[~NANMASK]
	left_yA=np.array(left_y)
	left_yA=np.delete(left_yA,DepthMASK1)
	left_yA=np.delete(left_yA,DepthMASK2)
	left_yA=left_yA[~NANMASK]
	rightxA=np.array(rightx)
	rightxA=np.delete(rightxA,DepthMASK1)
	rightxA=np.delete(rightxA,DepthMASK2)
	rightxA=rightxA[~NANMASK]
	left_x=left_xA.flatten().tolist()
	left_y=left_yA.flatten().tolist()
	rightx=rightxA.flatten().tolist()
	depth_valuesA= depth_valuesA[~NANMASK]
	depth_values=depth_valuesA.flatten().tolist()
	return depth_values,left_x,left_y,rightx
def delete_outliersD(Pts,disparityNs,max_disp,min_disp):
	left_y, left_x=Pts[:,0],Pts[:,1]
	rightx=left_x-disparityNs
	disparityNsA= np.array(disparityNs)
	
	DepthMASK1=np.where( disparityNsA > max_disp)[0]
	
	
	disparityNsA=np.delete(disparityNsA,DepthMASK1)
	DepthMASK2=np.where( disparityNsA < min_disp)[0]
	disparityNsA=np.delete(disparityNsA,DepthMASK2)
	NANMASK=np.isnan(disparityNsA)
	left_xA=np.array(left_x)
	left_xA=np.delete(left_xA,DepthMASK1)
	left_xA=np.delete(left_xA,DepthMASK2)
	left_xA=left_xA[~NANMASK]
	left_yA=np.array(left_y)
	left_yA=np.delete(left_yA,DepthMASK1)
	left_yA=np.delete(left_yA,DepthMASK2)
	left_yA=left_yA[~NANMASK]
	rightxA=np.array(rightx)
	rightxA=np.delete(rightxA,DepthMASK1)
	rightxA=np.delete(rightxA,DepthMASK2)
	rightxA=rightxA[~NANMASK]
	left_x=left_xA.flatten().tolist()
	left_y=left_yA.flatten().tolist()
	rightx=rightxA.flatten().tolist()
	disparityNsA= disparityNsA[~NANMASK]
	disparityNs=disparityNsA.flatten().tolist()
	return disparityNs,left_x,left_y,rightx
def triangulate(q1r,q1l,q2r,q2l):
	#sssss
	Q1=cv.triangulatePoints(PL,PRR,q1l.T,q1r.T)
	Q2=cv.triangulatePoints(PL,PRR,q2l.T,q2r.T)
	Q1 = np.transpose(Q1[:3] / Q1[3])
	Q2 = np.transpose(Q2[:3] / Q2[3])
	return Q1,Q2
def PredictionV(Xkm,Uk,Dt,Pkm,Qk):
	Xk=Xkm+0.1*(Xkm+Uk*Dt)
	print('Xk',Xk)
	Pk=Pkm+Qk
	return Xk,Pk

def upgradeV(Zk,Xk,Pk,Rk):
	Zmedia=Zk-Xk
	Sk=Pk+Rk
	Kk=Pk/Sk
	Xk=Xk+Kk*Zmedia
	Pk=(1-Kk)*Pk
	return Xk,Pk


if __name__=='__main__':
	max_depth=6.3
	min_depth=0.2
	min_disp=13
	max_disp=50
	Aprox_Vel=0
	Aprox_VelN=0
	Aprox_VelIMU=0
	Co=0
	magT=[]
	locT=[]
	mag_heading2=0
	AccinT=[]
	Aprox_VelinT=[]
	Aprox_VelIMUinT=[]
	initial_state = np.array([[0], [0]])  # Initial state: [angular position, angular velocity]
	initial_covariance = np.eye(2)  # Initial covariance matrix Pk
	
	process_noise = np.diag([0.5, 0.5])  # Process noise covariance matrix Qk
	
	measurement_noise = np.diag([0.6, 0.6])  # Measurement noise covariance matrix Rk
	
	kalman_filter = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)
	
	Pk=0.5
	Qk=0.9
	Rk=0.6

	Px=0
	Py=0
	VxinT=[]
	VyinT=[]
	PxinT=[]
	PyinT=[]
	theta_xinT=[]
	theta_yinT=[]
	theta_zinT=[]
	root=os.getcwd()
	start_pose=np.ones((3,4))
	start_translation=np.zeros((3,1))
	start_rotation=np.identity(3)
	start_pose=np.concatenate((start_rotation,start_translation),axis=1)
	if status1 != sl.ERROR_CODE.SUCCESS:
		print("Camera Open : "+repr(status1)+". Exit program.")
		exit()
	TT=mat1.timestamp.get_microseconds()
	print("TT")
	print(TT)
	timepassed=time.time()
	seconds_passed=time.time()
	alphaV=0.9
	alphaA=0.9
	alphaVL=0.92
	Update_steps=0
	IMU=np.array([0,0])
	IMU2=np.array([0,0,0])
	IMU_dot=np.array([0,0,0])
	A_velocityN=np.array([0,0,0])
	AcceleretionN=np.array([0,0,0])
	loc_headingN=0
	theta_xN=0
	theta_yN=0
	theta_zN=0
	NotF_in_time=[]
	F_in_time=[]
	TIMEforG=[]
	Pitch_inTIME=[]
	Yaw_inTIME=[]
	TIMEforP=[]
	RAWG=np.array([0,0,0])
	RAWGinT=[]
	SecKFinT=[]
	ThirdKFinT=[]
	dp=DepthMap(showImages=False)
	if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
		zed.retrieve_image(mat1,sl.VIEW.LEFT)
		cv.imwrite('images//savedorbT1_l.png',mat1.get_data())
		zed.retrieve_image(mat2,sl.VIEW.RIGHT)
		cv.imwrite('images//savedorbT1_R.png',mat2.get_data())
		imgPath2l=os.path.join(root,'images//savedorbT1_l.png')
		imgPath2r=os.path.join(root,'images//savedorbT1_R.png')
		cvImage_minus1_L=cv.imread(imgPath2l)
		cvImage_minus1_R=cv.imread(imgPath2r)
		disparity_m1=dp.computeDepthMapSGBM(cvImage_minus1_L,cvImage_minus1_R,showImages=False)
		print('punto__________')
	TM=start_pose
	seconds=time.time()
	seconds2=time.time()
	while Key!=113:
		Tat=math.tan(theta)
		Cp=math.cos(phi)
		Ct=math.cos(theta)
		Sp=math.sin(phi)
		St=math.sin(theta)
		err=zed.grab(runtime)
		Fdxm=np.array([[1,Sp*Tat,Cp*Tat],
		 [0,Cp,-Sp],
		 [0,(Sp/Ct),(Cp/Ct)]])
		Fdxm2=np.array(
		[[1,Sp*Tat,Cp*Tat],
		 [0,Cp,-Sp]])
		Hdxm=np.array(
		[g*St,-g*Ct*Sp,-g*Ct*Cp])
		A=np.array([[q*Cp*Tat-r*Sp*Tat,r*Cp*(Tat*Tat+1)+q*Sp*(Tat*Tat+1)],[-r*Cp-q*Sp,0]])
		C=np.array([[0,g*Ct],[-g*Cp*Ct,g*Sp*St],[g*Sp*Ct,g*Cp*St]])
		sampling_minus1=time.time()
		

		if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
			
			zed.retrieve_image(mat2,sl.VIEW.RIGHT)
			cvImage_R=mat2.get_data()
			zed.retrieve_image(mat1,sl.VIEW.LEFT)
			cvImage_L=mat1.get_data()
			timePassed3=time.time()-seconds
			if err == sl.ERROR_CODE.SUCCESS:
				zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
				#zed.retrieve_image(mat1,sl.VIEW.RIGHT)
				if mat1.timestamp.get_microseconds() > TT:

					timedif=time.time()-seconds_passed
					imu_data=sensors_data.get_imu_data()
					linear_acceleration=imu_data.get_linear_acceleration()
					angular_velocity=imu_data.get_angular_velocity()
					magnetometer_data = sensors_data.get_magnetometer_data()
					magnetic_field_dat=magnetometer_data.get_magnetic_field_calibrated()
					mag_heading=np.arctan2(magnetic_field_dat[0],magnetic_field_dat[2])
					mag_heading=mag_heading*180/np.pi
					if mag_heading<0:
						mag_heading=mag_heading+360
					if Co<2:
						#mag_heading2=np.arctan(magnetic_field_dat[0]/magnetic_field_dat[2])*180/np.pi
						mag_heading2=np.arctan2(magnetic_field_dat[0],magnetic_field_dat[2])
						mag_heading2=mag_heading2*180/np.pi
						if mag_heading2<0:
							mag_heading2=mag_heading2+360
						Co+=1
					loc_heading=mag_heading-mag_heading2
					loc_headingN=0.7*loc_headingN+(1-0.7)*loc_heading
					p=-math.radians(angular_velocity[2])
					q=-math.radians(angular_velocity[0]) #remapped
					r=math.radians(angular_velocity[1])
					A_velocity=np.array([p,q,r])
					A_velocityN=alphaV*A_velocityN+(1-alphaV)*A_velocity
					ax=-linear_acceleration[2]
					ay=-linear_acceleration[0]  #remapped
					az=linear_acceleration[1]
					Acceleretion=np.array([ax,ay,az])
					AcceleretionN=alphaA*AcceleretionN+(1-alphaA)*Acceleretion#predict stage
					IMU=IMU+0.03*np.dot(Fdxm2,A_velocityN)
					phi=IMU[0]
					theta=IMU[1]
					Pt2_minus11=Pt2_minus11+0.03*(np.dot(A,Pt2_minus11)+np.dot(Pt2_minus11,np.transpose(A))+Qt)
					x=x+1
					Ciclos=x/timedif
					TT=mat1.timestamp.get_microseconds()	
					#print(f'Timestmap:{TT}')
					#print(f'TIMESTEP:{x}')
					#print(f'Diferencial de tiempo (tiempo pasado desde inicio):{timedif}')		
					print(" \t raw not mapped Acceleration: [ {0} {1} {2} ] [m/sec^2]".format(linear_acceleration[0],linear_acceleration[1],linear_acceleration[2]))
					#print(f'ciclos (Hz):{Ciclos}')
					#print(" \t Filtered Acceleration: [ {0} {1} {2} ] [m/sec^2]".format(AcceleretionN[0],AcceleretionN[1],AcceleretionN[2]))
					print(" \t raw Angular Velocities: [ {0} {1} {2} ] [rad/sec]".format(p,q,r))
					NotF_in_time.append([p,q,r])
					F_in_time.append(A_velocityN)
					TIMEforG.append(timedif)

					#print(" \t Angular Velocities F:",A_velocityN)
					sampling=time.time()-sampling_minus1
					print(f'TIMESampling:{sampling}')
					Time_reference=time.time()-timepassed 
					if Time_reference > 0.1:
						X=X+1
						cv.imwrite('images//savedorbT_m1_l.png', cvImage_minus1_L)
						cv.imwrite('images//savedorbT_m1_R.png', cvImage_minus1_R)
						cv.imwrite('images//savedorbT1_l.png', cvImage_L)
						cv.imwrite('images//savedorbT1_R.png', cvImage_R)
						disparityN=dp.computeDepthMapSGBM(cvImage_L,cvImage_R,showImages=False)
						#print('punto2__________')
						frame,TMn,srcPts,dstPts,goodMatches=ORBDiferent(cvImage_L,cvImage_minus1_L)
						
						seconds=time.time()
						V=1
						#print(timePassed)
						print(X)
						#---------------------------------------------Kalman Filter

						Update_steps+=1
						print(f'update:{Update_steps}')
						Kinvers=np.linalg.multi_dot([C,Pt2_minus11,np.transpose(C)])+Rt #invers matrix calculation 
						Kinvers=np.linalg.inv(Kinvers)
						Kk=np.linalg.multi_dot([Pt2_minus11,np.transpose(C),Kinvers])
						IMU=np.dot(Kk,AcceleretionN-Hdxm)+IMU
						IDOT=(Identity-np.dot(Kk,C))
						Pt2_minus11=np.dot(IDOT,Pt2_minus11)
						print(" \t Predicted angle:",IMU)
						print(" \t Y-H:",AcceleretionN-Hdxm)
						print(" \t dot:",np.dot(Kk,AcceleretionN-Hdxm))
						print(" \t Pitch and yaw: [ {0} {1} ] [deg]".format(math.degrees(IMU[0]),math.degrees(IMU[1])))
						print(" \t P:",Pt2_minus11)
						timepassed1=time.time()-seconds_passed 
						timepassed=time.time()
						Pitch_inTIME.append(math.degrees(IMU[0]))
						Yaw_inTIME.append(math.degrees(IMU[1]))
						TIMEforP.append(timepassed1)
						XxX=np.array([p,q,r])
						RAWG=RAWG+XxX*0.3*180/np.pi
						RAWGinT.append(RAWG)
						print(RAWG)
						##############################
						TM=TM @ TMn
						R_n=TM[:3, :3]
						theta_x, theta_y, theta_z=TtoEuler(R_n)
						theta_xN=0.3*theta_xN+(1-0.3)*theta_x
						theta_yN=0.3*theta_yN+(1-0.3)*theta_y   #loc_headingN=0.7*loc_headingN+(1-0.7)*loc_heading                      
						theta_zN=0.3*theta_zN+(1-0.3)*theta_z
						theta_xinT.append(theta_xN)
						theta_yinT.append(theta_yN)
						theta_zinT.append(theta_zN)
						magT.append(mag_heading)
						locT.append(loc_headingN)
						kalman_filter.predict(0.3)
						kalman_filter.update(np.array([[theta_x], [math.degrees(IMU[1])]])) 
						SecKF=kalman_filter.state[0, 0]
						kalman_filter.predict(0.3)
						kalman_filter.update(np.array([[theta_y], [loc_headingN]]))
						ThirdKF=kalman_filter.state[0, 0]
						depth_values1,depth_values2,disparityN1,disparityN2=Depth_for_points(srcPts,dstPts,disparity_m1,disparityN,goodMatches)
						
						
						ThirdKFinT.append(ThirdKF)
						SecKFinT.append(SecKF)
						######################################
						#print(A_velocity)
					if V==1:
						frame=cv.resize(frame,(500,750))
						cv.imshow(win_name, frame)
						V=0

					
						depth_values1,left_x1,left_y1,rightx1=delete_outliers(depth_values1,srcPts,disparityN1,max_depth,min_depth)
						depth_values2,left_x2,left_y2,rightx2=delete_outliers(depth_values2,dstPts,disparityN2,max_depth,min_depth)

						disparityN1,left_x11,left_y11,rightx11=delete_outliersD(srcPts,disparityN1,max_disp,min_disp)
						disparityN2,left_x22,left_y22,rightx22=delete_outliersD(dstPts,disparityN2,max_disp,min_disp)


						q1_l=np.column_stack((left_x11, left_y11))
						q1_r=np.column_stack((rightx11,left_y11))
						q2_l=np.column_stack((left_x22,left_y22))
						q2_r=np.column_stack((rightx22,left_y22))

						Q1,Q2=triangulate(q1_r,q1_l,q2_r,q2_l)


						#sample_idx = np.random.choice(len(depth_values1), 6)
						#d1=np.array(depth_values1)
						#d2=np.array(depth_values2)
						Aprox_Vel=(np.mean(Q2[:,2])-np.mean(Q1[:,2]))/0.1
						print(Aprox_Vel)
						Aprox_VelN=alphaVL*Aprox_VelN+(1-alphaVL)*Aprox_Vel
						Aprox_VelinT.append(Aprox_VelN)

						disparity_m1=disparityN

						Pkm=Pk
						print('PKM:',Pkm)
						Dt=0.1
						AccinT.append(AcceleretionN[0])
						Xk,Pk=PredictionV(Aprox_VelIMU,AcceleretionN[0],Dt,Pkm,Qk)    ###MODIFICAR VARIABLES
						if X>1:
							Xk,Pk=upgradeV(Aprox_VelN,Xk,Pk,Rk)
							X=0
							Aprox_VelIMU=Xk
						Vy=Aprox_VelIMU*math.cos(math.radians(loc_headingN))
						Vx=Aprox_VelIMU*math.sin(math.radians(loc_headingN))
						if abs(A_velocityN[2])>0.01:
							Vx=0
							Vy=0
						VxinT.append(Vx)
						VyinT.append(Vy)
						Px=(Px+Vx*Dt)
						Py=(Py+Vy*Dt)
						PxinT.append(Px)
						PyinT.append(Py)
						Aprox_VelIMUinT.append(Aprox_VelIMU)

					

						

					elif(V==0):
						#print(timePassed)
						imgPath2r=os.path.join(root,'images//savedorbT1_r.png')
						imgPath2l=os.path.join(root,'images//savedorbT1_l.png')
						cvImage_minus1_L=cv.imread(imgPath2l)
						cvImage_minus1_R=cv.imread(imgPath2r)
						
						
				
		else:
			print("Error during capture:", err)
			break
		cv.putText(cvImage_L, str(np.round(TM[0, 0],2)), (260,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[0, 1],2)), (340,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[0, 2],2)), (420,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 0],2)), (260,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 1],2)), (340,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 2],2)), (420,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 0],2)), (260,130), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 1],2)), (340,130), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 2],2)), (420,130), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[0, 3],2)), (540,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 3],2)), (540,90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 3],2)), (540,130), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.putText(cvImage_L, str(np.round(Aprox_VelN,2)), (640,230), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.imshow(win_name2, cvImage_L)
		Key=cv.waitKey(5)  
	cv.destroyAllWindows()
	zed.close()
	NotF_in_time=np.array(NotF_in_time)
	F_in_time=np.array(F_in_time)
	TIMEforG=np.array(TIMEforG)
	RAWGinT=np.array(RAWGinT)
	Yy1=NotF_in_time[:,1]
	Yy2=F_in_time[:,1]

	# filtered Vs Not filtered

	plt.figure(1)
	plt.plot(TIMEforP,Yaw_inTIME,'r',label='filtered')
	plt.plot(TIMEforP,RAWGinT[:,1],'b',label='not filtered')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show()

	plt.figure(2)
	plt.plot(TIMEforP,magT,'r',label='mag_heading')
	plt.plot(TIMEforP,locT,'b',label='Loc_heading')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show()

#Visual Vs Not Visual
	plt.figure(3)
	plt.plot(TIMEforP,theta_xinT,'r',label='V odometry') #Pitch_inTIME Yaw_inTIME
	plt.plot(TIMEforP,Yaw_inTIME,'b',label='filtered')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show()
	print(len(TIMEforP))
	
	plt.figure(4)
	plt.plot(TIMEforP,theta_xinT,'r',label='V odometry')
	plt.plot(TIMEforP,SecKFinT,'b',label='second Kalman')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show() 

	plt.figure(5)
	plt.plot(TIMEforP,theta_yinT,'r',label='VO heading')
	plt.plot(TIMEforP,locT,'b',label='Loc_heading')
	plt.plot(TIMEforP,ThirdKFinT,'g',label='filtered heading')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.legend(prop={'size': 15})
	plt.show() 
	#Linear velocity 
	plt.figure(6) 
	plt.plot(TIMEforP,Aprox_VelinT,'r',label='V linear odometry')
	plt.plot(TIMEforP,Aprox_VelIMUinT,'b',label='V linear')
	plt.title("Velocity vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("M/s")
	plt.legend(prop={'size': 15})
	plt.show() 
	
	#plt.figure(5) 
	#plt.plot(TIMEforP,AccinT,'r',label='Acceleration')
	#plt.title("acceleration vs Time")
	#plt.xlabel("Time(s)")
	#plt.ylabel("M/s/s")
	#plt.legend(prop={'size': 15})
	
	#linear pose 

	plt.figure(7)
	plt.plot(PxinT,PyinT,'r',label='Pose')
	plt.title("Position X vs Position Y")
	plt.xlabel("Position X")
	plt.ylabel("Position Y")
	plt.legend(prop={'size': 15})
	plt.show() 
 
