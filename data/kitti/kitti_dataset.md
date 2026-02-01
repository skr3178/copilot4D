full review of the KITTI odometry dataset and how to access it with the devkit. 
                                                                                           
  ---                                                                                      
  Dataset Layout                                                                           
                                                                                           
  data/kitti/                                                                              
  ├── dataset/                                                                             
  │   ├── sequences/          # 22 sequences (00-21)                                       
  │   │   └── 00/                                                                          
  │   │       ├── velodyne/   # LiDAR scans: 000000.bin ... (4541 scans for seq 00)        
  │   │       ├── calib.txt   # Calibration matrices (P0-P3 projections + Tr velo-to-cam)  
  │   │       └── times.txt   # Timestamps in seconds                                      
  │   └── poses/              # Ground truth for sequences 00-10 only                      
  │       ├── 00.txt          # N x 12 values per line (3x4 T_w_cam0 flattened row-major)  
  │       └── ...                                                                          
  ├── devkit/                 # Official KITTI C++ evaluation code                         
  ├── kitti-devkit-odom/      # Extended devkit with ground truth + C++ eval               
  └── pykitti/                # Python helper library (the one you want)                   
                                                                                           
  Key Data Formats                                                                         
                                                                                           
  Velodyne scans (.bin): Raw binary, Nx4 float32 [x, y, z, reflectance] in meters (Velodyne
   frame). Read by determining point count from file size (file_size / (4 *                
  sizeof(float))).                                                                         
                                                                                           
  Poses (xx.txt): Each line is 12 floats = a 3x4 transformation matrix T_w_cam0            
  (world-from-cam0) stored row-major. Available only for sequences 00-10. Sequences 11-21  
  are the test/evaluation set.                                                             
                                                                                           
  Calibration (calib.txt):                                                                 
  - P0-P3: 3x4 camera projection matrices                                                  
  - Tr: 3x4 Velodyne-to-cam0 transform. To project a point X from velodyne to image i: x = 
  Pi * Tr * X                                                                              
                                                                                           
  Timestamps (times.txt): One float per line, in seconds (~0.1s between frames).           
                                                                                           
  Accessing with pykitti                                                                   
                                                                                           
  The pykitti library at data/kitti/pykitti/ provides the cleanest API. The key class is   
  pykitti.odometry in pykitti/pykitti/odometry.py.                                         
                                                                                           
  Basic Usage                                                                              
                                                                                           
  import sys                                                                               
  sys.path.insert(0, '/media/skr/storage/self_driving/CoPilot4D/data/kitti/pykitti')       
  import pykitti                                                                           
  import numpy as np                                                                       
                                                                                           
  basedir = '/media/skr/storage/self_driving/CoPilot4D/data/kitti/dataset'                 
  sequence = '00'                                                                          
                                                                                           
  # Load full sequence (or subset with frames=range(0, 100))                               
  dataset = pykitti.odometry(basedir, sequence)                                            
  # dataset = pykitti.odometry(basedir, sequence, frames=range(0, 20, 5))                  
                                                                                           
  print(len(dataset))  # number of frames                                                  
                                                                                           
  What's Available                                                                         
  Attribute: dataset.poses                                                                 
  Type: list of 4x4 np arrays                                                              
  Description: Ground truth T_w_cam0 (seq 00-10 only)                                      
  ────────────────────────────────────────                                                 
  Attribute: dataset.timestamps                                                            
  Type: list of timedelta                                                                  
  Description: Parsed from times.txt                                                       
  ────────────────────────────────────────                                                 
  Attribute: dataset.calib                                                                 
  Type: namedtuple                                                                         
  Description: Calibration data (see below)                                                
  ────────────────────────────────────────                                                 
  Attribute: dataset.velo                                                                  
  Type: generator                                                                          
  Description: Yields Nx4 arrays [x,y,z,reflectance]                                       
  ────────────────────────────────────────                                                 
  Attribute: dataset.get_velo(idx)                                                         
  Type: Nx4 np array                                                                       
  Description: Random access to a single scan                                              
  ────────────────────────────────────────                                                 
  Attribute: dataset.cam0 / dataset.cam1                                                   
  Type: generator                                                                          
  Description: Grayscale left/right images                                                 
  ────────────────────────────────────────                                                 
  Attribute: dataset.cam2 / dataset.cam3                                                   
  Type: generator                                                                          
  Description: RGB left/right images                                                       
  ────────────────────────────────────────                                                 
  Attribute: dataset.get_cam0(idx) etc.                                                    
  Type: PIL Image                                                                          
  Description: Random access to images                                                     
  Calibration Namedtuple                                                                   
                                                                                           
  dataset.calib.T_cam0_velo  # 4x4 Velodyne -> cam0 transform                              
  dataset.calib.T_cam1_velo  # 4x4 Velodyne -> cam1 transform                              
  dataset.calib.K_cam0       # 3x3 camera intrinsics for cam0                              
  dataset.calib.b_gray       # gray stereo baseline in meters                              
  dataset.calib.b_rgb        # RGB stereo baseline in meters                               
  dataset.calib.P_rect_00    # 3x4 projection matrix                                       
                                                                                           
  Example: Load velodyne + poses (what CoPilot4D needs)                                    
                                                                                           
  # Get a single velodyne scan                                                             
  scan = dataset.get_velo(0)        # Nx4 [x, y, z, reflectance]                           
  points_xyz = scan[:, :3]          # Nx3                                                  
                                                                                           
  # Get the ego pose for that frame                                                        
  T_w_cam0 = dataset.poses[0]      # 4x4 world-from-cam0                                   
                                                                                           
  # Transform velodyne points to cam0 frame                                                
  T_cam0_velo = dataset.calib.T_cam0_velo  # 4x4                                           
  points_hom = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])                      
  points_cam0 = (T_cam0_velo @ points_hom.T).T[:, :3]                                      
                                                                                           
  # Compute relative pose between frame i and j (action input)                             
  T_i = dataset.poses[i]                                                                   
  T_j = dataset.poses[j]                                                                   
  T_rel = np.linalg.inv(T_i) @ T_j  # SE(3) relative motion                                
                                                                                           
  Important Notes                                                                          
                                                                                           
  1. Only velodyne data is present in your sequences -- no image_0/image_1/image_2/image_3 
  folders. The cam0-cam3 generators will return empty. You have the LiDAR point clouds,    
  which is what the CoPilot4D paper uses.                                                  
  2. Ground truth poses exist for sequences 00-10 only. Sequences 11-21 are the test set   
  with no public poses.                                                                    
  3. Egomotion-compensated scans: The velodyne data has already been compensated for       
  ego-motion (unlike KITTI raw data), so each scan is consistent with the camera trigger   
  time.                                                                                    
  4. You can also install pykitti as a proper package: cd data/kitti/pykitti && pip install
   -e . to avoid the sys.path hack.   