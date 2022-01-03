
cd /preprocess/Disfa

# step1: 
(1) cp DISFA_DATASET_ROOT/Video_RightCamera.zip ./
(2) cp DISFA_DATASET_ROOT/ActionUnit_Labels.zip ./

# step2: 
(1) unzip Video_RightCamera.zip -d ./Videos
(2) unzip ActionUnit_Labels.zip -d ./Labels

# step3: python get_faces.py 
Noted that the OpenFace toolkit dir of "executable" must be replaced with yours.

# step4: python gene_dataset.py

