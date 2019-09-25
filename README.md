# find_my_pictures
Find pictures of a person within a set of pictures. 

### Requirements
This module requires [face_recognition](https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwi0qYLZuuTkAhVLQEEAHeeWBMAQFjAAegQIBhAC&url=https%3A%2F%2Fgithub.com%2Fageitgey%2Fface_recognition&usg=AOvVaw1ARIEn_v53-Z7n-ItnMaqz), dlib and cmake installed.

### Installing
```bash
cd ~
git clone https://github.com/tomasdembelli/find_my_pictures.git
python3 -m venv ~/find_my_pictures_env
source ~/find_my_pictures_env/bin/active
# make sure you are in the virtual environment before proceeding
cd ~/find_my_pictures
pip install -r requirements.txt
```

### Example
#### Step 1
```python
>>> import imp
>>> import os
>>> 
>>> home_dir = os.path.expanduser("~")
>>> fmp = imp.load_source('find_my_pictures', os.path.join(home_dir, 'find_my_pictures/find_my_pictures.py'))
>>> from find_my_pictures import FindMyPictures
>>> 
>>> session1 = FindMyPictures(verbose=True)    # Utility folders will be initialized if not given.
Images in ~/find_my_pictures/input_sample will be used for training.
Images in ~/find_my_pictures/input_stack will be analyzed to identify person of interest.
Images with positive matches will be stored in ~/find_my_pictures/output.
>>> 
```
This will create following folders if not given.
```
.
├── find_my_pictures.py
├── input_sample  
├── input_stack
└── output
```
#### Step 2
At this point:
- **input_sample**: Put the pictures of person of interest (poi) ONLY.
- Each picture should have only one face, which is, obviously, the face of the poi.
- Images with more than 1 face will be ignored.
- Handful of pictures should be enough for a good result.
- It is better if you provide pictures showing diffetent features of the face, such as with beard, with glasses, etc.

```python
>>> session1.train_poi(accuracy='medium')    # accuracy is optinal. Leaving blank will still give reasonable results.
# non  image files will be listed here
Analyzing the person of interest's face within sample images.
3 images will be analyzed.
Sample image ~/find_my_pictures/input_sample/IMG_20190901_150845.jpg (training data) can't contain more than 1 face.
Finished within 14.532799482345581 seconds.
>>> 
```
#### Step 3
At this point:
- **input_stack**:  Put the mixed pictures which you want to find the ones with poi in.
- **output**      : This will be used to store positive matched pictures. A unique folder will be created in it at each search.
- multiprocess  :   Optional. If not given, 1 processor will be used. Passing `half` could be ideal to be able to use your machine for other tasks while this module is searching for the positive matches. 
- copy          : If False, the positive matched image files will be moved from ```input_stack``` folder to ```output folder```. If True, they will be copied (duplicated).
```python
>>> session1.find_pictures(verbose=True, multiprocess='half', copy=False)    
# stats will be shown here
39 images will be analyzed.
obaba_diden.png  2 face detected
IMG-20190517-WA0001.jpg  2 face detected ** POSITIVE MATCH **
IMG_20190901_150845 (1).jpg      2 face detected ** POSITIVE MATCH **
...
Finished within 62.41517496109009 seconds.
Person of interest is recognised in 8 images.
Positive matches have been stored in ~/find_my_pictures/output/Positive_Match_2019_09_22_00-15.
```
#### Step 4
At this point:
- Check the results: 
```pash
cd ~/find_my_pictures/output
├── output
│   ├── Positive_Match_2019_09_21_23-22
│   │   ├── IMG-20190517-WA0001.jpg
│   │   ├── IMG_20190608_181730.jpg
│   │   ├── IMG-20190713-WA0067.jpg
│   │   └── IMG_20190901_150845 (1).jpg
│   └── Positive_Match_2019_09_22_00-15
```
#### To search for a new person within the same pictures:
- Replace all the images in the ```input_sample``` folder, and repeat the steps starting from **Step 2**.
#### To search for the same person within different set of images:
- Replace the image files in the ```input_stack``` folder, and repeat the steps starting from **Step 3**.

#### For more details:
```python
>>> help(FindMyPictures)
```

#### TODO
- Add a simple GUI.
- Rewrite the module in C++.
