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
#### Step 1: Importing the module and creating the necessary folders if they are not given.
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
This will create the following folders if not given.
```
.
├── find_my_pictures.py
├── input_sample  
├── input_stack
└── output
```
#### Step 2: Encoding the known person's face.
- **input_sample**: Put the pictures of person of interest (poi) ONLY in this folder.
- Each picture should have only one face, which is, obviously, the face of the poi.
- Images with more than 1 face will be ignored.
- Handful of pictures should be enough for a good result.
- It is better if you provide pictures showing diffetent features of the face, such as with beard, with glasses, etc.
- accuracy: Leaving blank should still give good results. `medium` and `high` might give better results at positive matching but the process of poi face encoding will take longer.
- poi_identifier: (optional) This string will be used to name the folder with positively matched pictures. If not given, a random name will be generated.

```python
>>> session1.encode_poi(accuracy='medium')    # accuracy is optinal. Leaving blank will still give reasonable results.
# non  image files will be listed here
Analyzing the person of interest's face within sample images.
3 images will be analyzed.
Sample image ~/find_my_pictures/input_sample/IMG_20190901_150845.jpg (training data) can't contain more than 1 face.
Finished within 14.532799482345581 seconds.
>>> 
```
#### Step 3: Finding the known person's pictures within mixed images.
- **input_stack**: Put the mixed pictures in this folder, or pass the absolute path for the folder containing the mixed images to `folder` argument.
- `multiprocess`: Optional. If not given, 1 processor will be used. Passing `half` could be ideal to be able to use your machine for other tasks while this module is searching for the positive matches. `full` is for using all processors. 
- `treshold`: Opional, default is set to 1000. This will be used to downscale images if one of their dimension is bigger than the treshold before comparison. It impacts the speed of the process massively. It is better to try different values and see the result.
- `find_pictures` method will return the list of positively identified images.

```python
>>> session1.find_pictures(verbose=True, multiprocess='half', treshold=600)    
# stats will be shown here
39 images will be analyzed.
...
IMG-20190517-WA0001.jpg	 2 face detected in      0.60 seconds ** POSITIVE MATCH **
IMG-20190820-WA0011.jpg	 2 face detected in      0.58 seconds
IMG-20190517-WA0001.jpg	 2 face detected in      0.60 seconds ** POSITIVE MATCH **
...
Finished within     13.61 seconds.
Person of interest is recognised in 10 images.
[`some/path/image_1`, `another/path/image_209`, ...]
```
#### Step 4: Saving the positevely identified pictures in a folder of choice.
- `image_list`: (optional) This is necessary if this method is used for saving a list of files in a folder of choice.
- `folder`: If not given `output` folder will be used.
- `copy`: If False, the positive matched image files will be moved from `input_stack` folder to `output` folder. If True, they will be copied (duplicated).
```python
>>> session1.save_result()
Positive matches have been copied to /home/tomasdem/repos/find_my_pictures/find_my_pictures/output/unique_name.
```
#### Step 5: Checking the results.
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
#### To search for a new person within the same mixed pictures:
- Call `session1.encode_poi(folder='path/to/new/poi_images', poi_identifier='John_Doe')` and continue from **Step 3**.
OR
- Replace all the images in the `input_sample` folder, annd repeat the steps starting from **Step 2**.
#### To search for the same person within different set of images:
- Call `session1.find_pictures(folder='path/to/new/mixed/images')` and continue from **Step 4**.
OR
- Replace the image files in the `input_stack` folder, and repeat the steps starting from **Step 3**.

#### For more details:
```python
>>> help(FindMyPictures)
```

#### TODO
- Add a simple GUI.
- Rewrite the module in C++.
