# find_my_pictures
Find pictures of a person within a set of pictures. 

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

### Examples
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
>>> session1.train_poi(accuracy='medium')    # Pictures of the person of interest within the input_sample folder 
wil be analyzed.
# non  image files will be listed here
Analyzing the person of interest's face within sample images.
3 images will be analyzed.
Sample image ~/find_my_pictures/input_sample/IMG_20190901_150845.jpg (training data) can't contain more than 1 face.
Finished within 14.532799482345581 seconds.
>>> 
>>> session1.find_pictures(verbose=True, multiprocess='half', copy=False)    # Image files in the input_stack folder 
will be analzed ant the positive matched ones will be moved to the output folder.
# stats will be shown here
39 images will be analyzed.
obaba_biden.png  2 face detected
IMG-20190517-WA0001.jpg  2 face detected ** POSITIVE MATCH **
IMG_20190901_150845 (1).jpg      2 face detected ** POSITIVE MATCH **
...
Finished within 62.41517496109009 seconds.
Person of interest is recognised in 8 images.
Positive matches have been stored in ~/find_my_pictures/output/Positive_Match_2019_09_22_00-15.
```
