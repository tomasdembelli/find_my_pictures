import face_recognition as fr
import os
import sys
from PIL import Image, ImageDraw
from multiprocessing import Manager, Pool
import numpy as np
import imghdr
import time
from datetime import datetime
from shutil import copy, move
import cv2 as cv


class FindMyPictures():
    """Find pictures of a person within a folder of images."""
    treshold = 1000
    
    def __init__(self, input_sample=None, input_stack=None, output=None, verbose=False):
        """Initiate necessary folders.
        
        Absolute path has to be provided for the folders.
        
        input_sample: This folder should contain only person of interest (poi) and each image
        has to have only one face.
        input_stack: The pictures of poi will be searched within this folder.
        output: Pictures with positive match will be stored in this folder.
        
        The folders will be auto created in the directory from where the module initiated
        if they are not provided. Make sure the process has the right privileges for creating
        directories.
        """
        # Initializing folders
        self.initiate_verbose = verbose
        cwd = os.getcwd()
        if input_sample:
            if os.path.isdir(input_sample):
                self.input_sample = input_sample
            else:
                raise ValueError(f'{input_sample} is not a directory.')
        else:
            self.input_sample = os.path.join(cwd, 'input_sample')
            if not os.path.isdir(self.input_sample):
                os.mkdir(self.input_sample)
        if input_stack:
            if os.path.isdir(input_stack):
                self.input_stack = input_stack
            else:
                raise ValueError(f'{input_stack} is not a directory.')
        else:
            self.input_stack = os.path.join(cwd, 'input_stack')
            if not os.path.isdir(self.input_stack):
                os.mkdir(self.input_stack)
        if output:
            if os.path.isdir(output):
                self.output = output
            else:
                raise ValueError(f'{output} is not a directory.')
        else:
            self.output = os.path.join(cwd, 'output')
            if not os.path.isdir(self.output):
                os.mkdir(self.output)
        if self.initiate_verbose:
            print(f'Images in {self.input_sample} will be used for training.')
            print(f'Images in {self.input_stack} will be analyzed to identify person of interest.')
            print(f'Images with positive matches will be stored in {self.output}.')
            
    def validate_image_folder(self, folder):
        """Ensure given folder contains image files and only the image files are used for analysis."""
        if os.path.isdir(folder):
            img_files = []
            file_list = os.listdir(folder)
            for item in file_list:
                if not os.path.isdir(item):
                    full_name = os.path.join(folder, item)
                    try:
                        if imghdr.what(full_name):
                            img_files.append(full_name)
                        else:
                            print(f'{item} is not an image file')
                    except:
                        print(f'{item} is not an image file. {sys.exc_info()[0]}')
                else:
                    print(f'{item} is a folder.')
            if len(img_files) == 0:
                print (f'There is no image in {folder}')
        else:
            raise ValueError (f'{folder} is not a directory.')
        return img_files
            
    def encode_poi(self, folder=None, poi_identifier=None, accuracy=None):
        """Create 128-dimension face encodings of the poi by analyzing sample image files.

        accuracy: medium or high (default is optimum value for computing resources)
        The higher the accuracy is the longer the process takes to complete.
        
        folder: Absolute path of the folder containing the training data.
        This is necessary if this method is being used standalone.
        
        poi_identfier: This will be used to name the folder for the positive matches.
        It will be checked in find_pictures method. If not given, it will be auto populated.
        """   
        if folder:
            if os.path.isdir(folder):
                self.sample_img = self.validate_image_folder(folder)
            else:
                raise ValueError(f'{folder} is not a valid directory.')
        elif hasattr(self, 'input_sample'):
            self.sample_img = self.validate_image_folder(self.input_sample)
        else:
            raise ValueError('Absolute path for the folder containing images of poi is missing.')
        # This will be checked in find_pictures.
        if poi_identifier:
            self.positive_folder_name = poi_identifier 
        print("Analyzing the person of interest's face within sample images.")
        if accuracy == 'medium':
            num_jitters = 10
        elif accuracy == 'high':
            num_jitters = 30
        elif not accuracy:
            num_jitters = 1
        start = time.time()
        print(f'{len(self.sample_img)} images will be analyzed.')
        self.known_enc = []
        for img in self.sample_img:
            known_enc = self._encode_img(img, num_jitters=num_jitters)
            if len(known_enc) > 1:
                print(f"Sample image {img} (training data) can't contain more than 1 face.")
            else:
                self.known_enc.extend(known_enc)
        end = time.time()
        process_time = end - start
        print(f'Finished within {process_time} seconds.')
        if len(self.known_enc) < 1:
            raise ValueError('Person of interest face recognition is unsuccessfull.')
         
    
    def _encode_img(self, img, num_jitters=1, treshold=1000):
        """Resize image and return encodings.
        
        num_jitters: The higher is the more accurate but proportionaly slower.
        10 is 10 times slower than 1.
        default is 1.
        
        treshold: Maximum dimension in image size. Larger images will be downscaled to the treshold.
        defaul is 1000.
        """
        if hasattr(self, 'treshold'):
            treshold = self.treshold
        else:
            treshold = treshold
        img_load = cv.imread(img)
        # face_recognition works with RGB
        img_load = cv.cvtColor(img_load, cv.COLOR_BGR2RGB) 
        x, y, h = img_load.shape
        if max(x, y) > treshold:
            scale = float(1/(max(x, y)/treshold))
            img_load = cv.resize(img_load, None, fx=scale, fy=scale)
        img_enc = fr.face_encodings(img_load, num_jitters=num_jitters)
        return img_enc

    
    def _analyze_img(self, img, verbose=False):
        """Return True if there is a positive match in a given image.
        
        If it is called from find_pictures method, positive matched image will be
        copyied/moved into output folder. 
        """
        if not hasattr(self, 'known_enc'):
            raise ValueError('Sample images have not been analyzed yet.')
        if len(self.known_enc) < 1:
            raise ValueError('There is no trained data for the person of interest.')
        start = time.time()
        img_enc = self._encode_img(img)
        end_enc = time.time()
        result = False
        if len(img_enc) > 0:
            for face in img_enc:
                match = fr.compare_faces(self.known_enc, face)
                # Break the loop at first face match.
                if True in match:
                    result = True
                    if hasattr(self, 'copy'):
                        if self.copy:
                            copy(img, self.positive_folder)
                        else:
                            move(img, self.positive_folder)
                        self.match_count.value += 1
                    break
        if (hasattr(self, 'find_verbose') and self.find_verbose) or verbose:
            status = [f'{img.split("/")[-1]}\t']
            if len(img_enc) > 0:
                status.append(f'{len(img_enc)} face detected in {(end_enc - start):9.2f} seconds')
            else:
                status.append(f'No face detected in {(end_enc - start):9.2f} seconds')
            if result:
                status.append(f'** POSITIVE MATCH **')
            print(' '.join(status))
        return result
    
                      
    def find_pictures(self, folder=None, multiprocess=None, verbose=False, copy=True, treshold=1000):
        """Looks for the person of interest in the input stack images.
        
        copy: If True, positive matched images will be copied into output folder. 
              If False, they will be moved to the output folder.
              
        multiprocess: full --> All processors will be used.
                      half --> Half of the processors will be used.
                      default --> 1 processor will be used.
                      
        folder: Absolute path for the folder containing the mixed images. This is necessary if this
        method is being used standalone.
        
        treshold: Maximum dimension in image size. Larger images will be downscaled to the treshold.
        defaul is 1000. This will be used in _analyse_img method.
        """
        if not hasattr(self, 'known_enc'):
            raise ValueError('Sample images have not been analyzed yet.')
        if len(self.known_enc) < 1:
            raise ValueError('There is no trained data for the person of interest.')
        if folder:
            if os.path.isdir(folder):
                self.stack_images = self.validate_image_folder(folder)
            else:
                raise ValueError(f'{folder} is not a valid directory.')
        elif hasattr(self, 'input_stack'):
            self.stack_images = self.validate_image_folder(self.input_stack)
        else:
            raise ValueError('Absolute path for the folder containing mixed images is missing.')
        self.treshold = treshold
        self.find_verbose = verbose
        if len(self.input_stack) < 1:
            raise ValueError(f'There is no image in the {self.input_stack} folder.')
        if hasattr(self, 'positive_folder_name'):
            positive_folder_name = self.positive_folder_name
        else:
            positive_folder_name = ''.join(['Positive_Match', '_', datetime.now().strftime('%Y_%m_%d_%H:%M:%S')])
        positive_folder = os.path.join(self.output, positive_folder_name)
        if not os.path.isdir(positive_folder):
            os.mkdir(positive_folder)
        self.positive_folder = positive_folder
        self.copy = copy    # False means moving images to positive match folder.
        print(f'{len(self.stack_images)} images will be analyzed.')
        start = time.time()
        cpu_num = os.cpu_count()
        if multiprocess == 'full':
            use_cpu = cpu_num
        elif multiprocess == 'half':
            use_cpu = int(cpu_num/2)
        else:
            use_cpu = 1
        #self.treshold = [treshold for i in self.stack_images]
        with Manager() as manager:
            self.match_count = manager.Value('i', 0)
            with Pool(use_cpu) as pool:
                pool.map(self._analyze_img, self.stack_images)
                end = time.time()
                self.process_time = end - start
                print(f'Finished within {self.process_time:9.2f} seconds.')
                if self.match_count.value > 0:
                    print(f'Person of interest is recognised in {self.match_count.value} images.')
                    print(f'Positive matches have been stored in {self.positive_folder}.')
                # Delete following attributes to decouple _analyze_img method
                del self.copy
                del self.find_verbose
