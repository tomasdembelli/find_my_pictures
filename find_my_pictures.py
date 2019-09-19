import face_recognition as fr
import os
from PIL import Image, ImageDraw
from multiprocessing import Manager, Pool
import numpy as np
import imghdr
import time
from datetime import datetime
from shutil import copy, move



class FindMyPictures():
    """Find pictures of interest within a folder of pictures."""
    def __init__(self, input_sample=None, input_stack=None, output=None, verbose=False):
        """Explain arguments here."""
        # Initializing folders
        self.initiate_verbose = verbose
        cwd = os.getcwd()
        if input_sample:
            self.input_sample = input_sample
        else:
            self.input_sample = os.path.join(cwd, 'input_sample')
        if input_stack:
            self.input_stack = input_stack
        else:
            self.input_stack = os.path.join(cwd, 'input_stack')
        if output:
            self.output = output
        else:
            self.output = os.path.join(cwd, 'output')
        if self.initiate_verbose:
            print(f'Images in {self.input_sample} will be used for training.')
            print(f'Images in {self.input_stack} will be analyzed to identify person of interest.')
            print(f'Images with positive matches will be stored in {self.output}.')
            
    def analyze_poi(self, accuracy=None):
        """Explain"""    
        print("Analyzing the person of interest's face.")
        if accuracy == 'medium':
            num_jitters = 10
        elif accuracy == 'high':
            num_jitters = 30
        elif not accuracy:
            num_jitters = 1
        start = time.time()
        self.sample_img = self.validate_image_folder(self.input_sample)
        if len(self.sample_img) < 1:
            raise ValueError(f'There is no image in the {self.input_sample} folder.')
        else:
            print(f'{len(self.sample_img)} images will be analyzed.')
            self.known_enc = []
            for img in self.sample_img:
                known_img = fr.load_image_file(img)
                known_enc = fr.face_encodings(known_img, num_jitters=num_jitters)
                if len(known_enc) > 1:
                    print(f"Sample image {img} (training data) can't contain more than 1 face.")
                else:
                    self.known_enc.extend(known_enc)
        end = time.time()
        process_time = end - start
        print(f'Finished within {process_time} seconds.')
        if len(self.known_enc) < 1:
            print('Person of interest face recognition is unsuccessfull.')
        
            
    def validate_image_folder(self, folder):
        """Ensure given folder contains image files."""
        if os.path.isdir(folder):
            img_files = []
            file_list = os.listdir(folder)
            for file in file_list:
                if imghdr.what(os.path.join(folder, file)):
                    img_files.append(os.path.join(folder, file))
                else:
                    print(f'{file} is not an image file')
            if len(img_files) == 0:
                raise ValueError(f'There is no image in {folder}')
        else:
            raise ValueError(f'{folder} is not a directory.')
        return img_files
    
    def analyze_img(self, img):
            """Return True if there is a positive match."""
            if len(self.known_enc) < 1:
                raise ValueError('There is no trained data for the person of interest.')
            img_load = fr.load_image_file(img)
            img_enc = fr.face_encodings(img_load)
            result = False
            if len(img_enc) > 0:
                for face in img_enc:
                    match = fr.compare_faces(self.known_enc, face)
                    # Break the loop at first match. No need to check other faces.
                    if True in match:
                        result = True
                        if self.copy:
                            copy(img, self.positive_folder)
                        else:
                            move(img, self.positive_folder)
                        self.match_count.value += 1
                        break
            if self.find_verbose:
                status = [f'{img.split("/")[-1]}\t']
                if len(img_enc) > 0:
                    status.append(f'{len(img_enc)} face detected')
                else:
                    status.append('No face detected')
                if result:
                    status.append('** POSITIVE MATCH **')
                print(' '.join(status))
            return result
    
    def find_pictures(self, verbose=False, copy=True):
        """Looks for the person of interest in the input stack images."""
        if len(self.known_enc) < 1:
            raise ValueError('There is no trained data for the person of interest.')
        self.stack_images = self.validate_image_folder(self.input_stack)
        if len(self.input_stack) < 1:
            raise ValueError(f'There is no image in the {self.input_stack} folder.')
        self.find_verbose = verbose
        positive_folder_name = ''.join(['Positive_Match', '_', datetime.now().strftime('%Y_%m_%d_%H-%M')])
        positive_folder = os.path.join(self.output, positive_folder_name)
        os.mkdir(positive_folder)
        if os.path.isdir(positive_folder):
            self.positive_folder = positive_folder
        else:
            raise ValueError(f'{positive_folder} is not a valid directory.')
        self.copy = copy    # False means move images to positive match folder.
        print(f'{len(self.stack_images)} images will be analyzed.')
        start = time.time()
        with Manager() as manager:
            self.match_count = manager.Value('i', 0)
            with Pool() as pool:
                pool.map(self.analyze_img, self.stack_images)
                end = time.time()
                self.process_time = end - start
                print(f'Finished within {self.process_time} seconds.')
                if self.match_count.value > 0:
                    print(f'Person of interest is recognised in {self.match_count.value} images.')
                    print(f'Positive matches have been stored in {self.positive_folder}.')
