import face_recognition as fr
import os
from PIL import Image, ImageDraw
from multiprocessing import Manager, Pool
import numpy as np
import imghdr
import time


class FindMyPictures():
    """Find pictures of interest within a folder of pictures."""
    def __init__(self, input_sample=None, input_stack=None,
                output=None, verbose=False):
        """Explain arguments here."""
        self.verbose = verbose
        # Initializing folders
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
            
        self.sample_img = self.validate_image_folder(self.input_sample)
        print('Learning the face of person of interest.')
        self.known_enc = []
        for img in self.sample_img:
            known_img = fr.load_image_file(img)
            known_enc = fr.face_encodings(known_img, num_jitters=10)
            if len(known_enc) > 1:
                raise ValueError('Sample image can not contain more than 1 face.')
            self.known_enc.extend(known_enc)
            
        self.stack_images = self.validate_image_folder(self.input_stack)
            
    def validate_image_folder(self, folder):
        """Ensure given folder contains image files."""
        if os.path.isdir(folder):
            img_files = []
            file_list = os.listdir(folder)
            for file in file_list:
                if imghdr.what(os.path.join(folder, file)):
                    img_files.append(os.path.join(folder, file))
                elif self.verbose:
                    print(f'{file} is not an image file')
            if len(img_files) == 0:
                raise ValueError(f'There is no image in {folder}')
        else:
            raise ValueError(f'{folder} is not a directory.')
        return img_files
    
    def analyze_img(self, img):
            """Return True if there is a positive match."""
            img_load = fr.load_image_file(img)
            img_enc = fr.face_encodings(img_load)
            result = False
            if len(img_enc) > 0:
                for face in img_enc:
                    match = fr.compare_faces(self.known_enc, face)
                    # Break the loop at first match. No need to check other faces.
                    if True in match:
                        result = True
                        self.match_count.value += 1
                        break
            if self.verbose:
                status = [f'{img.split("/")[-1]}\t']
                if len(img_enc) > 0:
                    status.append(f'{len(img_enc)} face detected')
                else:
                    status.append('No face detected')
                if result:
                    status.append('** POSITIVE MATCH **')
                print(' '.join(status))
            return result
    
    def find_pictures(self):
        """Looks for the person of interest in the input stack images."""
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