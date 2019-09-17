import face_recognition as fr
import os
from PIL import Image, ImageDraw
import multiprocessing
import numpy as np
import imghdr


class FindMyPictures():
    """Find pictures of interest within a folder of pictures."""
    def __init__(self, input_sample=None, input_stack=None,
                output=None, accuracy='medium'):
        
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
                else:
                    print(f'{file} is not an image file')
            if len(img_files) == 0:
                raise ValueError(f'There is no image in {folder}')
        else:
            raise ValueError(f'{folder} is not a directory.')
        return img_files
    
    def find_pictures(self):
        """Looks for the person of interest in the input stack images."""
        for img in self.stack_images:
            print(f'{img}', end='\t-->  ')
            img_load = fr.load_image_file(img)
            img_enc = fr.face_encodings(img_load)
            print(f'{len(img_enc)}', end='')
            for face in img_enc:
                match = fr.compare_faces(self.known_enc, face)
                if True in match:
                    print('MATCH', end='')
                else:
                    print('----', end='')
            print('')