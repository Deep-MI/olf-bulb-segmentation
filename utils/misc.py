
# Copyright 2021 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import fnmatch
import logging


def locate_file(pattern, root):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    file_path=[]
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            file_path.append(os.path.join(path, filename))
    return file_path

def locate_dir(pattern, root):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    dir_path=[]
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for dirname in fnmatch.filter(dirs, pattern):
            dir_path.append(os.path.join(path, dirname))
    return dir_path


def create_exp_directory(exp_dir_name):
    """
    Function to create a directory if it does not exist yet.
    :param str exp_dir_name: name of directory to create.
    :return:
    """
    if not os.path.exists(exp_dir_name):
        try:
            os.makedirs(exp_dir_name)
            print("Successfully Created Directory @ {}".format(exp_dir_name))
        except:
            print("Directory Creation Failed - Check Path")
    else:
        print("Directory {} Exists ".format(exp_dir_name))


def setup_logger(filename="log.txt"):

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(filename)
    logger.addHandler(fh)

    return logger


