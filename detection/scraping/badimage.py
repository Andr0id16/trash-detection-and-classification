import os
import cv2
import imghdr
import warnings

warnings.filterwarnings("error")

# NOTE: source dir must be single level directory
def check_images(s_dir, ext_list):
    """
    traverse source directory and compare file extensions to those in acceptable list\n
    append paths of bad images to bad_images list
    """
    bad_images = []
    bad_ext = []
    s_list = os.listdir(s_dir)
    for file in s_list:
        f_path = os.path.join(s_dir, file)
        tip = imghdr.what(f_path)
        print(f_path)
        if ext_list.count(tip) == 0:
            print(f"{f_path}: not compatible type")
            bad_images.append(f_path)
        if os.path.isfile(f_path):
            try:
                img = cv2.imread(f_path)
                shape = img.shape
            except:
                print("file ", f_path, " is not a valid image file")
                bad_images.append(f_path)
        else:
            print("*** fatal error")

    return bad_images, bad_ext


source_dir = r"scraped_images"

# list of acceptable extensions
good_exts = ["jpg", "jpeg", "png"]

# remove files with other extensions
bad_file_list, bad_ext_list = check_images(source_dir, good_exts)
if len(bad_file_list) != 0:
    print("improper image files are listed below")
    for i in range(len(bad_file_list)):
        os.remove(bad_file_list[i])
        print(bad_file_list[i])
else:
    print(" no improper image files were found")
