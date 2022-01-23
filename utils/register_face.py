from pathlib import Path  # for platform-specific path separators
from pprint import pprint
from glob import glob
# import graphical_utils as gu
# import image_loaders as img_ld

# Note: the face_collections module is in
#       the file face_collections.py included
#       in this project.
# Note: using as below simply means we can
#       reference anything from face_collections
#       by just saying fcol., so for example,
#       to use the create_collection function, we
#       would write fcol.create_collection
import face_collections as fcol

# since we will be referring to it
# frequently, create a global variable
# to store the name of the collection we are usingg
# Note: Python does NOT have constants
#       but we are using the programming convention
#       of all uppercase name to denote a constant
COLLECT_NAME: str = 'ProfFaces'

# variables for the different images directories
REF_FACE_DIR = 'reference_faces'  # contains images to be added to the collection

# check if the collection exists
# Note: strictly not necessary, since
#       create_collection will only create
#       the collection if it does not already
#       exist
print('Does ' + COLLECT_NAME + ' collection exist? ',
      'Yes' if fcol.collection_exists(COLLECT_NAME) else 'No')

# collection needs to exist first
print('Creating collection' + COLLECT_NAME)
fcol.create_collection(COLLECT_NAME)

# faces need to be added to the collection
print('Faces currently in the collection:')
pprint(fcol.list_faces(COLLECT_NAME))
print()

# checking to see if we need to add the faces,
# to the collection
if len(fcol.list_faces(COLLECT_NAME)) < 3:

    # We want to add a platform-specific
    # file separator for the filenames
    # so on Windows the file separator is \
    # whereas on Linux and Mac the file separator is /
    # here is the latest Pythonic way to do this

    # 1) create a path for the reference faces directory
    ref_images_dir = Path(REF_FACE_DIR)

    # create a list of filenames with the path
    # 2) notice the use of /
    # face_fnames = [str(ref_images_dir / fname)
    #                for fname in
    #                ['gaspar.jpg', 'small.jpg', 'ventura.jpg']]

    # alternatively, use glob to grab all JPEGs in the folder
    face_fnames = glob(REF_FACE_DIR + '/*.jpg')
    print('Here is the list of filenames: ', end='')
    pprint(face_fnames)
    for fname in face_fnames:
        fcol.add_face(COLLECT_NAME, fname)

    # print the face info in the collection
    print('Faces added to ' + COLLECT_NAME + ':')
    pprint(fcol.list_faces(COLLECT_NAME))


