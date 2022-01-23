import numpy as np 
import face_collections as fcol

COLLECT_NAME: str = 'ProfFaces'
from pprint import pprint

print('Faces currently in the collection:')
pprint(fcol.list_faces(COLLECT_NAME))
print()

fcol.delete_collection(COLLECT_NAME)