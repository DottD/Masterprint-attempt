import os
import shutil
from _1_segment_and_crop import scan_dir

dbpath = "/Users/MacD/Databases/NIST-SD14-150"
_, files = scan_dir(dbpath, ".jpg")

newpaths = (os.path.join(os.path.dirname(F), os.path.basename(F).split('.')[0], os.path.basename(F))
	for F in files)
for oldpath, newpath in zip(files, newpaths):
	_dir = os.path.dirname(newpath)
	if not os.path.exists(_dir): os.makedirs(_dir)
	shutil.move(oldpath, newpath)