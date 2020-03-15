if exist clean.bat (cd ..)
if exist dist (rmdir dist /S /Q)
python setup.py sdist
twine check dist/*
