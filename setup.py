from setuptools import setup, find_packages

setup(
    name="face_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python==4.8.0.76',
        'numpy==1.23.5',
        'face-recognition==1.3.0',
        'face-recognition-models==0.3.0',
        'dlib==19.24.2',
        'pymongo==4.6.1'
    ]
) 