# StudyClimate
A python program to study climatological data series

# Motivation
This repository surges as after years of collecting climatological data from my weather station and the need of make it simple to study the timeseries of climatological data.
The StudyClimate.py scripts allows the user to read one or multiple files containing climatological timeseries from one location and merges them into a single dataframe to an easier manipulation inside the script in order to obtain quantities of interest for analise data.

# Python environment
To launch the Python script, I recommend you to create a python environment for exclusive use of this script in where you will install just the necessary python libraries needed by the StudyClimate.py script.
You can do this with the "venv" command:

In Linux/MacOS:
python3 -m venv StudyClimate

In Windows:
py -m venv StudyClimate

Then activate the environment by typing
source env/bin/activate in your Linux or macOS  or .\env\Scripts\activate in Windows.

The next step is to install the necessary libraries, using the requirements.txt file included in this repository. You can do this as follows:

In Linux/macOS:
python3 -m pip install -r requirements.txt

In Windows:
py -m pip install -r requirements.txt

Now you can launch the script and experiment with your climatological data!
