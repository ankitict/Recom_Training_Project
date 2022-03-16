# E-commerce Items Recommendation Project
Recommendation Training Project

# Prerequisite
Download datasets zip file from Brazilian E-Commerce Public Dataset and place it in '../datasets' directory, unzip the .zip file (this should produce 9 .csv files)


# Setup instructions

1. Creating Python environment
This repository has been tested on Python 3.7.6.

2. Cloning the repository:
git clone https://github.com/ankitict/Recom_Training_Project.git

3. Navigate to the git clone repository.
cd Recom_Training_Project

4. Download raw data from the data source link and place in "datasets" directory

    Install virtualenv

    pip install virtualenv

    virtualenv ItemRecom

5. Activate it by running:
ItemRecom/Scripts/activate

6. Install project requirements by using:
pip3 install -r requirements.txt

# Steps To Train a Item Recommendation Model

1. Use main.py to preprocess, train and save files.
    python main.py
2. Use similar_items.pkl Model file to get Item Recommendation
