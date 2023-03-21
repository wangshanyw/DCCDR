# DCCDR

Code for paper [Disentangled Contrastive Learning for Cross-Domain Recommendation].

Accepted by DASFAA 2023.


## Requirement
* Python 3.6
* PyTorch 1.10.2
* Numpy


## Files in the folder
- `OurModel.py`: our implementation of model
- `run.py`: model training and testing
- `data/`: four cross-domain recommendation tasks based on two widely used datasets Amazon and Douban
    - `Amazon/Cell_Elec/`
    - `Amazon/Movie_Music/`
    - `Douban/Movie_Book/`
    - `Douban/Music_Book/`
- `utils/`
    - `load_data.py`: auxiliary functions constructing training set and testing set for cross-domain scenario
    - `parser.py`: some parameters concerned with the model
    - `helpers.py`: functions to save the model
   
   
## Running the code
1. default: python run.py (Task: Amazon-Movie-Music)
    
   If you want to choose a certain task:
   python run.py --dataset=[chosen dataset] --domain_1=[chosen domain 1] --domain_2=[chosen domain 2]
   
