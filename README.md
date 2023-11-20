Sick of using old MATLAB code from 2006 for your frequency area plots? This code provides an easy solution to 
convert your neural data from filetype to FRA plot for your PI. <br>

Other examples are in the examples/ folder.
Install the requirements.txt first for the dependencies:
```
pip install -r requirements.txt
```
Then run the code:
```
run_fra_pipeline_cruella.py
```

AFTER changing your paths. <br>

Please note this code is designed for TDT data from a 32 channel array (pip install tdt), but can easily be changed to suit
other file types as the base of this code is just using an array. <br>
Credits: the PCA cleaning algorithm and FRA plotting code was adapted from Stephen Town's code, which was originally written in MATLAB.