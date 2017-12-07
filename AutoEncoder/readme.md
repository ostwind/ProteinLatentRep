To run VariationalAutoEncoder:
    1. if you haven't parsed RRM file into individual one-hot-encoded csv files:
        make sure you have a sibling data/ directory to AutoEncoder/
        PF00076_rp55.txt should be under data/

        run python preprocessing.py (~30 mins)

     2. run python AutoEncoder.py