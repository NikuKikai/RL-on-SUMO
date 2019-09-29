from experiments import DoubleIsraelNumOfEpisodes

### disable some plt warnings.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
###

def main():
    exp = DoubleIsraelNumOfEpisodes()
    exp.train_default()

main()