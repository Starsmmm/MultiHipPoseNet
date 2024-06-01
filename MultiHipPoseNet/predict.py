from train import seed_everything
from eval.predict import predict
import os


# Test files for inference evaluation of training models (5-fold hierarchical validation)


file_path = os.getcwd()
if __name__ == "__main__":
    seed = 967
    seed_everything(seed)
    predict(file_path, 5,(False,0.02),True,['min_loss_epoch_weights0.pth','min_loss_epoch_weights1.pth','min_loss_epoch_weights2.pth','min_loss_epoch_weights3.pth','min_loss_epoch_weights4.pth'])
