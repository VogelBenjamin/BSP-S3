from argument_Processing import Argument_Processing_Unit
from model_Managing import Model_Managing_Unit
import sys

def main(argv):
    
    agu = Argument_Processing_Unit()
    model_ref, data_ref = agu.verify_inputs(argv)
    if not model_ref: model_ref = "rsvm_model"
    model_manager = Model_Managing_Unit(model_ref)
    agu.delegate_task(model_manager, data_ref)

if __name__ == "__main__":
    main(sys.argv[1:])
    