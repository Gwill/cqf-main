from typing import Tuple

#### Constants
PRICE_OPEN = "open"
PRICE_CLOSE = "close"
PRICE_HIGH = "high"
PRICE_LOW = "low"
VOLUME = "volume"
RETURN = "return"
TARGET_COL = f"{RETURN}_sign"
NP_RANDOM_SEED = 42
TS_RANDOM_SEED = 42
SKLEARN_RANDOM_SEED = 42
###


def print_dataset(ds, nested=False, numpy_format=False):
    """
    iterate though Dataset or Nested Dataset and print out elements/sequences
    """
    def iter_elements(tmp_ds):
        for j, element in enumerate(tmp_ds):
            if numpy_format:
                if isinstance(element, Tuple):
                    if j == 0:
                        print("-"*20)
                    print("x = ", element[0].numpy())
                    print("y = ", element[1].numpy())
                    print("-"*20)
                else:
                    print(element.numpy())
            else:
                if isinstance(element, Tuple):
                    print("x = ", element[0])
                    print("y = ", element[1])
                    print("-"*20)
                else:
                    print(element)
                    
    if nested:
        for i, nested_ds in enumerate(ds):
            if i ==0:
                print("="*50)
            print(f"Window {i}")
            iter_elements(nested_ds)
            print("="*50)
    else:
        iter_elements(ds)
                    