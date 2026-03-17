from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine


def check_split_sum(sum, vals=[], required_sum=1):
    '''
    A helper function to check that all partition split percentages sum to a desired value (defaults to 1). Throws an exception if they do not.

    Input:
        sum: The sum of the values.
        vals: The values themselves. If provided, this will give a more descriptive error message should the sum not be equal to the required value.
        required_sum (default = 1): The value to which the sum should be equal.

    '''

    # Compare sum with required sum
    if sum != required_sum:
        message = f"Percentage splits must add up to 1 or 100"
        
        # Append additional info if values were provided
        if isinstance(vals, list) and len(vals) != 0:
            message += " (currently: "
        
            for i, value in enumerate(vals):
                if i != len(vals)-1:
                    message += f"{value} + "
                else:
                    message += f"{value} = {sum})"

        # Throw the exception with the informational message
        raise Exception(message)


def partitionSet(X, y, train_p=60, val_p=20, test_p=20, seed=1234):
    '''
    Input: 
        1. X: The data points to be partitioned.
        2. y: The labels corresponding to the data points.
        3. train_p (default = 20): The percentage of the data to allot to training.
        4. val_p (default = 20): The percentage of the data to allot to validation.
        5. test_p (default = 20): The percentage of the data to allot to testing.
        6. seed (default = 1234): The random seed with which to partition the data.
    Returns: Partitioned features and labels in the following order:
        Features:
        1. train_x: Training features.
        2. val_x: Validation features.
        3. test_x: Testing features.

        Labels:
        4. train_y: Training labels.
        5. val_y: Validation labels.
        6. test_y: Test labels.
    '''

    sum = train_p + val_p + test_p
    split_list = [train_p, val_p, test_p]

    # Make sure partitions make sense
    if sum != 100:
        check_split_sum(sum, split_list)
    else: 
        train_p /= 100
        val_p /= 100
        test_p /= 100
        sum = train_p + val_p + test_p
        check_split_sum(sum, split_list)

    # Perform first split
    dev_x, test_x, dev_y, test_y = train_test_split(X, y, test_size=test_p, random_state=seed)

    # Perform second split
    train_x, val_x, train_y, val_y = train_test_split(dev_x, dev_y, test_size=val_p / (train_p + val_p), random_state=seed)

    return train_x, val_x, test_x, train_y, val_y, test_y


def getDefaultWineSets(seed=1234):
    '''
    Input: 
        seed (default = 1234): The random seed with which to partition the data.
    Returns: The Wine data set features and labels partitioned and returned like so:
        Features:
        1. train_x: Training features.
        2. val_x: Validation features.
        3. test_x: Testing features.

        Labels:
        4. train_y: Training labels.
        5. val_y: Validation labels.
        6. test_y: Test labels.
    '''
    
    wine_X, wine_y = load_wine(return_X_y=True)

    return partitionSet(wine_X, wine_y, seed=seed)