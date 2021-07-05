import sklearn.isotonic
import scipy.optimize
import numpy as np

def calibrate(val_uncalibrated: np.array, val_calibrated: np.array, test_uncalibrated: np.array, method: str) -> np.array:
    if method == "isotonic_regression":
        
        calibrator = sklearn.isotonic.IsotonicRegression(out_of_bounds="clip", y_min=-1, y_max=1).fit(val_uncalibrated, val_calibrated)
        return calibrator.predict(test_uncalibrated)

    elif method == "std_scaling":
        
        def criterion(s):
            regularization = (len(val_uncalibrated) / 2) * np.log(s)
            overconfidence = ((val_calibrated ** 2) / (2 * (s ** 2) * (val_uncalibrated ** 2))).sum()
            loss = regularization - overconfidence
            return loss

        opt = scipy.optimize.minimize_scalar(criterion)
        s = opt.x
        return s * test_uncalibrated
        
    else:
        raise NotImplementedError