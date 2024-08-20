"""

       mean       std    median      5.0%     95.0%     n_eff     r_hat
      b[0]      0.37      0.15      0.41      0.05      0.53     47.26      1.00
      b[1]      1.00      0.04      0.99      0.95      1.06     42.63      1.03
      b[2]      0.37      0.14      0.41      0.11      0.55     47.87      1.00
       lna     -8.09      0.04     -8.09     -8.16     -8.02    193.06      1.00
       lnc     -3.39      0.12     -3.40     -3.57     -3.19    176.24      1.00
  lnjitter    -10.99      1.53    -10.83    -13.22     -8.74    123.14      1.00
   lnrp[0]     -2.62      0.02     -2.62     -2.65     -2.59     63.56      1.00
   lnrp[1]     -2.55      0.30     -2.63     -2.99     -2.08     47.49      1.03
   lnrp[2]     -2.31      0.02     -2.31     -2.34     -2.29     61.55      1.00
  meanflux      0.00      0.00      0.00     -0.00      0.00    491.11      1.00
 period[0]     45.16      0.00     45.16     45.16     45.16    247.26      1.00
 period[1]     85.32      0.00     85.32     85.32     85.32    435.84      1.00
 period[2]    130.18      0.00    130.18    130.18    130.18    403.32      1.00
        q1      0.48      0.16      0.45      0.23      0.72     77.97      1.00
        q2      0.27      0.12      0.24      0.10      0.44    124.30      1.00
     rstar      0.91      0.04      0.92      0.84      0.97     57.86      1.00
     t0[0]    159.11      0.00    159.11    159.11    159.11    233.15      1.00
     t0[1]    295.31      0.00    295.31    295.31    295.32    481.22      1.01
     t0[2]    212.04      0.00    212.04    212.04    212.04    402.02      1.00

Number of divergences: 0

"""

from jkepler.transit.transitfit import TransitFit
from jkepler.transit.transitfit import compute_prediction
from jkepler.tests.read_testdata import read_testdata_transit

def test_compute_prediction():

    time, flux, error = read_testdata_transit()
    import matplotlib.pyplot as plt
    plt.plot(time, flux)
    plt.show()

    #fluxmodel = TransitFit.compute_flux(t0, period, ecc, omega, b, a, rp, u1, u2)
    
if __name__ == "__main__":
    test_compute_prediction()