import pytest
from jkepler.rv import rv_unit_amplitude as getrv
import pkg_resources

def read_testdata_rv():
    testdata = pkg_resources.resource_filename('jkepler', 'data/tests/v723mon_s12_rv.csv')
    return testdata

def test_getrv():
    dat = read_testdata_rv()

if __name__ == '__main__':
    test_getrv()
