from _bootstrap import ROOT
import sys,unittest
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_p5b*.py'))
    sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
