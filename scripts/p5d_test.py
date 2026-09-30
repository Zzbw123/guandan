from _bootstrap import ROOT
import sys,os,unittest
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_p5d*.py'))
    sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
