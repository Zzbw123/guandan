from _bootstrap import ROOT
import os,sys,unittest
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['RUN_P3F_CUDA']='1'
os.environ['RUN_P3G_CUDA']='1'
os.environ['RUN_P5F_CUDA']='1'
if __name__=='__main__':
    pattern='test_*.py' if '--all' in sys.argv else 'test_p5f*.py'
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
