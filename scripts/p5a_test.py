from _bootstrap import ROOT
import sys
sys.path.insert(0, str(ROOT))
import unittest
if __name__ == '__main__':
    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_p5a*.py'))
    sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
