# God damn you gst for stealing the args
import sys
argstore = sys.argv
sys.argv = sys.argv[:1]
import gst
sys.argv = argstore
