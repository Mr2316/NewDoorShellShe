import numpy as np 
from collections import Counter
import itertools


np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

docs = [
	"君不见 黄河之水天上来 奔流到海不复回",
	"君不见 高堂明镜悲白发 朝如青丝暮成雪"
]