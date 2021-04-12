import numpy as np 
from collections import Counter
import itertools


np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

docs = [
	"君不见 黄河之水天上来 奔流到海不复回",
	"君不见 高堂明镜悲白发 朝如青丝暮成雪",
	"人生得意须尽欢 莫使金樽空对月",
	"天生我材必有用 千金散尽还复来"
]