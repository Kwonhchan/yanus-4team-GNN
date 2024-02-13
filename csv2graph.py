import pandas as pd
import numpy as np
import networkx as nx

class csv2graph():
    
    #csv2graph
    def c2g(path):
        df = pd.read_csv("Dataset\전완_수도권방문지데이터_1.csv")
    
    G = nx.graph()
        