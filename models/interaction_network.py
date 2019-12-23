import torch.nn as nn
from models.cnn_network import CNNNetwork
from utils.graph_representation import imagine_a_graph

class InteractionNetwork(nn.Module):

    class RelationModel(nn.Module):
        """
        Encodes the relationships between a set of graph nodes, relation one hot vectors and relation attributes.
        """
        def __init__(self, D_S, D_R, D_E):
            self.linear_transform = nn.Linear(D_S + D_S + D_R, D_E)

        def forward(self, B):
            return self.linear_transform(B)            
    
    class ObjectModel(nn.Module):
        """
        Encodes the each object's state vector, external effects and per object aggregation effects.
        """
        def __init__(self, D_S, D_X, D_E, D_P):
            self.linear_transform = nn.Linear(D_S + D_X + D_E, D_P)
        
        def forward(self, C):
            return self.linear_transform(C)

    def __init__(self, model_config):
        D_S = model_config['D_S']
        D_R = model_config['D_R']
        D_E = model_config['D_E']
        D_X = model_config['D_X']
        D_P = model_config['D_P']
        NUM_CLASSES = model_config['NUM_CLASSES']

        self.phi_r = self.RelationModel(D_S, D_R, D_E)
        self.phi_o = self.ObjectModel(D_S, D_X, D_E, D_P)
        self.scores = nn.Linear(D_P, NUM_CLASSES)
        self.probs = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x: input image of shape (batch_size, width, height, num_channels)
        we represent an image pixels as graph nodes and we find the relationships
        between pixels using an object and relation model. 

        Returns
        -------
        class scores. A tensor of shape (batch_size, num_classes)
        """

        def m_of_G(O, R_s, R_r, R_a):
            return 0

        def get_e_hat(E, R_r):
            return E

        def aggregate(O, X, E_hat):
            return 0

        O, R_s, R_r, R_a, X = imagine_a_graph(x)
        B = m_of_G(O, R_s, R_r, R_a)
        E = self.phi_r(B)
        E_hat = get_e_hat(E, R_r)
        C = aggregate(O, X, E_hat)
        P = self.phi_o(C)
        scores = self.scores(P)
        probs = self.probs(scores)
        
        return 0
        