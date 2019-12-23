import torch
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

        self.image_as_a_graph_config = {
            'D_S': model_config['D_S'],
            'D_R': model_config['D_R'],
            'D_E': model_config['D_E'],
            'D_X': model_config['D_X']
        }

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
            """
            :param O: Matrix of dim batch_size, D_S * N_o
            :param R_s: Matrix of dim batch_size, N_o * N_R
            :param R_r: Matrix of dim batch_size, N_o * N_R
            :param R_a: Matrix of dim batch_size, D_R * N_R
            :return: A concatenation of OR_s, OR_r and R_a, which has a dimension of batch_size, (2D_S + D_R) * N_R
            """
            M1 = torch.matmul(O, R_s)
            M2 = torch.matmul(O, R_r)
            M3 = R_a

            # Concatenate along the second dimension
            return torch.cat([M1, M2, M3], dim=1)

        def get_e_hat(E, R_r):
            """
            :param E: Matrix of dim batch_size * D_E * N_R
            :param R_r: Matrix of dim batch_size * N_R * N_o
            :return: Matrix of batch_size * D_E * N_R
            """
            return torch.matmul(E, R_r)

        def aggregate(O, X, E_hat):
            """
            :param O: Matrix of dim batch_size * D_S * N_o
            :param X: Matrix of dim batch_size * D_X * N_o
            :param E_hat: Matrix of dim batch_size * D_E * N_o
            :return: Matrix of dim batch_size * (D_S + D_X + D_E) * N_o
            """
            return torch.cat([O, X, E_hat], dim=1)

        O, R_s, R_r, R_a, X = imagine_a_graph(x, self.image_as_a_graph_config)
        B = m_of_G(O, R_s, R_r, R_a)
        E = self.phi_r(B)
        E_hat = get_e_hat(E, R_r)
        C = aggregate(O, X, E_hat)
        P = self.phi_o(C)
        scores = self.scores(P)
        probs = self.probs(scores)

        return probs
