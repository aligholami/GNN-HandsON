import torch


def imagine_a_graph(img, img_as_a_graph_config, device):
    """
    Parameters
    ----------
    img: an image of shape (batch_size, width_size, height_size, num_channels)
    img_as_a_graph_config: a dictionary consisting of matrix dimensions (see return section.)

    Returns
    -------
    A bunch of matrices representing the image as a graph.
    O: a D_S * N_o dimensional matrix representing object states.
    R_s: an N_o*N_R dimensional binary matrix representing senders indices.
    R_r: an N_o*N_R dimensional binary matrix representing receivers indices.
    R_a: an D_R*N_R dimensional matrix representing the edge attributes.
    X: a D_X * N_o dimensional matrix representing the graph node attributes.
    N_o: Number of nodes in the graph, representing the image.
    N_R: Number of directed relations in the graph.
    """

    D_S = img_as_a_graph_config['D_S']
    D_R = img_as_a_graph_config['D_R']

    image_shape = list(img.shape)
    batch_size = image_shape[0]

    width = img.shape[2]
    height = img.shape[3]
    N_o = width * height    # Number of spatials
    D_X = img.shape[1]     # Number of channels for each spatial

    # Initialize the graph nodes
    O = torch.FloatTensor(batch_size, D_S, N_o).uniform_(-1, 1).to(device)

    # Graph node attributes (fill them with the pixel values)
    X = img.view(-1, D_X, width*height)
    # Assume a complete graph (max number of edges will be N_o * (N_o - 1) / 2)
    N_R = N_o * (N_o - 1) / 2
    N_R_Directed = int(N_R * 2)

    # Senders and receivers indices
    relation_number = 0
    R_s = torch.zeros([batch_size, N_o, N_R_Directed]).to(device)
    R_r = torch.zeros([batch_size, N_o, N_R_Directed]).to(device)

    for s_idx in range(N_o):
        for r_idx in range(N_o):
            if s_idx != r_idx:
                R_s[:, s_idx, relation_number] = 1.0
                R_r[:, r_idx, relation_number] = 1.0
                relation_number += 1

    # Graph edge attributes
    R_a = torch.FloatTensor(batch_size, D_R, N_R_Directed).uniform_(-1, 1).to(device)

    return O, R_s, R_r, R_a, X, N_o, N_R_Directed
