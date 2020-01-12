import torch


def regions_as_a_graph(region_features, batch_indexes, num_max_regions, region_as_a_graph_config, device,):
    """
    :param region_features: a tensor of shape (batch_size * max_num_regions, feature_map_size) as a padded tensor.
    :param region_as_a_graph_config: a dictionary consisting of matrix dimensions.
    :param device: cuda or cpu.
    :return: A bunch of matrices representing the image as a graph.
    O: a D_S * N_o dimensional matrix representing object states.
    R_s: an N_o * N_R dimensional binary matrix representing senders indices.
    R_r: an N_o * N_R dimensional binary matrix representing receivers indices.
    R_a: an D_R * N_R dimensional matrix representing the edge attributes.
    X: a D_X * N_o dimensional matrix representing the graph node attributes.
    N_o: Number of nodes in the graph, representing the image.
    N_R: Number of directed relations in the graph.
    """

    D_S = region_as_a_graph_config['D_S']
    D_R = region_as_a_graph_config['D_R']
    N_o = num_max_regions
    batch_size = region_features.shape[0] // num_max_regions
    region_features = region_features.view(batch_size, num_max_regions, -1)
    D_X = region_features.shape[2]

    # Initialize the graph nodes
    O = torch.FloatTensor(batch_size, D_S, N_o).uniform_(-1, 1).to(device)

    # Graph Node Attributes
    X = region_features.view(batch_size, -1, num_max_regions).to(device)

    # In this case, X has the shape (128, 50000, 27)

    # Assume a complete graph (max number of edges will be N_o * (N_o - 1) / 2)
    N_R = N_o * (N_o - 1) / 2
    N_R_Directed = int(N_R * 2)

    # Senders and receivers indices
    R_s = torch.zeros([batch_size, N_o, N_R_Directed]).to(device)
    R_r = torch.zeros([batch_size, N_o, N_R_Directed]).to(device)

    for cnt, (ix_s, ix_e) in enumerate(batch_indexes):
        relation_number = 0
        image_specific_N_o = ix_e - ix_s
        for s_idx in range(image_specific_N_o):
            for r_idx in range(image_specific_N_o):
                if s_idx != r_idx:
                    R_s[cnt, s_idx, relation_number] = 1.0
                    R_r[cnt, r_idx, relation_number] = 1.0
                    relation_number += 1

    # Graph edge attributes
    R_a = torch.FloatTensor(batch_size, D_R, N_R_Directed).uniform_(-1, 1).to(device)

    return O, R_s, R_r, R_a, X, N_o, N_R_Directed


def img_as_a_graph(img, img_as_a_graph_config, device):
    """
    :param img: an image of shape (batch_size, width_size, height_size, num_channels)
    :param img_as_a_graph_config: a dictionary consisting of matrix dimensions (see return section.)
    :param device: cuda or cpu.
    :return: A bunch of matrices representing the image as a graph.
    O: a D_S * N_o dimensional matrix representing object states.
    R_s: an N_o * N_R dimensional binary matrix representing senders indices.
    R_r: an N_o * N_R dimensional binary matrix representing receivers indices.
    R_a: an D_R * N_R dimensional matrix representing the edge attributes.
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
    N_o = width * height  # Number of spatials
    D_X = img.shape[1]  # Number of channels for each spatial

    # Initialize the graph nodes
    O = torch.FloatTensor(batch_size, D_S, N_o).uniform_(-1, 1).to(device)

    # Graph node attributes (fill them with the pixel values)
    X = img.view(-1, D_X, width * height)
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
