import torch 

def imagine_a_graph(img):
    """
    Parameters
    ----------
    img: an image of shape (batch_size, width_size, height_size, num_channels)

    Returns
    -------
    A bunch of matrices representing the image as a graph.
    O: a D_S * N_o dimensional matrix representing object states.
    R_s: a N_o*N_R dimensional binary matrix representing senders indexes.
    R_r: a N_o*N_R dimensional binary matrix representing recievers indexes.
    X: a D_X * N_o dimensional matrix representing the graph node attributes.
    """


    return 0