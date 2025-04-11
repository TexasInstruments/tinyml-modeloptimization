def get_8bit_layers(module):
    layers = []
    # layers += ['conv1', 'bn1', 'relu1']

    layers += ['pointwise2', 'bn22', 'relu22']
    layers += ['pointwise3', 'bn32', 'relu32']
    # layers += ['pointwise4', 'bn42', 'relu42']
    # layers += ['pointwise5', 'bn52', 'relu52']

    layers += ['depthwise2', 'bn21', 'relu21']
    layers += ['depthwise3', 'bn31', 'relu31']
    # layers += ['depthwise4', 'bn41', 'relu41']
    # layers += ['depthwise5', 'bn51', 'relu51']

    # layers += ['avgpool', 'flatten6']
    # layers += ['fc6']
    return layers

def get_4bit_layers(module):
    layers = []

    # layers += ['conv1', 'bn1', 'relu1']

    # layers += ['pointwise2', 'bn22', 'relu22']
    # layers += ['pointwise3', 'bn32', 'relu32']
    # layers += ['pointwise4', 'bn42', 'relu42']
    # layers += ['pointwise5', 'bn52', 'relu52']

    # layers += ['depthwise2', 'bn21', 'relu21']
    # layers += ['depthwise3', 'bn31', 'relu31']
    # layers += ['depthwise4', 'bn41', 'relu41']
    # layers += ['depthwise5', 'bn51', 'relu51']

    # layers += ['avgpool', 'flatten6']
    # layers += ['fc6']
    return layers

def get_2bit_layers(module):
    layers = []
    return layers