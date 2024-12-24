def generateName(name, prefix='', suffix='', separator='_'):
    """ Standardized form of deriving a name with a prefix or a suffix with <separator> separating them. """

    if prefix:
        prefix = prefix + separator

    if suffix:
        suffix = separator + suffix
    return str(prefix) + str(name) + str(suffix)


def generateKey(parent, child, suffix=''):
    return parent + '-' + child + suffix

def generateCollectionKey(collection_number):
    return f"{collection_number:03d}"

def generateLabeledTopic(topic, collection_key=None, type='2d', suffix=''):
    """Returns a standarized string for labeled sensor msg topics.

    Args:
        topic (str): The topic of the sensor msg that is labeled.
        collection_key (str, optional): If there is one labeled topic per collection use the collection key. Defaults to None.
        type (str, optional): The type of labeled data, 2d or 3d. Defaults to '2d'.

    Returns:
        str: The topic used to publish labeled data. 
    """

    possible_types = ['2d', '3d']
    assert type in possible_types, "generateLabeledTopic type must be one of " + str(possible_types)

    if collection_key is None:
        return topic + '/labeled' + type + suffix
    else:
        return topic + '/c' + collection_key + '/labeled' + type + suffix
