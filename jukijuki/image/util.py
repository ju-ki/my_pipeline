def get_file_path(image_id, config):
    """_summary_

    Args:
        image_id (string): image_id
        config (_type_): config
    Example usage:
        train_df = train_df['image_id'].apply(lambda x: get_file_path(config, x))

    Returns: image_dir + image_id
    """
    assert hasattr(config, "image_dir"), "Please create image_dir(string './') attribute"
    return config.image_dir + f"{image_id}.jpg"