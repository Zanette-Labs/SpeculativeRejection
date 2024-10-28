def get_batches(num_trajectories: int, batch_size: int) -> list[int]:
    full_batches = num_trajectories // batch_size
    batches: list[int] = [batch_size] * full_batches
    if num_trajectories % batch_size > 0:
        batches.append(num_trajectories % batch_size)
    return batches
