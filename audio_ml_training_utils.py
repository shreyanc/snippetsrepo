import os
import pickle

def compute_envelope(self, x, k=201):
    kernel_size = k
    abs_x = torch.abs(x)
    max_envelope = F.max_pool1d(abs_x, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
    filtered_envelope = F.avg_pool1d(max_envelope, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
    return filtered_envelope

def cache_envelope(audio_file, k, envelope, cache_dir):
    """
    Caches the calculated envelope data for an audio file with a specified smoothing factor 'k'.
    
    Args:
        audio_file (str): The path to the input audio file.
        k (int): The smoothing factor parameter for envelope calculation.
        envelope (numpy.ndarray): The calculated envelope data to be cached.
        cache_dir (str): The directory where the cached data will be stored.
        
    Returns:
        None
    """
    # Create the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # Create the cache file path based on the audio file name and 'k'
    cache_file = os.path.join(cache_dir, f'{os.path.basename(audio_file)}_k{k}.pkl')
    
    # Serialize and save the envelope data to the cache file
    with open(cache_file, 'wb') as f:
        pickle.dump(envelope, f)

def load_cached_envelope(audio_file, k, cache_dir):
    """
    Loads the cached envelope data for a specific audio file and smoothing factor 'k'.
    
    Args:
        audio_file (str): The path to the input audio file.
        k (int): The smoothing factor parameter used for envelope calculation.
        cache_dir (str): The directory where the cached data is stored.
        
    Returns:
        envelope (numpy.ndarray or None): The cached envelope data if available, or None if not found.
    """
    # Create the cache file path based on the audio file name and 'k'
    cache_file = os.path.join(cache_dir, f'{os.path.basename(audio_file)}_k{k}.pkl')
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        # Deserialize and load the envelope data from the cache file
        with open(cache_file, 'rb') as f:
            envelope = pickle.load(f)
        return envelope
    else:
        # Return None if the cache file doesn't exist
        return None
