from fnmatch import fnmatch
import requests
from tqdm.auto import tqdm


def zip_glob(zip_file, glob):
    return [fn for fn in zip_file.namelist() if fnmatch(fn, glob)]


def download(url, out_path, description="Download"):
    """
    Downloads a given file to disk.

    This function is adapted from the tqdm examples.
    It thus follows the MPL license: https://mozilla.org/MPL/2.0/.
    """
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(
        open(out_path, "wb"), "write",
        unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
        desc=description, total=int(response.headers.get('content-length', 0))
    ) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)
    fout.close()
