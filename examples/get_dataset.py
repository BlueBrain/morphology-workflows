"""Script to fetch morphologies used in examples."""
import json
import re
import ssl
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd

NEUROMORPHO_URL = "http://neuromorpho.org"


def get_url(url):
    """Open a URL without SSL verification."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    req = Request(url)
    response = urlopen(req, context=ctx)  # pylint: disable=consider-using-with
    return response


def get_swc_by_neuron_index(neuronIndex, folder="morphologies"):
    """Download a neuron by index and store it into a SWC file.

    Args:
        neuronIndex(int): the neuron index in the database.
        folder(str): The folder in which the morphologies are written.

    Adapted from https://github.com/NeuroBox3D/neuromorpho/blob/master/rest_wrapper/rest_wrapper.py
    """
    url = f"{NEUROMORPHO_URL}/api/neuron/id/{neuronIndex}"
    response = get_url(url)

    neuron_name = json.loads(response.read().decode("utf-8"))["neuron_name"]
    url = f"{NEUROMORPHO_URL}/neuron_info.jsp?neuron_name={neuron_name}"
    html = get_url(url).read().decode("utf-8")
    p = re.compile(r"<a href=dableFiles/(.*)>Morphology File \(Standardized\)</a>", re.MULTILINE)
    m = re.findall(p, html)
    for match in m:
        file_name = match.replace("%20", " ").split("/")[-1]
        response = get_url(f"{NEUROMORPHO_URL}/dableFiles/{match}")
        filename = folder / file_name
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.read().decode("utf-8"))
        print("Fetched", filename)
        return filename


def main():
    """Select neurons and download them from the NeuroMorpho database."""
    brainRegion = "barrel"
    numNeurons = 500

    url = f"{NEUROMORPHO_URL}/api/neuron/select?q=brain_region:{brainRegion}&size={numNeurons}"
    response = get_url(url)

    neurons = json.loads(response.read().decode("utf-8"))
    df = pd.DataFrame(neurons["_embedded"]["neuronResources"])
    df["type"] = df["cell_type"].apply(lambda t: t[-1])
    df = df[df["type"] == "principal cell"]
    folder = Path("morphologies")
    folder.mkdir(exist_ok=True)
    for gid in df.index:
        filename = get_swc_by_neuron_index(df.loc[gid, "neuron_id"], folder=folder)
        df.loc[gid, "morph_path"] = filename

    df["morph_name"] = df["neuron_name"]
    df[["morph_path", "morph_name", "brain_region"]].to_csv("dataset.csv")


if __name__ == "__main__":
    main()
