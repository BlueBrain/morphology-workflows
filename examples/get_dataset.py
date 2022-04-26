import pandas as pd
import os
import re
import json
from urllib.request import urlopen, Request
from pathlib import Path
import ssl

NEUROMORPHO_URL = "http://neuromorpho.org"


def get_swc_by_neuron_index(neuronIndex, folder="morphologies"):
    """Download a neuron by index and store it into a SWC file
    Keyword arguments:
    neronIndex -- the neuron index in the database

    Adapted from https://github.com/NeuroBox3D/neuromorpho/blob/master/rest_wrapper/rest_wrapper.py
    """

    url = "%s/api/neuron/id/%i" % (NEUROMORPHO_URL, neuronIndex)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = Request(url)
    response = urlopen(req, context=ctx)

    neuron_name = json.loads(response.read().decode("utf-8"))["neuron_name"]
    url = "%s/neuron_info.jsp?neuron_name=%s" % (NEUROMORPHO_URL, neuron_name)
    html = urlopen(url, context=ctx).read().decode("utf-8")
    p = re.compile(r"<a href=dableFiles/(.*)>Morphology File \(Standardized\)</a>", re.MULTILINE)
    m = re.findall(p, html)
    for match in m:
        file_name = match.replace("%20", " ").split("/")[-1]
        response = urlopen("%s/dableFiles/%s" % (NEUROMORPHO_URL, match), context=ctx)
        filename = folder / file_name
        with open(filename, "w") as f:
            f.write(response.read().decode("utf-8"))
            return filename


if __name__ == "__main__":
    brainRegion = "barrel"
    cell_type = "interneuron"
    numNeurons = 500

    url = "%s/api/neuron/select?q=brain_region:%s&size=%i" % (
        NEUROMORPHO_URL,
        brainRegion,
        numNeurons,
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = Request(url)
    response = urlopen(req, context=ctx)

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
