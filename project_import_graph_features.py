import collections as col
import networkx as nx
import pandas as pd
import sys

def subfinder(iset, fset, fname, iname):
    '''Fast matching of package names in files paths'''
    if len(fset.intersection(iset)) == len(iset):
        if iname in fname:
            return True
    return False

def get_graph(filename):

    # Get correct import names
    imports_dict = {}
    files_dict = {}
    imports = col.defaultdict(list)

    # read imports
    with open(filename) as fd:
        for line in fd:
            sline = line.strip().split(":")
            if len(sline) < 3: continue # empty line
            node_name = sline[0].replace(".java","").replace("/",".")
            # handle "import static" case
            if "import static" in sline[2]:
                value = '.'.join(sline[2].replace("import static ","").replace(";","").split(".")[0:-1])
            else:
                value = sline[2].replace("import ","").replace(";","")

            imports_dict[value] = set(value.split("."))
            files_dict[node_name] = set(node_name.split("."))

            imports[node_name].append(value)

    # remap filenames to imports
    imports_fixed = col.defaultdict(list)
    for kf, vf in files_dict.items():
        for ki, vi in imports_dict.items():
            if subfinder(vi, vf, kf, ki):
                imports_fixed[ki] = imports[kf]
                break
        else:
            imports_fixed[kf] = imports[kf]

    G = nx.from_dict_of_lists(imports_fixed, nx.DiGraph())

    return G


def process_filename(filename):
    G = get_graph(filename)
    return process_graph(G)


def process_graph(G):
    n = {}
    # Out nodes
    n["out"] = dict(G.out_degree())

    # In nodes
    n["in"] = dict(G.in_degree())

    # Pagerank
    n["pr"] = nx.pagerank(G, alpha=0.9)

    # Hubs and auths
    n["h"], n["a"] = nx.hits(G, max_iter = 1000)

    a = pd.DataFrame.from_dict(n)
    return a

def main(filename):
    a = process_filename(filename)
    print(a.to_csv())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage %s filename"%(argv[0]))
        exit()
    main(sys.argv[1])
