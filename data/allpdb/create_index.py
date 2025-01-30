import os
import datetime

def parse_date(date):
    month, day, year = date.split("/")
    day = int(day)
    month = int(month)
    year = int(year)
    if year >= 30:
        year += 1900
    else:
        year += 2000
    return datetime.date(year=year, month=month, day=day)

def parse_resolution(resolution):
    if resolution in ("NOT", ""):
        return 999.0
    if "," in resolution:
        return max(*map(parse_resolution, resolution.split(",")))
    return float(resolution)

def read_entries(entries_path):
    entry_dict = dict()
    with open(entries_path, "rt") as f:
        fit = iter(f)
        header = next(fit)
        separator = next(fit)
        for line in fit:
            code, _, date, *_, resolution, exptype = line.strip().split("\t")
            date = parse_date(date)
            resolution = parse_resolution(resolution)
            entry_dict[code] = dict(
                date=date,
                resolution=resolution,
                exptype=exptype)
    return entry_dict

def read_clusters(cluster_path, clusters=None, entry_clusters=None):
    clusters = clusters or {}
    entry_clusters = entry_clusters or {}
    with open(cluster_path, "rt") as f:
        for line in f:
            representative, entry = line.strip().split("\t")
            representative_base = representative.split("_")[0].upper()
            entry_base = entry.split("_")[0].upper()
            if representative not in clusters:
                clusters[representative] = set()
            if representative_base not in entry_clusters:
                entry_clusters[representative_base] = dict()
            clusters[representative].add(entry)
            if entry_base not in entry_clusters[representative_base]:
                entry_clusters[representative_base][entry_base] = dict()
    for cluster, members in clusters.items():
        cluster_base = cluster.split("_")[0].upper()
        for member in members:
            member_base = member.split("_")[0].upper()
            if cluster not in entry_clusters[cluster_base][member_base]:
                entry_clusters[cluster_base][member_base][cluster] = []
            entry_clusters[cluster_base][member_base][cluster].append(member)
    return clusters, entry_clusters

def create_entry_index(entry_clusters, entries, npz_assemblies, filter):
    result = []
    for cluster, members in entry_clusters.items():
        filtered_cluster = []
        for member in members:
            metadata = entries[member]
            if filter(metadata) and (member in npz_assemblies):
                filtered_cluster.append(dict(member=member,
                                             assemblies=[
                                                assembly
                                                for assembly in npz_assemblies[member]
                                             ],
                                             clusters=[
                                                dict(
                                                    subcluster=subcluster,
                                                    members=submembers)
                                                for subcluster, submembers in members[member].items()],
                                             metadata=metadata))
        result.append(filtered_cluster)
    return result

def create_chain_index(chain_clusters, entries, npz_assemblies, filter):
    result = []
    for _, members in chain_clusters.items():
        filtered_cluster = []
        for member in members:
            member_base = member.split("_")[0].upper()
            chain = member.split("_")[1]
            if member_base not in entries:
                continue # skip obsolete structures
            metadata = entries[member_base]
            if filter(metadata) and (member_base in npz_assemblies):
                filtered_cluster.append(
                    dict(entry=member_base,
                         chain=chain,
                         assemblies=npz_assemblies[member_base],
                         metadata=metadata))
        if filtered_cluster:
            result.append(filtered_cluster)
    return result

def write_entry_index(path, aa_index, na_index):
    with open(path, "wt") as f:
        for idc, cluster in enumerate(aa_index):
            for member in cluster:
                for assembly in member["assemblies"]:
                    for subcluster in member["clusters"]:
                        for submember in subcluster["members"]:
                            f.write(f"AA,{idc},{member},{assembly},{subcluster['subcluster']},{submember}\n")
        for idc, cluster in enumerate(na_index):
            for member in cluster:
                for assembly in member["assemblies"]:
                    for subcluster in member["clusters"]:
                        for submember in subcluster["members"]:
                            f.write(f"NA,{idc},{member},{assembly},{subcluster['subcluster']},{submember}\n")

def write_chain_index(path, aa_index, na_index):
    with open(path, "wt") as f:
        f.write(f"kind,cluster,PDBID,chain,assembly path,resolution,date\n")
        for idx, cluster in enumerate(aa_index):
            for member in cluster:
                for assembly in member["assemblies"]:
                    f.write(f"AA,{idx},{member['entry']},{member['chain']},{assembly},{member['metadata']['resolution']},{member['metadata']['date']}\n")
        offset = idx
        for idx, cluster in enumerate(na_index):
            for member in cluster:
                for assembly in member["assemblies"]:
                    f.write(f"NA,{idx + offset},{member['entry']},{member['chain']},{assembly},{member['metadata']['resolution']},{member['metadata']['date']}\n")

def compute_index(path, start_date="12/31/90", cutoff_date="12/31/21", cutoff_resolution=4.0):
    def entry_filter(x):
        return (
            x["resolution"] <= cutoff_resolution
            and x["date"] >= parse_date(start_date)
            and x["date"] < parse_date(cutoff_date))

    npz_path = f"{path}/assembly_npz/"
    entry_path = f"{path}/entries.idx"
    cluster_aa = f"{path}/clusterSeqresAA_cluster.tsv"
    cluster_na = f"{path}/clusterSeqresNA_cluster.tsv"
    subfolders_npz = os.listdir(npz_path)
    available_npz = [
        name
        for subfolder in subfolders_npz
        for name in os.listdir(f"{npz_path}/{subfolder}")
    ]
    npz_assemblies = {}
    for name in available_npz:
        basename = name.split("-")[0].upper()
        if not basename in npz_assemblies:
            npz_assemblies[basename] = []
        npz_assemblies[basename].append(f"{npz_path}/{name[1:3]}/{name}")
    entries = read_entries(entry_path)
    aa_clusters, _ = read_clusters(cluster_aa)
    na_clusters, _ = read_clusters(cluster_na)
    aa_index = create_chain_index(aa_clusters, entries, npz_assemblies, entry_filter)
    na_index = create_chain_index(na_clusters, entries, npz_assemblies, entry_filter)
    return dict(AA=aa_index, NA=na_index)

if __name__ == "__main__":
    from flexloop.utils import parse_options
    opt = parse_options(
        "generate index for PDB dataset.",
        entries="entries.idx",
        cluster_aa="ClusterSeqresAA_cluster.tsv",
        cluster_na="ClusterSeqresNA_cluster.tsv",
        npz_dir="assembly_npz/",
        start_date="01/01/90",
        cutoff_date="12/31/21",
        cutoff_resolution=3.5,
        out_path="testindex"
    )

    def entry_filter(x):
        return (
            x["resolution"] <= opt.cutoff_resolution
            and x["date"] >= parse_date(opt.start_date)
            and x["date"] < parse_date(opt.cutoff_date))

    subfolders_npz = os.listdir(opt.npz_dir)
    available_npz = [
        name
        for subfolder in subfolders_npz
        for name in os.listdir(f"{opt.npz_dir}/{subfolder}")
    ]
    npz_basename = [name.split("-")[0] for name in available_npz]
    npz_assemblies = {}
    for name in available_npz:
        basename = name.split("-")[0].upper()
        if not basename in npz_assemblies:
            npz_assemblies[basename] = []
        npz_assemblies[basename].append(f"{opt.npz_dir}/{name[1:3]}/{name}")
    entries = read_entries(opt.entries)
    aa_clusters, aa_entry_clusters = read_clusters(opt.cluster_aa)
    na_clusters, na_entry_clusters = read_clusters(opt.cluster_na)
    aa_index = create_chain_index(aa_clusters, entries, npz_assemblies, entry_filter)
    na_index = create_chain_index(na_clusters, entries, npz_assemblies, entry_filter)
    write_chain_index(opt.out_path, aa_index, na_index)
