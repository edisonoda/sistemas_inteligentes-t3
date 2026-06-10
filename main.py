import os
import csv

from sklearn.cluster import DBSCAN
import numpy as _np

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

def main(map_file):
    victims_list = []
    clusters = []

    with open(map_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Pula a primeira linha (cabeçalho)

        for row in csvreader:
            x = int(row[0])  # coordenada x
            y = int(row[1])  # coordenada y
            id = int(row[3])   # victim id number
            tri = int(row[16])  # Triagem START: 0 GRN, 1 YEL, 2 RED, 3 BLK
            sobr = float(row[17])  # Prob. de sobrevivencia

            if id != -1:
                victims_list.append((id, x, y, tri, sobr))

    coords = _np.array([[v[1], v[2]] for v in victims_list])
    tris = _np.array([v[3] for v in victims_list]).reshape(-1, 1)

    # Scale tri so it has comparable range to coordinates
    features_raw = _np.hstack([coords, tris])
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features_raw)

    labels = DBSCAN(eps=0.2, min_samples=1).fit_predict(features)

    groups = defaultdict(list)
    for (v, x, y, tri, sobr), lbl in zip(victims_list, labels):
        groups[int(lbl)].append((v, x, y, tri, sobr))

    for lbl, members in groups.items():
        # compute centroid and cluster tri (max severity)
        xs = [m[1] for m in members]
        ys = [m[2] for m in members]
        tris_m = [m[3] for m in members]
        sobrs_m = [m[4] for m in members]

        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        cluster_tri = max(tris_m)
        cluster_seqs = [m[0] for m in members]

        sobr_min = min(sobrs_m)
        sobr_max = max(sobrs_m)
        sobr_mean = _np.mean(sobrs_m)
        sobr_std = _np.std(sobrs_m)

        clusters.append({
            'members': cluster_seqs,
            'centroid': centroid,
            'tri': int(cluster_tri),
            'sobr_min': float(sobr_min),
            'sobr_max': float(sobr_max),
            'sobr_mean': float(sobr_mean),
            'sobr_std': float(sobr_std)
        })

    print(f"Created {len(clusters)} clusters using DBSCAN")

    # Write each cluster members to a file cluster_i.txt in the agent config folder
    for i, cl in enumerate(clusters):
        fname = os.path.join(".", f"clusters/cluster_{i + 1}.txt")
        with open(fname, 'w') as fh:
            for seq in cl.get('members', []):
                fh.write(f"{seq}\n")
    if clusters:
        print(f"Wrote {len(clusters)} cluster files to clusters/ directory")


if __name__ == '__main__':
    print("------------------")
    print("--- INICIO ---")
    print("------------------")

    map_file = os.path.join(".", "map.csv")
    main(map_file)


