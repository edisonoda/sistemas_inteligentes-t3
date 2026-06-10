import os
import csv

from sklearn.cluster import DBSCAN
import numpy as _np

from collections import defaultdict

def main(map_file):
    victims_list = []
    clusters = []

    with open(map_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Pula a primeira linha (cabeçalho)

        for row in csvreader:
            x = int(row[0])  # coordenada x
            y = int(row[1])  # coordenada y
            obst = float(row[2])  # obstaculo
            id = int(row[3])   # victim id number
            idade = int(row[4])
            fc = int(row[5])    # freq cardiaca
            fr = int(row[6])    # freq respiratoria
            pas = int(row[7])   # pressao arterial sistolica
            spo2 = int(row[8])  # saturacao de oxigenio
            temp = float(row[9])  # temperatura corporal
            pr = int(row[10])    # pulso radial (0 ou 1)
            sg = int(row[11])    # sangramento (0 ou 1)
            fx = int(row[12])    # fratura exposta (0 ou 1)
            queim = int(row[13])  # queimardura (niveis)
            gcs = int(row[14])   # Coma - Glasgow scale
            avpu = int(row[15])  # estado de consciencia
            tri = int(row[16])  # Triagem START: 0 GRN, 1 YEL, 2 RED, 3 BLK
            sobr = float(row[17])  # Prob. de sobrevivencia

            if id != -1:
                victims_list.append((id, x, y, tri))

    coords = _np.array([[v[1], v[2]] for v in victims_list])
    tris = _np.array([v[3] for v in victims_list]).reshape(-1, 1)

    # Scale tri so it has comparable range to coordinates
    grid_scale = 94
    tri_scale = float(grid_scale) * 0.1
    features = _np.hstack([coords, tris * tri_scale])

    # eps chosen as a small fraction of grid diagonal
    diag = (_np.sqrt(grid_scale ** 2 + grid_scale ** 2))
    eps = max(1.0, diag * 0.05)

    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(features)

    groups = defaultdict(list)
    for (v, x, y, tri), lbl in zip(victims_list, labels):
        groups[int(lbl)].append((v, x, y, tri))

    for lbl, members in groups.items():
        # compute centroid and cluster tri (max severity)
        xs = [m[1] for m in members]
        ys = [m[2] for m in members]
        tris_m = [m[3] for m in members]
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        cluster_tri = max(tris_m)
        cluster_seqs = [m[0] for m in members]
        clusters.append({
            'members': cluster_seqs,
            'centroid': centroid,
            'tri': int(cluster_tri)
        })

    print(f"Created {len(clusters)} clusters using DBSCAN")

    # Write each cluster members to a file cluster_i.txt in the agent config folder
    for i, cl in enumerate(clusters):
        fname = os.path.join(".", f"clusters/cluster_{i}.txt")
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


