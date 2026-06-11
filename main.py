import os
import csv

from sklearn.cluster import DBSCAN, KMeans
import numpy as _np

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, pairwise_distances

import matplotlib.pyplot as plt

def read_map(map_file):
    victims_list = []

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
    
    return victims_list

def params_dbscan(features):
    best_params = None

    parameters = [
        {'eps': 0.1, 'min_samples': 6},
        {'eps': 0.12, 'min_samples': 5},
        {'eps': 0.14, 'min_samples': 4},
        {'eps': 0.15, 'min_samples': 3},
    ]

    for params in parameters:
        labels = DBSCAN(**params).fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN with eps={params['eps']} and min_samples={params['min_samples']} found {n_clusters + 1} clusters")

        if n_clusters > 1:
            s_score = silhouette_score(features, labels)
        else:
            s_score = -1.0

        n_noise = list(labels).count(-1)
        coverage = (len(labels) - n_noise) / len(labels)

        print(f"\tNoise: {n_noise} | coverage: {coverage:.2f} | silhouette: {s_score:.2f}")

        if best_params is None or coverage > best_params['coverage']:
            best_params = {
                'params': params,
                'coverage': coverage,
                's_score': s_score,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'labels': labels
            }
    
    return best_params

def make_clusters(victims_list, labels):
    clusters = []

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

    return clusters

def cluster_dbscan(clusters):
    print(f"\nCluster\t\tSobr (min)\tSobr (max)\tSobr (med)\tDesvio")

    # Write each cluster members to a file cluster_i.txt in the agent config folder
    for i, cl in enumerate(clusters):
        print(f"Cluster {i + 1}:\t{cl.get('sobr_min')}\t\t{cl.get('sobr_max')}\t\t{cl.get('sobr_mean'):.2f}\t\t{cl.get('sobr_std'):.2f}")
        fname = os.path.join(".", f"clusters/cluster_{i + 1}.txt")
        with open(fname, 'w') as fh:
            for seq in cl.get('members', []):
                fh.write(f"{seq}\n")
    
    if clusters:
        print(f"Wrote {len(clusters)} cluster files to clusters/ directory")

def show_individual_points(features, labels, n_clusters):
    n_samples = features.shape[0]
    dists = pairwise_distances(features, metric='euclidean')

    a = _np.zeros(n_samples)
    b = _np.zeros(n_samples)

    for i in range(n_samples):
        # Calcula a(i) - média das distâncias para outros pontos de dados no mesmo cluster
        brothers = dists[i, labels == labels[i]]

        if len(brothers) == 1:
            a[i] = 0
        else:
            a[i] = _np.sum(brothers)/(len(brothers) - 1)

        # Calcula b(i) - menor média das distâncias para pontos de dados em outros clusters
        min_mean_dist = float('inf')
        for k in range(n_clusters):
            if k != labels[i]:
                # distancia media para todos os individuos do cluster k
                mean_dist = _np.mean(dists[i, labels == k])
                if mean_dist < min_mean_dist:
                    min_mean_dist = mean_dist

        # número de clusters calculado pelo número de labels de clusters
        b[i] = min_mean_dist

    # Calcula os escores de silhueta para cada ponto de dados
    silhouette_scores = (b - a) / _np.maximum(a, b)

    # Print os coeficientes individuais de silhueta
    print(f"\n(x, y, tri):\t\ta\t\tb\t\tsilhouette")
    for i, txt in enumerate(features):
        print(f"({features[i, 0]:.2f}, {features[i, 1]:.2f}, {features[i, 2]:.2f}):\ta = {a[i]:.2f}\tb = {b[i]:.2f}\ts = {silhouette_scores[i]:.3f}")

    # Calcula a silhueta média
    mean_silhouette = _np.mean(silhouette_scores)
    print(f"\nEscore de silhueta média para {n_clusters + 1} clusters: {mean_silhouette:.4f}")
    faixa_boa = _np.sum(silhouette_scores >= 0.7)
    faixa_raz = _np.sum((silhouette_scores >= 0.25) & (silhouette_scores < 0.7))
    faixa_ruim = _np.sum(silhouette_scores < 0.25)

    print("\nDistribuição dos escores de silhueta:")
    print(f"s(i) >= 0.7         (BOM)      : {faixa_boa}")
    print(f"0.25 <= s(i) < 0.7  (RAZOÁVEL) : {faixa_raz}")
    print(f"s(i) < 0.25         (RUIM)     : {faixa_ruim}")

def plot_dbscan(labels, clusters, coords):
    # Plot victims colored by cluster
    plt.figure(figsize=(12, 12))
    plt.axis('equal')

    plt.scatter(
        0,
        0,
        marker='s',
        s=200,
        color='red',
        label='Base'
    )

    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label

        if label == -1:
            # Noise points
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c='black',
                marker='x',
                s=30,
                label='Noise'
            )
        else:
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=30,
                label=f'Cluster {label}'
            )

    for i, cl in enumerate(clusters):
        cx, cy = cl['centroid']

        plt.scatter(
            cx,
            cy,
            marker='*',
            s=300,
            edgecolors='black'
        )

        plt.text(
            cx,
            cy,
            f'C{i+1}',
            fontsize=10,
            ha='center'
        )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('DBSCAN Victim Clusters')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_kmeans(coords, victims_list):
    MARGINAL_SSE = 1.0
    last_sse = float('inf')
    best_params = None

    scaler = MinMaxScaler()
    features = scaler.fit_transform(coords)

    print("\n\n")

    sse_values = []
    k_values = range(2, 13)

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=21,
            n_init=10
        )

        labels = kmeans.fit_predict(features)
        s_score = silhouette_score(features, labels)
        sse = kmeans.inertia_

        sse_values.append(sse)

        print(f"K={k:2d}\tSSE={sse:.4f}\tSilhouette={s_score:.4f}")

        if best_params is None and last_sse - sse < MARGINAL_SSE:
            best_params = {
                'k': k,
                's_score': s_score,
                'sse': sse,
                'labels': labels
            }
        
        last_sse = sse

    labels = best_params['labels']
    best_params.pop('labels')
    print('\nBest params:', best_params)

    clusters = make_clusters(victims_list, labels)
    print(f"\nCluster\t\tSobr (min)\tSobr (max)\tSobr (med)\tDesvio")
    for i, cl in enumerate(clusters):
        print(f"Cluster {i + 1}:\t{cl.get('sobr_min')}\t\t{cl.get('sobr_max')}\t\t{cl.get('sobr_mean'):.2f}\t\t{cl.get('sobr_std'):.2f}")

    plt.figure(figsize=(8,5))

    plt.plot(
        k_values,
        sse_values,
        marker='o'
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    plt.grid(True)

    plt.xticks(list(k_values))

    plt.show()


def main(map_file):
    victims_list = read_map(map_file)
    coords = _np.array([[v[1], v[2]] for v in victims_list])
    tris = _np.array([v[3] for v in victims_list]).reshape(-1, 1)

    # Normalizes tri and coords
    features_raw = _np.hstack([coords, tris])
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features_raw)

    # DBSCAN
    best_params = params_dbscan(features)
    labels = best_params['labels']
    best_params.pop('labels')
    print('\nBest params:', best_params)

    clusters = make_clusters(victims_list, labels)

    cluster_dbscan(clusters)
    show_individual_points(features, labels, best_params['n_clusters'])
    plot_dbscan(labels, clusters, coords)

    # K-Means
    calculate_kmeans(coords, victims_list)

if __name__ == '__main__':
    print("------------------")
    print("--- INICIO ---")
    print("------------------")

    map_file = os.path.join(".", "map.csv")
    main(map_file)


