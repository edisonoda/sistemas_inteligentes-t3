import os
import csv

def main(map_file):
    victims_list = []

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

    for victim in victims_list:
        print(f"ID: {victim[0]}, Coordenadas: ({victim[1]}, {victim[2]}), Triagem: {victim[3]}")


if __name__ == '__main__':
    print("------------------")
    print("--- INICIO ---")
    print("------------------")

    map_file = os.path.join(".", "map.csv")
    main(map_file)


