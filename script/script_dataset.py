import os
import shutil

def split_and_move_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    i = 10000
    image_size = 1280

    for line in lines:
        line = line.strip()
        elements = line.split(' ')

        if len(elements) >= 2:
            image_path = elements[0]
            new_filename = str(i) 
            new_filepath = os.path.join('C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\CVDL\\datasetFocus\\images\\val\\', new_filename+'.jpg')

            # Sposta l'immagine nella nuova cartella
            shutil.move(image_path, new_filepath)

            # Crea il nuovo file di testo con il secondo contenuto
            label_path = 'C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\CVDL\\datasetFocus\\labels\\val\\'
            with open(label_path + new_filename+'.txt', 'w') as new_file:
                n = elements[1].split(',')
                w = float(n[2]) - float(n[0])
                h = float(n[3]) - float(n[1])
                n[0] = (float(n[0]) + w / 2) / 416
                n[1] = (float(n[1]) + h / 2) /416
                n[2] = w / 416
                n[3] = h / 416
                label = n[4] + ' ' + str(n[0]) + ' ' + str(n[1]) + ' ' + str(n[2]) + ' ' + str(n[3])
                new_file.write(label)

        else:
            print(f"Attenzione: la riga '{line}' non contiene abbastanza elementi.")
        i += 1

# Esempio di utilizzo
split_and_move_file('C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\CVDL\\datasetFocus\\valid_annotation_custom_people_focus.txt')