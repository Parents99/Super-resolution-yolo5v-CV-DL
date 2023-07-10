nome_file = "C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\CVDL\\datasetFocus\\val1.txt"  # Sostituisci con il nome del tuo file

with open(nome_file, 'r') as file:
    contenuto = file.read()

contenuto_modificato = contenuto.replace('dataset3', 'datasetFocus')

with open(nome_file, 'w') as file:
    file.write(contenuto_modificato)

print("Il carattere _ è stato rimosso dal file.")
