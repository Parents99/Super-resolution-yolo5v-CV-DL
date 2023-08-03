def add_jpg_to_paths(input_file, output_file):
    with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
        for line in f_input:
            new_line = line.strip() + ".jpg\n"
            f_output.write(new_line)

if __name__ == "__main__":
    input_file_path = "C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\CVDL\\datasetFocus\\val1.txt"  # Inserisci qui il percorso del file di input
    output_file_path = "C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\CVDL\\datasetFocus\\val2.txt"  # Inserisci qui il percorso del file di output

    add_jpg_to_paths(input_file_path, output_file_path)
    print("Operazione completata. Il nuovo file è stato creato con il nome 'output.txt'")