import os
from modules.util import get_filenames_from_folder, remove_extension, remove_path

def change_label_in_line(text, curr_labels:list, new_labels:list):
    words = text.split()
    if words[0] not in curr_labels:
        print(f"Class {words[0]} not in referenced labels {curr_labels}")
        return None
    words[0] = new_labels[curr_labels.index(words[0])]
    new_text = ""
    for idx, word in enumerate(words):
        new_text += word+" "
    return new_text[:-1]

def change_labels(labels_folder, curr_labels:list, new_labels:list, new_labels_folder="new_labels"):
    '''
    Modify the chosen labels of a dataset for new ones.
    
    Not listing a label in curr_labels will effectively remove it from the dataset

        - labels_folder: path to folder with images labels files
        - curr_labels: list of dataset labels to rename
        - new_labels: list of new labels
        - new_labels_folder: (optional) path to new images images labels folder
    '''
    labels = get_filenames_from_folder(labels_folder)
    for label_file in labels:
        with open(f"{label_file}", "r") as file:
            new_lines = []
            new_label_file = None
            
            for line in file.readlines():
                if line.split()[0] in curr_labels:
                    if not os.path.exists(new_labels_folder):
                        os.makedirs(new_labels_folder)
                    new_label_file = f"{new_labels_folder}/{remove_path(label_file)}"
                    new_line = change_label_in_line(line, curr_labels, new_labels)
                    new_lines.append(new_line)
            
            if new_label_file:
                with open(new_label_file, "x") as new_file:
                    # for id, line in enumerate(file.readlines()):
                    #     new_line = change_label_in_line(line, curr_labels, new_labels)
                    #     new_file.write(new_line)
                    new_file.writelines(new_lines)
                
def delete_images_without_label(images_folder, labels_folder):
    img_files = get_filenames_from_folder(images_folder)
    label_files = get_filenames_from_folder(labels_folder)
    
    for img in img_files:
        if f"{labels_folder}/{remove_path(remove_extension(img))}.txt".replace("\\","/") not in label_files:
            os.remove(img)