import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Class name to label mapping
class_map = {
    'vehicle': 0,
    'walker': 1,
    'tree': 2
}

xml_folder = Path(r"C:\label_xml")    # folder with your XML files
output_folder = Path(r"C:\label_txt")   # folder to save YOLO txt files
output_folder.mkdir(exist_ok=True)

for xml_file in xml_folder.glob("*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower()
        if class_name not in class_map:
            continue
        
        label = class_map[class_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    txt_file = output_folder / (xml_file.stem + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))

print("Conversion to YOLO txt done!")
